"""Replay a saved failure without changing research weights or data."""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, '/data2/ls/code/DiT-IC-bt-self-flow')
import torch
import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.data import CounterSampler, ImageDataset
from models.sana_self_flow import unwrap_sana_transformer

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--backward', action='store_true')
args = parser.parse_args()
torch.set_num_threads(1)
state = read_checkpoint(args.checkpoint)
config, step = state['config'], state['step']
root = Path('/data2/ls/checkpoints/bt_self_flow_v1')
report_path = root / config['model']['backend'] / 'joint'
failure = sorted(Path(args.checkpoint).parent.glob(f'nonfinite_{step:07d}_*.json'))
if not failure:
    failure = sorted(report_path.glob(f'nonfinite_{step:07d}_*.json'))
diagnostic = json.loads(failure[-1].read_text()) if failure else {'micro': 0, 'step': step}
if isinstance(diagnostic, dict) and 'ranks' in diagnostic:
    diagnostic = diagnostic['ranks'][0]
if isinstance(diagnostic, list):
    diagnostic = diagnostic[0]
print('diagnostic', diagnostic, flush=True)
micro = diagnostic['micro']
dataset = ImageDataset(config['data']['train'], config['data']['crop'])
sampler = CounterSampler(len(dataset), config['train']['seed'], batch_size=config['train']['batch_size'],
                         micro_batches_seen=state['micro_seen'], stream=23)
batch = config['train']['batch_size']
keys = list(itertools.islice(iter(sampler), micro * batch, (micro + 1) * batch))
image = torch.stack([dataset[key] for key in keys]).cuda()
control = trainer.sample_control(config, step, 0, (batch, 8, 8), torch.device('cuda'))
model = trainer.make_model(config, torch.device('cuda')).train()
model.load_bt_state(state['raw'])
loss_scale = state['scaler']['scale']
del state
with torch.no_grad(), torch.autocast('cuda', enabled=False):
    clean = model.encode_clean_latent(image)
    coarse = model.codec(clean, control)['mean']
    scale = model.distortion_scale(clean, coarse) if model.backend == 'D' else {}
trace = []


def summary(value):
    if isinstance(value, torch.Tensor):
        return {'dtype': str(value.dtype), 'shape': list(value.shape),
                'finite': bool(torch.isfinite(value).all()),
                'max_abs': float(value.detach().abs().max())}
    if isinstance(value, (tuple, list)):
        return [summary(v) for v in value]
    return None


class NonfiniteModule(Exception):
    pass


def finite(value):
    if isinstance(value, dict):
        return value.get('finite', True)
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return True


def hook(name):
    def record(module, inputs, output):
        item = {'module': name, 'class': type(module).__name__, 'inputs': summary(inputs), 'output': summary(output)}
        trace.append(item)
        if not finite(item['output']):
            raise NonfiniteModule(name)
    return record


sana = unwrap_sana_transformer(model.DiT)
handles = [module.register_forward_hook(hook(name)) for name, module in sana.named_modules()]
first = None
try:
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        model.predict(coarse, scale.get('log_rho'))
except NonfiniteModule as exc:
    first = str(exc)
finally:
    for handle in handles:
        handle.remove()
with torch.no_grad(), torch.autocast('cuda', enabled=False):
    reference, _ = model.predict(coarse, scale.get('log_rho'))
result = {'checkpoint': args.checkpoint, 'step': step, 'micro': micro, 'keys': keys,
          'first_nonfinite': first, 'coarse': summary(coarse), 'fp32_reference': summary(reference), 'trace': trace}
if args.backward:
    from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
    loss_fn = ImageLoss(**config['image_loss']).cuda().eval()
    attempts = []
    for _ in range(12):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            reconstructed = model(image, rate_control=control)
            loss, metrics = reconstruction_objective(reconstructed, image, control,
                                                     model.distortion_scale.s_data, loss_fn)
        assert torch.isfinite(loss), 'corrected AMP forward still failed'
        (loss * loss_scale).backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        ok = all(bool(torch.isfinite(g).all()) for g in gradients)
        attempts.append({'scale': loss_scale, 'gradients_finite': ok})
        if ok:
            break
        del reconstructed, loss, metrics, gradients
        loss_scale /= 2
    assert ok, 'no finite AMP backward at the saved failure'
    result['backward'] = {'attempts': attempts, 'loss': float(loss.detach()),
                          'metrics': {k: float(v.detach()) for k, v in metrics.items()},
                          'gradient_norms': {name: float(torch.stack([p.grad.float().norm() / loss_scale
                                            for p in module.parameters() if p.grad is not None]).norm())
                                             for name, module in [('router', model.codec.router),
                                                                  ('ga', model.codec.g_a)]},
                          'encoder_frozen': all(p.grad is None for p in model.vae.encoder.parameters()),
                          'peak_memory_bytes': torch.cuda.max_memory_allocated()}
Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'trace'}), flush=True)
print(json.dumps(trace[-8:]), flush=True)
