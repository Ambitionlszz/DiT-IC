"""Read-only forward diagnosis; never step an optimizer or change checkpoints."""
import gc
import json
from pathlib import Path
import sys

REPO = Path('/data2/ls/code/DiT-IC-c-local-precision-sf-20261028')
sys.path.insert(0, str(REPO))

import torch
from models.bt_self_flow.checkpoint import read_checkpoint, restore_rng
from models.bt_self_flow.data import CounterSampler, ImageDataset
from models.bt_self_flow.experiment import load_config, make_model
from models.bt_self_flow.quantization_residual import CResidualTeacher
from models.bt_self_flow.native_relational import spatial_relation, relation_distance
from models.sana_self_flow import unwrap_sana_transformer
from train_bt_self_flow import sample_control, seed_all

ROOT = Path('/data2/ls/checkpoints/c_native_relational_sf_20261027')
OUTPUT = Path(__file__).with_suffix('.json')
SOURCE = Path('/data2/ls/checkpoints/ditic-50k-warmups/c_precision_v2_lambda16_C_joint10k_20261023/0010000.pt')


def tensor_stats(value):
    dtype = str(value.dtype)
    value = value.detach().float()
    return dict(shape=list(value.shape), dtype=dtype, finite=bool(torch.isfinite(value).all()),
                max_abs=float(value.abs().max()), spatial_std=float(value.std(dim=1).mean()) if value.ndim == 3 else None)


def tensors(value):
    if torch.is_tensor(value) and value.is_floating_point() and value.numel():
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for item in value for v in tensors(item)]
    if isinstance(value, dict):
        return [v for item in value.values() for v in tensors(item)]
    return []


class FirstNonfinite(Exception):
    pass


seed_all(903)
config = load_config(ROOT/'configs/joint.json')
checkpoint = read_checkpoint(ROOT/'joint/paused.pt')
device = torch.device('cuda:0')
torch.cuda.set_device(device)
model = make_model(config, device)
model.load_bt_state(checkpoint['raw'])
model.train()
dataset = ImageDataset(config['data']['train'], config['data']['crop'])
sampler = iter(CounterSampler(len(dataset), config['train']['seed'], rank=0, world_size=2,
    batch_size=4, micro_batches_seen=checkpoint['micro_seen'], stream=23))
keys = [next(sampler) for _ in range(4)]
images = torch.stack([dataset[key] for key in keys]).to(device)
control = sample_control(config, checkpoint['step'], 0, (4, 8, 8), device)
report = dict(checkpoint=str(ROOT/'joint/paused.pt'), step=checkpoint['step'],
    micro_seen=checkpoint['micro_seen'], rank=0, batch=4, q=control.value,
    sample_keys=keys, data_fingerprint=dataset.fingerprint,
    no_optimizer_update=True, cases={})
history = []
handles = []
first = None


def hook(name):
    def record(module, args, output):
        global first
        values = tensors(output)
        if not values:
            return
        bad = any(not bool(torch.isfinite(v).all()) for v in values)
        if bad:
            first = dict(module=name, type=type(module).__name__,
                inputs=[tensor_stats(v) for v in tensors(args)],
                outputs=[tensor_stats(v) for v in values], preceding_modules=list(history))
            raise FirstNonfinite(name)
        history.append(dict(module=name, max_abs=max(float(v.detach().float().abs().max()) for v in values)))
        del history[:-8]
    return record


for name, module in model.named_modules():
    if not any(module.children()):
        handles.append(module.register_forward_hook(hook(name)))
restore_rng(checkpoint['rng_by_rank'][0])
try:
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        result = model(images, control, capture=True)
        report['cases']['latest_fp16'] = dict(feature=tensor_stats(result.feature), image=tensor_stats(result.image))
except FirstNonfinite:
    report['cases']['latest_fp16'] = first
finally:
    for handle in handles:
        handle.remove()
    handles.clear()
OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'latest_fp16':report['cases']['latest_fp16']}), flush=True)

if first:
    module = dict(model.named_modules())[first['module']]
    handles.append(module.register_forward_hook(lambda m, a, o: report['cases'].update(
        fp32_at_overflow_module=dict(inputs=[tensor_stats(v) for v in tensors(a)], outputs=[tensor_stats(v) for v in tensors(o)]))))
teacher = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer).to(device)
teacher.load_ema_state_dict(checkpoint['teacher'])
restore_rng(checkpoint['rng_by_rank'][0])
with torch.no_grad(), torch.autocast('cuda', enabled=False):
    result = model(images, control, capture=True)
    views = model.relational_views(result, control, teacher, config, checkpoint['step'], 0, 0)
    target = views.target
    relation = spatial_relation(result.feature)
    report['cases']['latest_fp32'] = dict(clean=tensor_stats(result.clean), coarse=tensor_stats(result.coarse),
        feature=tensor_stats(result.feature), image=tensor_stats(result.image),
        teacher_spatial_std=float(views.teacher_spatial_std),
        native_correct_relation=float(relation_distance(relation,target).mean()),
        native_wrong_relation=float(relation_distance(relation,target.roll(1,0)).mean()),
        teacher_to_wrong_image=float(relation_distance(target,target.roll(1,0)).mean()),
        teacher_to_cross_image_average=float(relation_distance(target,target.mean(0,keepdim=True).expand_as(target)).mean()))
    del result, views, target, relation
for handle in handles:
    handle.remove()
del teacher, checkpoint
gc.collect()
torch.cuda.empty_cache()
OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
source = read_checkpoint(SOURCE)
model.load_bt_state(source['raw'])
seed_all(903)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    result = model(images, control, capture=True)
    report['cases']['source_c10k_fp16_same_images'] = dict(clean=tensor_stats(result.clean),
        coarse=tensor_stats(result.coarse),feature=tensor_stats(result.feature), image=tensor_stats(result.image))
OUTPUT.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
