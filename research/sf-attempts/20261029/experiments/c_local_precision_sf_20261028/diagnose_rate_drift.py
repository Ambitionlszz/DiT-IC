"""Read-only CPU probes; no optimizer update or interaction with live training."""
import gc
import json
import math
from pathlib import Path
import sys
import time

REPO = Path('/data2/ls/code/DiT-IC-c-local-precision-sf-20261028')
sys.path.insert(0, str(REPO))
import torch
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, make_model
from models.bt_self_flow.local_precision import (
    LocalPrecisionTeacher, centered_token_loss, sample_local_control, precision_tokens,
)
from models.bt_self_flow.packet import LocalPrecisionControl, RateControl
from models.bt_self_flow.quantization_residual import c_sana_forward
from models.sana_self_flow import unwrap_sana_transformer

ROOT = Path('/data2/ls/checkpoints/c_local_precision_sf_20261028')
OUT = Path(__file__).with_suffix('.json')
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
config = load_config(ROOT/'configs/joint.json')
config['model']['checkpoint_blocks'] = False
report = dict(read_only=True, device='cpu', sample='first four sorted Kodak images, 512 center crops',
              limitation='small deterministic diagnostic, not a causal training ablation',
              codec_stats={}, gradients=[])

def save():
    OUT.write_text(json.dumps(report, indent=2)+'\n')

started = time.monotonic()
model = make_model(config, torch.device('cpu'))
checkpoint = read_checkpoint(ROOT/'joint/0012000.pt')
model.load_bt_state(checkpoint['raw'])
dataset = ImageDataset(config['data']['valid'], crop=512, training=False)
images = torch.stack([dataset[i] for i in range(4)])
with torch.no_grad():
    clean = model.encode_clean_latent(images)
teacher = LocalPrecisionTeacher(model)
teacher.load_ema_state_dict(checkpoint['teacher'])
report['images'] = [str(p) for p in dataset.paths[:4]]
report['model_load_seconds'] = time.monotonic()-started
print('model and latents ready', report['model_load_seconds'], flush=True)
save()

def cosine(gs, gr):
    zero = torch.tensor(0.,dtype=torch.float64)
    dot = sum(((a.double()*b.double()).sum() for a,b in zip(gs,gr) if a is not None and b is not None),zero)
    ns = sum((a.double().square().sum() for a in gs if a is not None),zero).sqrt()
    nr = sum((a.double().square().sum() for a in gr if a is not None),zero).sqrt()
    return dict(sf_norm=float(ns), rate_norm=float(nr), dot=float(dot),
                cosine=float(dot/(ns*nr).clamp_min(1e-30)))

model.train()
parameters = [(name, p) for name,p in model.codec.named_parameters()
              if p.requires_grad and name.startswith(('g_a.', 'g_s.', 'router.', 'analysis_precision.', 'synthesis_precision.'))]
for qa, qb in [(0.,1.), (1.,0.), (.2,.6), (.6,.2)]:
    tick = time.monotonic()
    torch.manual_seed(903)
    anchor = RateControl(qa)
    control, _ = sample_local_control(config, anchor, 12000, 0, 0,
        (2,clean.shape[-2]//2,clean.shape[-1]//2), 'cpu', quality_b=qb)
    source = model.codec.analyze(clean[:2], anchor)
    view = model.codec.quantize_analysis(source, control)
    delta = view['delta']
    _, feature, _, _ = c_sana_forward(unwrap_sana_transformer(model.DiT), view['mean'],
        model._fixed_text_condition(2, clean.device), model.self_flow_timestep(2, clean.device),
        precision_tokens=precision_tokens(model.precision_adapter,delta.log(),clean.shape[-2:]),
        capture_layer=model.student_layer, stop_layer=model.student_layer)
    from dataclasses import fields, replace
    detached = replace(source, **{f.name:getattr(source,f.name).detach() for f in fields(source)
                                  if torch.is_tensor(getattr(source,f.name))})
    with torch.no_grad():
        fine = model.codec.quantize_analysis(detached,
            LocalPrecisionControl(max(qa,qb),analysis_quality=qa), compute_likelihood=False)
        target = teacher(fine['mean'], model.fixed_text_embed,
            model.self_flow_timestep(2,clean.device),fine['delta'].log())
    mask = delta.detach()>fine['delta']*(1+1e-6)
    sf, fraction = centered_token_loss(feature,target,mask,clean.shape[-2:])
    wrong, _ = centered_token_loss(feature,target.roll(1,0),mask,clean.shape[-2:])
    shuffle, _ = centered_token_loss(feature,target[:,torch.arange(target.shape[1]-1,-1,-1)],mask,clean.shape[-2:])
    rate = sum(-p.log2().sum() for p in view['likelihoods'].values())/(2*512*512)
    native = model.codec.quantize_analysis(source, anchor)
    native_rate = sum(-p.log2().sum() for p in native['likelihoods'].values())/(2*512*512)
    base = (view['mean']-clean[:2]).square().mean()/model.distortion_scale.s_data.square()
    inputs = [source.y]+[p for _,p in parameters]
    gs = torch.autograd.grad(sf, inputs, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(rate, inputs, retain_graph=True, allow_unused=True)
    gn = torch.autograd.grad(native_rate, inputs, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(base, inputs, allow_unused=True)
    groups = {'source_y':cosine(gs[:1],gr[:1])}
    native_groups = {'source_y':cosine(gs[:1],gn[:1])}
    base_groups = {'source_y':cosine(gs[:1],gb[:1])}
    for prefix in ('g_a.', 'g_s.', 'router.', 'analysis_precision.', 'synthesis_precision.'):
        idx = [i+1 for i,(name,_) in enumerate(parameters) if name.startswith(prefix)]
        groups[prefix] = cosine([gs[i] for i in idx],[gr[i] for i in idx])
        native_groups[prefix] = cosine([gs[i] for i in idx],[gn[i] for i in idx])
        base_groups[prefix] = cosine([gs[i] for i in idx],[gb[i] for i in idx])
    record = dict(qa=qa,qb=qb,sf=float(sf),wrong_image_sf=float(wrong),
                  spatial_flip_sf=float(shuffle),valid_fraction=float(fraction),rate_est=float(rate),
                  gradients=groups, native_rate_gradients=native_groups, base_gradients=base_groups,
                  radial_gain_derivatives={'sf':float((gs[0]*source.y.detach()).sum()),
                    'mixed_rate':float((gr[0]*source.y.detach()).sum()),
                    'native_rate':float((gn[0]*source.y.detach()).sum())},
                  seconds=time.monotonic()-tick)
    report['gradients'].append(record)
    save()
    print(json.dumps(record),flush=True)
    del source,view,feature,target,fine,sf,wrong,shuffle,rate,inputs,gs,gr,gn,gb,native,native_rate,base
    gc.collect()

del teacher
gc.collect()
for step in (10000,12000):
    checkpoint = read_checkpoint(ROOT/f'joint/{step:07d}.pt')
    model.codec.load_state_dict(checkpoint['ema']['codec'])
    model.codec.eval()
    model.codec.update(force=True)
    report['codec_stats'][str(step)] = []
    for q in (0.,.4,1.):
        with torch.no_grad():
            source = model.codec.analyze(clean,RateControl(q))
            view = model.codec.quantize_analysis(source,RateControl(q),trace=True)
            symbols = torch.cat([part['symbols'].flatten() for part in view['trace']]).float()
            unique,counts = symbols.unique(return_counts=True)
            probability = counts.float()/counts.sum()
            record = dict(q=q,y_rms=float(source.y.square().mean().sqrt()),
                symbol_rms=float(symbols.square().mean().sqrt()),zero_fraction=float((symbols==0).float().mean()),
                marginal_symbol_entropy=float(-(probability*probability.log2()).sum()),
                main_est=float(-view['likelihoods']['y'].log2().sum()/(4*512*512)),
                hyper_est=float(-view['likelihoods']['h'].log2().sum()/(4*512*512)),
                allocation_std=float(source.allocation.std()),
                geometric_step=float(view['delta'].log().mean().exp()),
                coarse_mse=float((view['mean']-clean).square().mean()))
        report['codec_stats'][str(step)].append(record)
        print(json.dumps(dict(step=step,**record)),flush=True)
        save()
report['total_seconds'] = time.monotonic()-started
save()
