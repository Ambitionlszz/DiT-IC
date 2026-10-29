"""Read-only fixed-latent interventions and gradient probes, no optimizer step."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_HOME', '/data2/ls/checkpoints/torch-hub')
REPO = Path('/data2/ls/code/DiT-IC-c-context-sf-20261028')
sys.path.insert(0, str(REPO))
import torch
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.context_sf import ContextTeacher, context_distance
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, make_model
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import RateControl
from models.sana_self_flow import unwrap_sana_transformer

parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, required=True)
parser.add_argument('--gradients', action='store_true')
args = parser.parse_args()
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
ROOT = Path('/data2/ls/checkpoints/c_context_sf_20261028')
OUT = Path(__file__).with_name(f'diagnose_context_{args.step}.json')
report = dict(step=args.step, read_only=True, no_optimizer_update=True,
    sample='8 fixed Kodak 512 center crops; batch=4; FP32 EMA interventions; raw-state FP16 gradient probes',
    interventions=[], codec_stats=[], gradients=[], attention=[])
start = time.monotonic()

def save():
    report['elapsed_seconds'] = time.monotonic()-start
    OUT.write_text(json.dumps(report, indent=2)+'\n')

config = load_config(ROOT/'configs/joint.json')
config['model']['checkpoint_blocks'] = False
model = make_model(config, torch.device('cuda'))
checkpoint = read_checkpoint(ROOT/f'joint/{args.step:07d}.pt')
model.load_bt_state(checkpoint['ema'])
model.eval()
loss_fn = ImageLoss(**config['image_loss']).cuda()
data = ImageDataset(config['data']['valid'], crop=512, training=False)
indices = [0, 3, 6, 9, 12, 15, 18, 21]
report['images'] = [str(data.paths[i]) for i in indices]
images = torch.stack([data[i] for i in indices]).cuda()
with torch.no_grad():
    clean = model.encode_clean_latent(images)
print('model ready', args.step, time.monotonic()-start, flush=True)
save()

hooks = []
for index in (8,19,27):
    def hook(module, inputs, kwargs, output, index=index):
        context = kwargs.get('encoder_hidden_states')
        if context is None or context.shape[1] != 64:
            return
        query = inputs[0].float()
        gate = model.context_readout.gates[index-model.student_layer-1].item()
        report['attention'].append(dict(layer=index+1, gate=gate,
            query_rms=float(query.square().mean().sqrt()),
            context_output_rms=float(output.float().square().mean().sqrt()),
            gated_output_rms=float((gate*output.float()).square().mean().sqrt()),
            output_spatial_std=float(output.float().std(1, unbiased=False).square().mean().sqrt())))
    hooks.append(unwrap_sana_transformer(model.DiT).transformer_blocks[index].attn2.register_forward_hook(hook, with_kwargs=True))

with torch.no_grad():
    for q in (0., .4, 1.):
        control = RateControl(q)
        for offset in (0,4):
            target = images[offset:offset+4]
            z = clean[offset:offset+4]
            source = model.codec.analyze(z, control)
            view = model.codec.quantize_analysis(source, control, trace=True)
            symbols = torch.cat([p['symbols'].flatten() for p in view['trace']]).float()
            _, counts = symbols.unique(return_counts=True)
            p = counts.float()/counts.sum()
            report['codec_stats'].append(dict(q=q, offset=offset,
                y_rms=float(source.y.square().mean().sqrt()),
                symbol_rms=float(symbols.square().mean().sqrt()),
                zero_fraction=float((symbols==0).float().mean()),
                marginal_symbol_entropy=float(-(p*p.log2()).sum()),
                main_est=float(-view['likelihoods']['y'].log2().sum()/(4*512*512)),
                hyper_est=float(-view['likelihoods']['h'].log2().sum()/(4*512*512)),
                coarse_mse=float((view['mean']-z).square().mean()),
                allocation_std=float(source.allocation.std())))
            baseline = None
            for mode in ('native','off','wrong_image'):
                model.context_mode = mode
                begin = len(report['attention'])
                latent, image, context = model.repair(view['mean'], {'log_step':view['delta'].log()}, capture=True)
                for entry in report['attention'][begin:]:
                    entry.update(q=q, offset=offset, mode=mode)
                if mode == 'native':
                    baseline = image.clone()
                records = []
                for k in range(4):
                    distortion, metrics = loss_fn(image[k:k+1], target[k:k+1])
                    records.append(dict(d411=float(distortion), **{k:float(v) for k,v in metrics.items()}))
                centered = context-context.mean(1,keepdim=True)
                record = dict(q=q, offset=offset, mode=mode, metrics=records,
                    output_mse_vs_native=float((image-baseline).square().mean()),
                    context_rms=float(context.square().mean().sqrt()),
                    context_spatial_rms=float(centered.square().mean().sqrt()),
                    context_mean_rms=float(context.mean(1).square().mean().sqrt()))
                report['interventions'].append(record)
            save()
            print('intervention',args.step,q,offset,flush=True)
for hook in hooks:
    hook.remove()
model.context_mode='native'

if args.gradients:
    model.load_bt_state(checkpoint['raw'])
    model.train()
    model.checkpoint_blocks=True
    teacher=ContextTeacher(model).cuda()
    teacher.load_ema_state_dict(checkpoint['teacher'])
    parameters=tuple(model.sf_parameters())
    ids={id(p) for p in model.context_generator.parameters()}
    groups={'prefix':[i for i,p in enumerate(parameters) if id(p) not in ids],
            'generator':[i for i,p in enumerate(parameters) if id(p) in ids]}
    for qa,qb in ((0.,1.),(.4,1.),(1.,0.),(1.,1.)):
        torch.manual_seed(903)
        with torch.autocast('cuda',dtype=torch.float16):
            result=model(images[:4],RateControl(qa),capture=True)
            rd,metrics=reconstruction_objective(result,images[:4],RateControl(qa),model.distortion_scale.s_data,
                loss_fn,config['objective'],config['quality'])
            views=model.context_views(result,RateControl(qa),teacher,config,args.step,0,0,quality_b=qb)
            sf=views.native_loss(result.feature)
            wrong=context_distance(result.feature,views.target.roll(1,0)).mean()
            flip=context_distance(result.feature,views.target.flip(1)).mean()
            gs=[g.detach().float() for g in torch.autograd.grad(sf,parameters,retain_graph=True)]
            gr=[None if g is None else g.detach().float() for g in torch.autograd.grad(rd,parameters,allow_unused=True)]
            if views.mixed is not None:
                mixed=model.mixed_context_loss(views,4)
                gm=torch.autograd.grad(mixed,parameters)
                gs=[g+h.detach().float() for g,h in zip(gs,gm)]
            else:
                mixed=sf.detach()*0
        output={}
        for label, ix in groups.items():
            ns=sum(gs[i].square().sum() for i in ix).sqrt()
            nr=sum(gr[i].square().sum() for i in ix if gr[i] is not None).sqrt()
            dot=sum((gs[i]*gr[i]).sum() for i in ix if gr[i] is not None)
            output[label]=dict(sf_norm=float(ns),rd_norm=float(nr),sf_rd_norm_ratio=float(ns/nr.clamp_min(1e-30)),
                cosine=float(dot/(ns*nr).clamp_min(1e-30)))
        report['gradients'].append(dict(qa=qa,qb=qb,native_sf=float(sf),mixed_sf=float(mixed),
            correct_unweighted_sf=float(context_distance(result.feature.detach(),views.target).mean()),
            wrong_image_sf=float(wrong),spatial_flip_sf=float(flip),groups=output))
        save()
        print('gradient',args.step,qa,qb,output,flush=True)
        del result,views,rd,sf,wrong,flip,gs,gr,mixed
        if 'gm' in locals(): del gm
        gc.collect()
        torch.cuda.empty_cache()
save()
print('complete',str(OUT),report['elapsed_seconds'],flush=True)
