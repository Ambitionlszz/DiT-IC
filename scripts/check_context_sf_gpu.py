"""C10K/512/b4 preflight with real packets, SF isolation and resumed Adam updates."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCH_HOME','/data2/ls/checkpoints/torch-hub')
import torch
from models.bt_self_flow.checkpoint import DeploymentEMA, read_checkpoint
from models.bt_self_flow.context_sf import ContextTeacher, accumulate_sf
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, make_model, optimizer_groups
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.quantization_residual import c_sana_forward
from models.sana_self_flow import unwrap_sana_transformer


def norm(parameters):
    return sum(p.grad.float().square().sum().item() for p in parameters if p.grad is not None)**.5


def run(config_path,checkpoint_path,output):
    config=load_config(config_path)
    source=read_checkpoint(checkpoint_path)
    model=make_model(config,torch.device('cuda:0'))
    model.load_bt_state(source['raw'])
    teacher=ContextTeacher(model).cuda()
    ema=DeploymentEMA(model)
    ema.state={part:{k:v.cuda() for k,v in state.items()} for part,state in source['ema'].items()}
    groups,aux=optimizer_groups(model,config)
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),weight_decay=0)
    optimizer.load_state_dict(source['optimizer'])
    auxiliary=torch.optim.AdamW(aux,lr=config['optimizer']['aux_lr'],weight_decay=0)
    auxiliary.load_state_dict(source['aux_optimizer'])
    scaler=torch.amp.GradScaler('cuda',init_scale=64.)
    losses=ImageLoss(**config['image_loss']).cuda().eval()
    dataset=ImageDataset(config['data']['train'],config['data']['crop'])
    images=torch.stack([dataset[(i*19,903+i)] for i in range(config['train']['batch_size'])]).cuda()
    report=dict(batch=len(images),crop=config['data']['crop'],source_step=source['step'],passed=False)
    del source
    gc.collect()
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        result=model(images,RateControl(.2),capture=True)
        original,_,_,_=c_sana_forward(unwrap_sana_transformer(model.DiT),result.coarse,
            model._fixed_text_condition(len(images),images.device),model.self_flow_timestep(len(images),images.device))
        prediction,_=model.predict(result.coarse,result.scale_info['log_step'])
        report['source_prediction_max_error']=(prediction-original).abs().max().item()
        assert torch.equal(prediction,original),report
        report['context_shape']=list(result.feature.shape)
        del result,original,prediction
        model.codec.update(force=True)
        packet=model.compress(images[:1],RateControl(.2))
        direct=model(images[:1],RateControl(.2)).image
        decoded=model.decompress(packet.to_bytes())
        report['packet_image_mse']=(direct.float()-decoded.float()).square().mean().item()
        report['packet_total_bpp']=len(packet.to_bytes())*8/(images.shape[-2]*images.shape[-1])
        assert report['packet_image_mse']<1e-6,report
        del packet,direct,decoded
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        result=model(images,RateControl(.2),capture=True)
        views=model.context_views(result,RateControl(.2),teacher,config,10000,0,0,quality_b=1.)
        native=views.native_loss(result.feature)
    allowed={id(p) for p in model.sf_parameters()}
    accumulate_sf(native,model.sf_parameters(),retain_graph=True)
    assert all(p.grad is None or id(p) in allowed for p in model.parameters())
    report['native_sf_whitelist_passed']=True
    report['native_sf_grad_norm']=norm(model.sf_parameters())
    assert report['native_sf_grad_norm']>0
    model.zero_grad(set_to_none=True)
    del result,native
    gc.collect()
    with torch.autocast('cuda',dtype=torch.float16):
        mixed=model.mixed_context_loss(views,len(images))
    accumulate_sf(mixed,model.sf_parameters())
    assert all(p.grad is None or id(p) in allowed for p in model.parameters())
    report['mixed_sf_whitelist_passed']=True
    report['mixed_sf_grad_norm']=norm(model.sf_parameters())
    assert report['mixed_sf_grad_norm']>0
    del views,mixed
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    report['updates']=[]
    for i,qa in enumerate((.0,.2,1.,1.)):
        qb=(1.,1.,0.,1.)[i]
        control=RateControl(qa)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):
            result=model(images,control,capture=True)
            views=model.context_views(result,control,teacher,config,10000+i,0,0,quality_b=qb)
            native=views.native_loss(result.feature)
            rd,metrics=reconstruction_objective(result,images,control,model.distortion_scale.s_data,
                losses,config['objective'],config['quality'])
        accumulate_sf(scaler.scale(native),model.sf_parameters(),retain_graph=True)
        scaler.scale(rd).backward()
        total_sf=native.detach().item()
        if views.mixed is not None:
            with torch.autocast('cuda',dtype=torch.float16):
                mixed=model.mixed_context_loss(views,len(images))
            accumulate_sf(scaler.scale(mixed),model.sf_parameters())
            total_sf+=mixed.item()
            del mixed
        scaler.unscale_(optimizer)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert norm(model.codec.router.parameters())>0,'native RD must still update router'
        clip=torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],config['optimizer']['max_grad_norm'])
        scale=scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale()>=scale,'real resumed update overflowed'
        auxiliary.zero_grad(set_to_none=True)
        model.codec.entropy_bottleneck_loss().backward()
        auxiliary.step()
        ema.update(model)
        teacher.update_context_student(model,config['self_flow']['teacher_decay'])
        report['updates'].append(dict(qa=qa,qb=qb,rd=rd.item(),sf=total_sf,grad_norm=clip.item(),loss_scale=scaler.get_scale(),
            mixed_images=len(views.indices),gates=[p.item() for p in model.context_readout.gates]))
        del result,views,native,rd,metrics
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        native=model(images,RateControl(.2)).image
        model.context_mode='off'
        off=model(images,RateControl(.2)).image
        model.context_mode='wrong_image'
        wrong=model(images,RateControl(.2)).image
        model.context_mode='native'
        report['trained_readout_image_mse']=(native.float()-off.float()).square().mean().item()
        report['wrong_image_context_mse']=(native.float()-wrong.float()).square().mean().item()
        # At four updates this proves actual use, not improved reconstruction.
        assert report['trained_readout_image_mse']>0
        packet=model.compress(images[:1],RateControl(.2))
        direct=model(images[:1],RateControl(.2)).image
        decoded=model.decompress(packet.to_bytes())
        report['trained_packet_image_mse']=(direct.float()-decoded.float()).square().mean().item()
        assert report['trained_packet_image_mse']<1e-6
    torch.cuda.synchronize()
    report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        free_device_bytes=torch.cuda.mem_get_info()[0],passed=True)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    run(a.config,a.checkpoint,a.output)
