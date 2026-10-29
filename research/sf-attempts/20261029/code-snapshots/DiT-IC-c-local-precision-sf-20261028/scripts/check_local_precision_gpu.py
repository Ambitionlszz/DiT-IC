"""Real C10K/512 validation of outputs, paid packets and gradient routes."""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.decoder import decode_latent
from models.bt_self_flow.experiment import load_config, make_model
from models.bt_self_flow.local_precision import LocalPrecisionTeacher, centered_token_loss, sample_local_control
from models.bt_self_flow.packet import LocalPrecisionControl, RateControl
from models.sana_self_flow import unwrap_sana_transformer
from train_bt_self_flow import seed_all


def grad_norm(module):
    values = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def run(args):
    seed_all(903)
    config = load_config(args.config)
    checkpoint = read_checkpoint(args.checkpoint)
    model = make_model(config, torch.device('cuda:0'))
    model.load_bt_state(checkpoint['raw'])
    del checkpoint
    gc.collect()
    teacher = LocalPrecisionTeacher(model).cuda()
    dataset = ImageDataset(config['data']['train'], 512)
    images = torch.stack([dataset[(i*19, 903+i)] for i in range(4)]).cuda()
    report = dict(crop=512, batch=4, source_step=10000, uniform=[], mixed=[], no_optimizer_update=True)
    model.eval()
    model.codec.update(force=True)
    with torch.no_grad():
        # Validate under the same FP32 evaluation policy used by Kodak.
        for q in config['quality']['values']:
            result = model(images[:1], RateControl(q), capture=True)
            prediction = model.DiT(result.coarse,
                encoder_hidden_states=model._fixed_text_condition(1, images.device),
                timestep=model.self_flow_timestep(1, images.device), return_dict=False)[0]
            reference = decode_latent(model.vae, model.sched.step(prediction, result.coarse)/
                model.vae.config.scaling_factor, model.checkpoint_blocks).clamp(-1, 1)
            error = float((reference-result.image).abs().max())
            packet = model.compress(images[:1], RateControl(q))
            decoded = model.decompress(packet.to_bytes())
            packet_error = float((decoded-result.image).abs().max())
            assert error == 0. and packet_error < 2e-4, (q, error, packet_error)
            report['uniform'].append(dict(q=q, zero_adapter_max_error=error, packet_max_error=packet_error,
                total_bpp=8*len(packet.to_bytes())/(512*512)))
            del result, prediction, reference, decoded
        for qa, qb in ((0., 1.), (1., 0.)):
            control, _ = sample_local_control(config, RateControl(qa), 10000, 0, 0,
                                              (4, 8, 8), 'cuda', quality_b=qb)
            mixed = (control.tensor(4, 8, 8, 'cuda') != qa).flatten(1).sum(1).argmax()
            control = LocalPrecisionControl(control.low, control.high, control.high_mask[mixed:mixed+1], qa)
            result = model(images[:1], control)
            packet = model.compress(images[:1], control)
            decoded = model.decompress(packet.to_bytes())
            view = model.codec.decompress(packet)
            error = float((decoded-result.image).abs().max())
            assert error < 2e-4
            assert torch.equal(view['delta'].log(), result.scale_info['log_step'])
            report['mixed'].append(dict(qa=qa, qb=qb, packet_max_error=error,
                total_bpp=8*len(packet.to_bytes())/(512*512), mask_bits=8*len(packet.mask)))
            del result, decoded, view
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        control, qb = sample_local_control(config, RateControl(0.), 10000, 0, 0,
                                          (4, 8, 8), 'cuda', quality_b=1.)
        result = model(images, RateControl(0.), local_control=control, capture=True)
        sf, metrics = model.local_sf(result, RateControl(0.), qb, teacher, config)
        with torch.no_grad():
            fine = model.codec.quantize_analysis(result.source, LocalPrecisionControl(1., analysis_quality=0.),
                                                compute_likelihood=False)
            target = teacher(fine['mean'], model.fixed_text_embed,
                model.self_flow_timestep(4, images.device), fine['delta'].log())
            coarse_mask = result.scale_info['delta'] > fine['delta']*(1+1e-6)
            permutation = torch.randperm(target.shape[1], device=target.device)
            report['content'] = dict(correct=float(sf),
                wrong_image=float(centered_token_loss(result.feature, target.roll(1, 0), coarse_mask, (16, 16))[0]),
                shuffled=float(centered_token_loss(result.feature, target[:, permutation], coarse_mask, (16, 16))[0]),
                teacher_base_nmse=float((fine['mean']-result.clean).square().mean()/model.distortion_scale.s_data.square()),
                student_base_nmse=float((result.coarse-result.clean).square().mean()/model.distortion_scale.s_data.square()))
    sf.backward()
    blocks = unwrap_sana_transformer(model.DiT).transformer_blocks
    report['sf_only_gradients'] = {name: grad_norm(module) for name, module in (
        ('g_a', model.codec.g_a), ('g_s', model.codec.g_s), ('router', model.codec.router),
        ('precision_adapter', model.precision_adapter), ('prefix8', blocks[:8]), ('tail20', blocks[8:]), ('vae', model.vae))}
    assert all(report['sf_only_gradients'][name] > 0 for name in ('g_a', 'g_s', 'router', 'precision_adapter', 'prefix8'))
    assert report['sf_only_gradients']['tail20'] == report['sf_only_gradients']['vae'] == 0.
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    report.update(passed=True, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'output'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
