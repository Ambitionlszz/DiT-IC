"""Real 512px GPU preflight: native equivalence, task gradients, target readout."""
import argparse
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.experiment import make_model, load_config
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.dual_rate_task import DualRateTeacher
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import RateControl
from models.sana_self_flow import unwrap_sana_transformer


def run(config_path, checkpoint_path, output):
    config = load_config(config_path)
    model = make_model(config, torch.device('cuda:0'))
    source = read_checkpoint(checkpoint_path)
    model.load_bt_state(source['raw'], initialize_conditioner=True)
    model.train()
    dataset = ImageDataset(config['data']['train'], 512)
    images = torch.stack([dataset[(i*19, 903+i)] for i in range(config['train']['batch_size'])]).cuda()
    losses = ImageLoss(**config['image_loss']).cuda().eval()
    teacher = DualRateTeacher(model.DiT, model.text_proj, model.quality_conditioner, model.teacher_layer).cuda()
    control = RateControl(.4)
    report = dict(batch=images.shape[0], crop=512)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        clean = model.encode_clean_latent(images)
        view = model.codec.quantize_analysis(model.codec.analyze(clean, control), control)
        native = model.DiT(view['mean'], encoder_hidden_states=model._fixed_text_condition(images.shape[0], images.device),
                           timestep=model.self_flow_timestep(images.shape[0], images.device), return_dict=False)[0]
        from models.sana_self_flow import quality_conditioned_sana_forward
        adapted, _ = quality_conditioned_sana_forward(unwrap_sana_transformer(model.DiT), view['mean'],
            model._fixed_text_condition(images.shape[0], images.device), model.self_flow_timestep(images.shape[0], images.device),
            model.quality_map(view['mean'], control.value), model.quality_conditioner)
        # Exact split/full comparison on the real pretrained model as well.
        pred_full, h20 = quality_conditioned_sana_forward(unwrap_sana_transformer(model.DiT), view['mean'],
            model._fixed_text_condition(images.shape[0], images.device), model.self_flow_timestep(images.shape[0], images.device),
            model.quality_map(view['mean'], control.value), model.quality_conditioner, capture_layer=model.teacher_layer)
        image_split = model.readout(h20, view['mean'], model.quality_map(view['mean'], control.value))
        image_full = model.vae.decode(model.sched.step(pred_full, view['mean'])/model.vae.config.scaling_factor).sample.clamp(-1, 1)
        report['split_full_image_max_error'] = (image_split-image_full).abs().max().item()
        report['split_full_image_mse'] = (image_split.float()-image_full.float()).square().mean().item()
        assert report['split_full_image_mse'] < 1e-6, report
        del image_split, image_full, h20, pred_full
        report['zero_adapter_max_error'] = (native.float()-adapted.float()).abs().max().item()
        torch.testing.assert_close(native.float(), adapted.float(), atol=.003, rtol=.003)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.float16):
        result = model(images, control)
        loss, metrics = reconstruction_objective(result, images, control, model.distortion_scale.s_data, losses,
            config['objective'], config['quality'])
    loss.backward()
    report['main_loss'] = loss.item()
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        aux, metrics = model.auxiliary_task(result, images, control, teacher, losses, config, 10000, 0, 0)
    # Supervise only pixels to verify the path independently of cosine.
    metrics['sf_aux_distortion'].backward()
    report['auxiliary'] = {k:v.detach().item() for k,v in metrics.items()}
    sana = unwrap_sana_transformer(model.DiT)
    report['prefix_grad_norm'] = sum(p.grad.float().square().sum().item() for b in sana.transformer_blocks[:8]
        for p in b.parameters() if p.grad is not None)**.5
    report['predictor_grad_norm'] = sum(p.grad.float().square().sum().item() for p in model.representation_projector.parameters() if p.grad is not None)**.5
    report['codec_has_aux_grad'] = any(p.grad is not None for p in model.codec.parameters())
    report['tail_has_aux_grad'] = any(p.grad is not None for b in sana.transformer_blocks[8:] for p in b.parameters())
    report['vae_has_aux_grad'] = any(p.grad is not None for p in model.vae.parameters())
    assert report['prefix_grad_norm'] > 0 and report['predictor_grad_norm'] > 0
    assert not any(report[k] for k in ('codec_has_aux_grad', 'tail_has_aux_grad', 'vae_has_aux_grad'))
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    # Multiple quality/image pairs. Correct teacher may be incompatible with
    # a different scheduler base; record this rather than silently claiming an oracle gain.
    comparisons = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for index, q in enumerate((0., .4, .8, 1.)):
            batch_images = torch.stack([dataset[(100+index*101+j*43, 1903+j)] for j in range(images.shape[0])]).cuda()
            c = RateControl(q)
            clean = model.encode_clean_latent(batch_images)
            source_view = model.codec.analyze(clean, c)
            from types import SimpleNamespace
            current = SimpleNamespace(clean=clean, coarse=model.codec.quantize_analysis(source_view, c, compute_likelihood=False)['mean'])
            selection, mixed, qmap, high, highq, *_ = model.auxiliary_views(current, c, config, 10000+index, 0, 0)
            target = teacher(high, model.fixed_text_embed, model.self_flow_timestep(high.shape[0], high.device), highq)
            views = dict(oracle=target, wrong_image=target.flip(0), shuffle=target.roll(target.shape[1]//2, dims=1))
            record = dict(q=q)
            correct_image = None
            for name, feature in views.items():
                reconstructed = model.readout(feature, mixed, qmap)
                record[name] = losses(reconstructed, batch_images[selection])[0].item()
                if name == 'oracle':
                    correct_image = reconstructed
                else:
                    record[name+'_image_mse_vs_correct'] = (reconstructed.float()-correct_image.float()).square().mean().item()
            comparisons.append(record)
    report['teacher_readout_comparisons'] = comparisons
    report['target_content_sensitive'] = all(r['wrong_image_image_mse_vs_correct'] > 1e-7 and
                                             r['shuffle_image_mse_vs_correct'] > 1e-7 for r in comparisons)
    report['oracle_beats_wrong_fraction'] = sum(r['oracle'] < r['wrong_image'] for r in comparisons)/len(comparisons)
    print(json.dumps(report), flush=True)
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    assert report['target_content_sensitive'], 'teacher readout is insensitive to content'
    report['peak_memory_bytes'] = torch.cuda.max_memory_allocated()
    report['passed'] = True
    Path(output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.config, a.checkpoint, a.output)
