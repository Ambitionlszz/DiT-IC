"""Actual diffusers blocks: frozen readout, input gradients and two-rank reduction."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from diffusers import AutoencoderDC

sys.path.insert(0, str(Path(__file__).parent))
from test_bt_backend import tiny_model
from models.sana_self_flow import TokenQualityConditioner, quality_conditioned_sana_forward, unwrap_sana_transformer
from models.bt_self_flow.dual_rate_task import (CDualRateTaskCodec, FullFeaturePredictor,
    DualRateTeacher, frozen_decode, synchronize_gradients)
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.packet import RateControl


def task_model():
    base = tiny_model('C')
    base.__class__ = CDualRateTaskCodec
    base.quality_conditioner = TokenQualityConditioner(6, 32, 4)
    base.representation_projector = FullFeaturePredictor(32)
    base.vae = AutoencoderDC(latent_channels=4, encoder_block_out_channels=(16, 16),
        decoder_block_out_channels=(16, 16), encoder_layers_per_block=(1, 1),
        decoder_layers_per_block=(1, 1), encoder_qkv_multiscales=((), ()),
        decoder_qkv_multiscales=((), ()), attention_head_dim=8)
    base.vae.config.scaling_factor = 1.
    return base


@pytest.mark.parametrize('checkpointed', [False, True])
def test_readout_matches_current_C_and_freezes_weights_on_recompute(checkpointed):
    model = task_model().train()
    model.checkpoint_blocks = checkpointed
    nn.init.normal_(model.quality_conditioner.block_projection.weight, std=.002)
    nn.init.normal_(model.quality_conditioner.output_projection.weight, std=.002)
    mixed = torch.randn(2, 4, 8, 8)
    qmap = torch.randint(6, (2, 8, 8))
    time = model.self_flow_timestep(2, mixed.device)
    sana = unwrap_sana_transformer(model.DiT)
    prediction, target = quality_conditioned_sana_forward(sana, mixed,
        model._fixed_text_condition(2, mixed.device), time, qmap, model.quality_conditioner,
        capture_layer=model.teacher_layer)
    expected = model.vae.decode(model.sched.step(prediction, mixed)).sample.clamp(-1, 1)
    feature = target.detach().requires_grad_()
    actual = model.readout(feature, mixed, qmap)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual.square().mean().backward()
    assert feature.grad.norm() > 0 and torch.isfinite(feature.grad).all()
    assert all(p.grad is None for p in model.parameters())


def test_auxiliary_image_gradient_reaches_shared_prefix_and_predictor_only():
    model = task_model().train()
    model.checkpoint_blocks = True
    teacher = DualRateTeacher(model.DiT, model.text_proj, model.quality_conditioner, model.teacher_layer)
    mixed, high = torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8)
    qmap = torch.randint(6, (2, 8, 8))
    model.auxiliary_views = lambda *a: (torch.tensor([0, 1]), mixed, qmap, high,
        torch.ones_like(qmap)*5, torch.tensor([0, 0]), torch.tensor([5, 5]), qmap[:, None] == 0)
    loss, metrics = model.auxiliary_task(None, torch.randn(2, 3, 16, 16), RateControl(0), teacher,
        ImageLoss(mse_weight=4, lpips_weight=0, dists_weight=0), {}, 0, 0, 0)
    # Use ONLY image supervision here: cosine must not conceal a broken path.
    metrics['sf_aux_distortion'].backward()
    assert model.representation_projector.fc2.weight.grad.norm() > 0
    first = unwrap_sana_transformer(model.DiT).transformer_blocks[0]
    assert any(p.grad is not None and p.grad.norm() > 0 for p in first.parameters() if p.requires_grad)
    assert all(p.grad is None for p in unwrap_sana_transformer(model.DiT).transformer_blocks[1:].parameters())
    assert all(p.grad is None for p in model.codec.parameters())
    assert all(p.grad is None for p in model.vae.parameters())
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())


def test_separate_quality_analyses_and_decoded_mixing():
    model = task_model()
    calls = []
    class Codec(nn.Module):
        def analyze(self, clean, q):
            calls.append(q.value)
            return SimpleNamespace(clean=clean, q=q.value)
        def quantize_analysis(self, source, q, **kw):
            assert q.value == source.q
            return {'mean': source.clean+source.q}
    model.codec = Codec()
    config = dict(train=dict(seed=903), self_flow=dict(cell_fraction=.25))
    a = torch.zeros(24, 4, 8, 8, requires_grad=True)
    clean = torch.ones_like(a, requires_grad=True)
    selection, mixed, qmap, high, highq, qa, qb, mask = model.auxiliary_views(
        SimpleNamespace(coarse=a, clean=clean), RateControl(.4), config, 10000, 0, 0)
    assert len(selection) == 12 and mask.float().mean() == .25
    assert not mixed.requires_grad and not high.requires_grad
    assert sorted(calls) == sorted((qb.unique().float()/5).tolist()) or all(
        any(abs(v-i/5) < 1e-5 for v in calls) for i in qb.unique().tolist())
    torch.testing.assert_close(mixed[:, 0][mask[:, 0]], (1+qb.float()/5)[:, None, None].expand(-1, 8, 8)[mask[:, 0]])
    assert torch.equal(qmap, torch.where(mask[:, 0], qb[:, None, None], qa[:, None, None]))


def reduction_worker(rank, path):
    dist.init_process_group('gloo', init_method='file://'+path, rank=rank, world_size=2)
    try:
        params = [nn.Parameter(torch.tensor([1., 2.])), nn.Parameter(torch.tensor([3.])), nn.Parameter(torch.tensor([4.]))]
        # Two sequential losses, then synchronization; rank-local unused param.
        (params[0].sum()*(rank+1)).backward()
        if rank == 0:
            (params[1].sum()*6).backward()
        synchronize_gradients(params, 2, bucket_bytes=4)
        torch.testing.assert_close(params[0].grad, torch.tensor([1.5, 1.5]))
        torch.testing.assert_close(params[1].grad, torch.tensor([3.]))
        torch.testing.assert_close(params[2].grad, torch.tensor([0.]))
    finally:
        dist.destroy_process_group()


def test_two_rank_sequential_gradient_mean(tmp_path):
    mp.spawn(reduction_worker, (str(tmp_path/'rendezvous'),), nprocs=2, join=True)


def test_migration_preserves_C_moments_and_initializes_only_new_groups():
    from test_c_sf_migration import source_state, TinyC
    from scripts.prepare_c_sf_ablation import paired_configs
    from scripts.migrate_dual_rate_task import migrate
    source, _, _ = source_state()
    config = paired_configs()['pure_c']
    config['self_flow'] = dict(enabled=True, method='dual_rate_task_v1')
    class TaskTiny(TinyC):
        def __init__(self):
            super().__init__()
            self.quality_conditioner = nn.Linear(3, 3)
            self.representation_projector = nn.Linear(3, 3)
        def export_state(self):
            return dict(super().export_state(), conditioner=self.quality_conditioner.state_dict(),
                        projector=self.representation_projector.state_dict())
    model = TaskTiny()
    result = migrate(source, model, config, SimpleNamespace(fingerprint='same'), {}, 'pure-C10K')
    assert result['micro_seen'] == result['step'] == 10000
    assert result['migration']['preserved_global_samples'] == 80000
    assert result['dual_rate_task'] == dict(prepared=False)
    assert result['scheduler']['last_epoch'] == source['scheduler']['last_epoch']
    assert len(result['optimizer']['state']) == len(source['optimizer']['state'])
    groups = result['optimizer']['param_groups']
    assert groups[-1]['name'] == 'task_sf'
    assert all(i not in result['optimizer']['state'] for i in groups[-1]['params'])
    for kind in ('raw', 'ema'):
        for part, values in source[kind].items():
            for key, tensor in values.items():
                torch.testing.assert_close(result[kind][part][key], tensor, atol=0, rtol=0)
    for old, new in zip(source['optimizer']['state'].values(), result['optimizer']['state'].values()):
        for key in old:
            torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
