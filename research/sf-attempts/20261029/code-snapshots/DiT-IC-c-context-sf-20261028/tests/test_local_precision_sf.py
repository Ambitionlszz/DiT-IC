"""Paid mixed packets, native equivalence, SF gradient routing and state."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.bt_self_flow.checkpoint import DeploymentEMA, clone_state
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.experiment import load_config
from models.bt_self_flow.local_precision import (CLocalPrecisionCodec, LocalPrecisionTeacher,
    centered_token_loss, sample_local_control)
from models.bt_self_flow.model import BTCodec
from models.bt_self_flow.packet import BTImagePacket, LocalPrecisionControl, RateControl, fixed_side_bits
from test_bt_backend import tiny_model
from test_c_sf_migration import TinyC, source_state
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def settings():
    return load_config(ROOT/'configs/bt_self_flow/c_local_precision.json')


def local_model():
    model = tiny_model('C')
    model.__class__ = CLocalPrecisionCodec
    model.codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4,
        precision_conditioning=True, local_precision_protocol=True, rate_likelihood='noise')
    model.precision_adapter = nn.Linear(1, 32)
    nn.init.zeros_(model.precision_adapter.weight)
    nn.init.zeros_(model.precision_adapter.bias)
    model.codec.update(force=True)
    return model


@pytest.fixture(autouse=True)
def tiny_decode(monkeypatch):
    decode = lambda vae, latent, checkpoint_blocks: vae.decode(latent).sample
    monkeypatch.setattr('models.bt_self_flow.model.decode_latent', decode)
    monkeypatch.setattr('models.bt_self_flow.decoder.decode_latent', decode)


@pytest.mark.parametrize('qa,qb', [(0., 1.), (1., 0.), (.4, .4)])
def test_sampling_and_real_mixed_packet_roundtrip(qa, qb):
    model = local_model().eval()
    control, value = sample_local_control(settings(), RateControl(qa), 10000, 0, 0,
        (4, 4, 4), 'cpu', quality_b=qb)
    assert value == qb
    if qa != qb:
        values = control.tensor(4, 4, 4, 'cpu')
        assert sorted((values != qa).flatten(1).sum(1).tolist()) == [0, 0, 4, 4]
        assert torch.equal(values, sample_local_control(settings(), RateControl(qa), 10000, 0, 0,
            (4, 4, 4), 'cpu', quality_b=qb)[0].tensor(4, 4, 4, 'cpu'))
        index = int((values != qa).flatten(1).sum(1).argmax())
        single = LocalPrecisionControl(control.low, control.high, control.high_mask[index:index+1], qa)
    else:
        single = control
    image = torch.rand(1, 3, 256, 256)*2-1
    with torch.no_grad():
        simulated = model(image, single)
        packet = model.compress(image, single)
        data = packet.to_bytes()
        decoded = model.decompress(data)
        view = model.codec.decompress(BTImagePacket.from_bytes(data))
    torch.testing.assert_close(decoded, simulated.image, atol=2e-4, rtol=0)
    torch.testing.assert_close(view['delta'].log(), simulated.scale_info['log_step'], atol=0, rtol=0)
    billed_side = 8*(len(data)-len(packet.main_stream)-len(packet.hyper_stream))
    assert billed_side == simulated.side_bits == fixed_side_bits(4, 4, 'C', qa != qb)
    # Reusing across q without the new explicit anchor remains forbidden.
    with pytest.raises(ValueError, match='cannot be reused'):
        model.codec.quantize_analysis(simulated.source, RateControl(.8 if qa != .8 else 0.))


def test_zero_condition_exactly_preserves_native_C_and_capture_is_single_forward():
    model = local_model().eval()
    old = deepcopy(model)
    old.__class__ = BTCodec
    image = torch.rand(4, 3, 256, 256)*2-1
    with torch.no_grad():
        for q in (0., .4, 1.):
            actual = model(image, RateControl(q), capture=True)
            expected = old(image, RateControl(q))
            torch.testing.assert_close(actual.image, expected.image, atol=0, rtol=0)
    assert model.prediction_calls == 3
    assert actual.feature.shape == (4, 64, 32)


def test_actual_student_sf_reaches_codec_teacher_has_no_graph_and_ema_condition_updates():
    model = local_model().train()
    teacher = LocalPrecisionTeacher(model)
    ema = DeploymentEMA(model)
    control, qb = sample_local_control(settings(), RateControl(0.), 10000, 0, 0,
                                      (4, 4, 4), 'cpu', quality_b=1.)
    result = model(torch.rand(4, 3, 256, 256)*2-1, RateControl(0.),
                   local_control=control, capture=True)
    rng = torch.get_rng_state().clone()
    sf, metrics = model.local_sf(result, RateControl(0.), qb, teacher, settings())
    assert torch.equal(rng, torch.get_rng_state())
    assert metrics['sf_valid_fraction'] == .875
    sf.backward()
    for module in (model.codec.g_a, model.codec.g_s, model.codec.router, model.precision_adapter):
        assert any(p.grad is not None and p.grad.norm() > 0 for p in module.parameters())
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert model.prediction_calls == 1
    with torch.no_grad():
        model.precision_adapter.weight.fill_(2.)
    teacher.update_local_student(model, .5)
    ema.update(model)
    assert torch.all(teacher.precision_adapter.weight == 1.)
    torch.testing.assert_close(ema.state['precision_adapter']['weight'], torch.full_like(model.precision_adapter.weight, .002))
    teacher_state = clone_state({'teacher': teacher.get_ema_state_dict()})['teacher']
    teacher.load_ema_state_dict(teacher_state)
    assert set(model.export_state()) == {'codec', 'distortion_scale', 'vae', 'dit', 'text', 'precision_adapter'}


def test_token_loss_keeps_channel_content_and_zero_masks_zero_grad():
    value = torch.randn(4, 16, 8, requires_grad=True)
    mask = torch.ones(4, 1, 2, 2, dtype=torch.bool)
    correct = centered_token_loss(value, value.detach(), mask, (4, 4))[0]
    wrong = centered_token_loss(value, value.detach().roll(1, 0), mask, (4, 4))[0]
    rotated = centered_token_loss(value, value.detach()@torch.linalg.qr(torch.randn(8, 8))[0], mask, (4, 4))[0]
    assert abs(correct) < 1e-6 and wrong > .5 and rotated > .5
    zero, valid = centered_token_loss(value, torch.randn_like(value), ~mask, (4, 4))
    zero.backward()
    assert zero == valid == 0 and torch.count_nonzero(value.grad) == 0


def test_migration_preserves_moments_and_adds_only_adapter():
    from scripts.migrate_local_precision import migrate
    source, _, _ = source_state()
    class Target(TinyC):
        def __init__(self):
            super().__init__()
            self.precision_adapter = nn.Linear(1, 3)
            nn.init.zeros_(self.precision_adapter.weight)
            nn.init.zeros_(self.precision_adapter.bias)
        def export_state(self):
            return dict(super().export_state(), precision_adapter=self.precision_adapter.state_dict())
        def load_bt_state(self, state):
            super().load_bt_state(state)
            self.precision_adapter.load_state_dict(state['precision_adapter'])
    target = Target()
    migrated = migrate(source, target, settings(), SimpleNamespace(fingerprint='same'), {}, 'source')
    assert migrated['step'] == migrated['micro_seen'] == 10000
    assert len(migrated['optimizer']['state']) == len(source['optimizer']['state'])
    assert migrated['scheduler'] == source['scheduler']
    for kind in ('raw', 'ema'):
        for part, values in source[kind].items():
            for key, value in values.items():
                assert torch.equal(value, migrated[kind][part][key])
        assert all(torch.count_nonzero(v) == 0 for v in migrated[kind]['precision_adapter'].values())


def test_real_trainer_exact_resume_includes_teacher_adapter_adam_and_ema(tmp_path):
    import io
    from contextlib import redirect_stdout
    from unittest.mock import patch
    from models.bt_self_flow.checkpoint import METHOD, SCHEMA, read_checkpoint, save_atomic
    from test_bt_training import small_config
    from test_quantization_residual_sf import make_images
    import train_bt_self_flow as trainer
    make_images(tmp_path)
    config = small_config(tmp_path, 'joint')
    config['model']['backend'] = 'C'
    config['self_flow'] = dict(settings()['self_flow'], warmup=0, ramp=2)
    config['quality'] = settings()['quality']
    config['objective'] = dict(base_weight=.1, repaired_weight=0.)
    config['train'].update(steps=3, batch_size=2, accumulation_steps=1, effective_batch=2, router_ramp_steps=0)
    config['image_loss'] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    initial = tmp_path/'initial.pt'
    save_atomic(initial, dict(method=METHOD, schema=SCHEMA, phase='backend_init', backend='shared',
        step=0, raw=clone_state(local_model().export_state(), cpu=True), config=config, assets={'test': 'local'}))
    args = SimpleNamespace(resume=None, init_checkpoint=str(initial), max_updates=None)
    def factory(cfg, device):
        model = local_model().to(device)
        model.vae.requires_grad_(False)
        model.checkpoint_blocks = True
        return model
    with patch.object(trainer, 'make_model', side_effect=factory), \
         patch.object(trainer, 'asset_identity', return_value={'test': 'local'}), redirect_stdout(io.StringIO()):
        trainer.train(config, args, 0, 1, torch.device('cpu'))
        split = deepcopy(config)
        split['train']['output'] = str(tmp_path/'split')
        args.max_updates = 2
        trainer.train(split, args, 0, 1, torch.device('cpu'))
        args.resume, args.init_checkpoint, args.max_updates = str(tmp_path/'split/paused.pt'), None, None
        trainer.train(split, args, 0, 1, torch.device('cpu'))
    full = read_checkpoint(tmp_path/'joint/complete.pt')
    resumed = read_checkpoint(tmp_path/'split/complete.pt')
    for kind in ('raw', 'ema'):
        for part, values in full[kind].items():
            for key, value in values.items():
                assert torch.equal(value, resumed[kind][part][key]), (kind, part, key)
    for key, value in full['teacher'].items():
        assert torch.equal(value, resumed['teacher'][key]), key
    for identity, state in full['optimizer']['state'].items():
        for key, value in state.items():
            assert torch.equal(value, resumed['optimizer']['state'][identity][key])
    assert resumed['local_precision']['teacher_initialized']


def test_self_attention_keeps_large_finite_projection_through_amp():
    from test_bt_backend import tiny_sana
    from models.bt_self_flow.precision import StableSanaLinearAttention
    attention = tiny_sana().transformer_blocks[0].attn1
    attention.to_out[0] = nn.Sequential(nn.Linear(32, 2, bias=False), nn.Linear(2, 32, bias=False))
    with torch.no_grad():
        attention.to_v.weight.fill_(1.)
        for parameter in attention.to_out[0].parameters():
            parameter.fill_(100.)
    hidden = torch.ones(1, 16, 32, requires_grad=True)
    processor = StableSanaLinearAttention()
    expected = processor(attention, hidden)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = processor(attention, hidden)
    assert actual.dtype == torch.float32 and actual.abs().max() > 65504
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    assert torch.isfinite(hidden.grad).all()
