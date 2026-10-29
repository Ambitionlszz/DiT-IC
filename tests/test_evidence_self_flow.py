"""Cross-rate evidence: gradient contract, independent receiver, EMA and recovery."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import timedelta
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, DeploymentEMA, clone_state, read_checkpoint, save_atomic
from models.bt_self_flow.data import counter_seed
from models.bt_self_flow.evidence_model import EvidenceSelfFlowCodec
from models.bt_self_flow.evidence_training import average_gradients, evidence_backward, higher_control, sf_coefficient
from models.bt_self_flow.experiment import load_config, optimizer_groups
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.packet import RateControl
from test_c_precision_v3 import full_factory, activate_conditions, config_for, write_image


torch.set_num_threads(1)


class LoRADecoder(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_A = nn.Conv2d(4, 2, 1, bias=False)
        self.lora_B = nn.Conv2d(2, 3, 1, bias=False)
        nn.init.normal_(self.lora_B.weight, std=.03)

    def forward(self, value):
        return self.base(value) + self.lora_B(self.lora_A(value))


def factory(config=None, device='cpu'):
    model = full_factory()
    model.__class__ = EvidenceSelfFlowCodec
    model.initialize_evidence(config['evidence_self_flow'] if config else {})
    model.vae.decoder = LoRADecoder(model.vae.decoder)
    return model.to(device)


def tiny_config(root):
    config = config_for(root, 'joint')
    config['evidence_self_flow'] = dict(method='cross_rate_evidence_self_flow_v1', teacher_evidence='high',
        sf_enabled=True, beta=.25, coefficient=.05, warmup=0, ramp=1, student_layer=1, teacher_layer=3)
    config['image_loss'] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    config['train'].update(accumulation_steps=1, effective_batch=2)
    return config


def parameters(model, config):
    groups, _ = optimizer_groups(model, config)
    return ([p for g in groups if g['name'] == 'codec' for p in g['params']],
            [p for g in groups if g['name'] != 'codec' for p in g['params']])


def run_backward(model, config, image, q=.4, step=2):
    codec, backend = parameters(model, config)
    return evidence_backward(model, image, RateControl(q), ImageLoss(mse_weight=4., lpips_weight=0., dists_weight=0.),
        config, step, 0, 1, torch.amp.GradScaler('cuda', enabled=False), codec, backend, torch.bfloat16)


def test_parameter_specific_gradients_match_independent_reference(tmp_path):
    config = tiny_config(tmp_path)
    model = factory(config).train()
    activate_conditions(model)
    nn.init.normal_(model.evidence_projection.weight, std=.01)
    reference = deepcopy(model)
    image = torch.rand(1, 3, 256, 256)*2-1
    torch.manual_seed(51)
    initial_rng = torch.get_rng_state()
    result, loss, metrics = run_backward(model, config, image)
    final_rng = torch.get_rng_state()
    assert torch.isfinite(loss) and metrics['sf_weight'] == pytest.approx(.05)
    actual = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}

    torch.set_rng_state(initial_rng)
    codec, backend = parameters(reference, config)
    low = RateControl(.4)
    clean = reference.encode_clean_latent(image)
    source = reference.codec.analyze(clean, low)
    view = reference.codec.quantize_analysis(source, low)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(counter_seed(config['train']['seed'], 2, 9101))
        with torch.no_grad():
            high = higher_control(config, low, 2, 0)
            high_view = reference.codec.quantize_analysis(reference.codec.analyze(clean, high), high)
        _, teacher_image, target = reference.repair(view['mean'].detach(), precision=view['delta'].detach(),
            evidence=(high_view['mean'], high_view['delta']), capture_layer=reference.teacher_layer)
        teacher_distortion = 4*F.mse_loss(teacher_image, image)
    student = reference.result_from_view(source, view, low, capture=True)
    native, values = reconstruction_objective(student, image, low, reference.distortion_scale.s_data,
        ImageLoss(mse_weight=4., lpips_weight=0., dists_weight=0.), config['objective'], config['quality'])
    sf = (1-F.cosine_similarity(student.feature.float(), target.detach().float(), dim=-1, eps=1e-6)).mean()
    expected_codec = torch.autograd.grad(native, codec, retain_graph=True, allow_unused=True)
    objective = .75*4*F.mse_loss(student.image, image) + .25*teacher_distortion + .05*sf
    expected_backend = torch.autograd.grad(objective, backend, allow_unused=True)
    lookup = {id(p): name for name,p in reference.named_parameters()}
    for parameter, expected in zip(codec+backend, expected_codec+expected_backend):
        value = actual[lookup[id(parameter)]]
        if expected is None:
            assert value is None
        else:
            torch.testing.assert_close(value, expected, atol=2e-6, rtol=2e-4)
    assert any(actual[n] is not None and actual[n].abs().sum() > 0 for n in actual if 'precision_embedding' in n)
    assert any(actual[n] is not None and actual[n].abs().sum() > 0 for n in actual if n.startswith('vae.'))
    assert torch.equal(final_rng, torch.get_rng_state()), 'teacher consumed native student RNG'


@pytest.mark.parametrize('case', ['interface', 'max_quality'])
def test_identity_cases_match_native_gradient_exactly(tmp_path, case):
    config = tiny_config(tmp_path)
    if case == 'interface':
        config['evidence_self_flow'].update(teacher_evidence='low', sf_enabled=False)
    model = factory(config).train()
    reference = deepcopy(model)
    image = torch.rand(1, 3, 256, 256)*2-1
    q = 1. if case == 'max_quality' else .2
    torch.manual_seed(71)
    rng = torch.get_rng_state()
    _, _, metrics = run_backward(model, config, image, q=q)
    torch.set_rng_state(rng)
    result = reference(image, rate_control=RateControl(q))
    loss, _ = reconstruction_objective(result, image, RateControl(q), reference.distortion_scale.s_data,
        ImageLoss(mse_weight=4., lpips_weight=0., dists_weight=0.), config['objective'], config['quality'])
    loss.backward()
    for (name, p), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
        if name.endswith('.quantiles') or not p.requires_grad:
            continue
        if ref.grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, ref.grad, atol=1e-6, rtol=1e-4)
    assert metrics['sf'] == 0 and metrics['sf_weight'] == 0
    assert model.prediction_calls == 1


def test_fresh_packet_receiver_ema_and_strict_context_restore(tmp_path):
    config = tiny_config(tmp_path)
    sender = factory(config).eval()
    activate_conditions(sender)
    sender.codec.update(force=True)
    ema = DeploymentEMA(sender)
    with torch.no_grad():
        sender.evidence_projection.weight.add_(1)
    ema.update(sender)
    torch.testing.assert_close(ema.state['evidence_projection']['weight'],
                               sender.evidence_projection.weight*.001)
    receiver = factory(config).eval()
    receiver.load_bt_state(clone_state(sender.export_state()))
    image = torch.rand(1, 3, 256, 256)*2-1
    packet = sender.compress(image, RateControl(.4)).to_bytes()
    expected = sender(image, rate_control=RateControl(.4)).image
    def forbidden(*args, **kwargs):
        raise AssertionError('receiver accessed source / planner')
    receiver.encode_clean_latent = forbidden
    receiver.codec.router.forward = forbidden
    actual = receiver.decompress(packet)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert receiver.prediction_calls == 1
    state = sender.export_state()
    del state['evidence_projection']
    with pytest.raises(ValueError, match='missing'):
        receiver.load_bt_state(state)


def ddp_worker(rank, directory):
    root = Path(directory)
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method=f"file://{root/'rendezvous'}",
                            timeout=timedelta(seconds=180))
    config = tiny_config(root)
    models = []
    def recording_factory(cfg, device):
        model = factory(cfg, device)
        models.append(model)
        return model
    def args(resume=None, maximum=None):
        return SimpleNamespace(init_checkpoint=None if resume else str(root/'init.pt'), resume=resume, max_updates=maximum)
    try:
        with patch.object(trainer, 'make_model', side_effect=recording_factory), \
             patch.object(trainer, 'asset_identity', return_value={}), redirect_stdout(io.StringIO()):
            trainer.train(config, args(), rank, 2, torch.device('cpu'))
            whole = clone_state(models[-1].export_state())
            config['train']['output'] = str(root/'split')
            trainer.train(config, args(maximum=1), rank, 2, torch.device('cpu'))
            trainer.train(config, args(resume=str(root/'split/paused.pt')), rank, 2, torch.device('cpu'))
        for part, values in models[-1].export_state().items():
            for name, value in values.items():
                assert torch.equal(value, whole[part][name]), f'resume differed: {part}.{name}'
        for parameter in models[-1].parameters():
            if parameter.requires_grad:
                expected = parameter.detach().clone()
                dist.broadcast(expected, 0)
                assert torch.equal(parameter, expected), 'distributed parameter divergence'
        # A rank-local unused parameter must still receive the global mean;
        # a globally unused one must remain None for AdamW's update semantics.
        probes = [nn.Parameter(torch.zeros(2)) for _ in range(3)]
        probes[0].grad = torch.full((2,), float(rank+1))
        probes[1].grad = torch.full((2,), 2.) if rank else None
        average_gradients(probes, 2)
        assert torch.equal(probes[0].grad, torch.full((2,), 1.5))
        assert torch.equal(probes[1].grad, torch.ones(2)) and probes[2].grad is None
    finally:
        dist.destroy_process_group()


def test_two_rank_main_aux_updates_and_exact_resume(tmp_path):
    write_image(tmp_path)
    config = tiny_config(tmp_path)
    model = full_factory(config)
    model.vae.decoder = LoRADecoder(model.vae.decoder)
    save_atomic(tmp_path/'init.pt', dict(method=METHOD, schema=SCHEMA, phase='backend_init', backend='shared', step=0,
        raw=clone_state(model.export_state(), cpu=True), config=config, assets={}, calibration={}))
    mp.spawn(ddp_worker, args=(str(tmp_path),), nprocs=2)
    checkpoint = read_checkpoint(tmp_path/'split/complete.pt')
    assert len(checkpoint['rng_by_rank']) == 2
    assert 'evidence_projection' in checkpoint['raw'] and 'evidence_projection' in checkpoint['ema']


def test_recipe_keeps_native_c_rd_and_requested_effective_batch():
    root = Path(__file__).resolve().parents[1]/'configs/bt_self_flow'
    native = load_config(root/'precision_v3/c_joint.yaml')
    full = load_config(root/'evidence_v1/e3_self_flow.yaml')
    # Portable native asset paths are explicitly relocated by this host recipe.
    for key in ('sana_path', 'text_path'):
        native['model'][key] = full['model'][key]
    for name in ('model', 'quality', 'objective', 'optimizer', 'image_loss'):
        assert native[name] == full[name]
    assert full['train']['batch_size'] == 4 and full['train']['accumulation_steps'] == 1
    assert full['train']['effective_batch'] == 8
    settings = full['evidence_self_flow']
    assert [sf_coefficient(settings, i) for i in (0,1000,2000,3000)] == [0.,0.,.025,.05]
