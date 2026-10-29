"""Deployed context, strict SF gradient isolation, real packets and exact resume."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from models.bt_self_flow.checkpoint import METHOD, SCHEMA, DeploymentEMA, clone_state, read_checkpoint, save_atomic
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.context_sf import (CContextCodec, ContextGenerator, ContextReadout,
    ContextTeacher, ContextViews, accumulate_sf, context_distance)
from models.bt_self_flow.experiment import load_config
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.quantization_residual import c_sana_forward
from models.sana_self_flow import unwrap_sana_transformer
from test_bt_backend import tiny_model
from test_bt_training import small_config
from test_quantization_residual_sf import make_images
import train_bt_self_flow as trainer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def context_model():
    model = tiny_model("C")
    model.__class__ = CContextCodec
    model.codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4,
                               precision_conditioning=True, rate_likelihood="noise")
    model.context_generator = ContextGenerator(32, 8)
    model.context_readout = ContextReadout(3)
    model.context_mode = "native"
    return model


def settings():
    return load_config(ROOT/"configs/bt_self_flow/c_context_sf.json")


@pytest.fixture(autouse=True)
def tiny_ae_decode(monkeypatch):
    monkeypatch.setattr("models.bt_self_flow.decoder.decode_latent",
                        lambda vae, latent, checkpoint_blocks: vae.decode(latent).sample)
    monkeypatch.setattr("models.bt_self_flow.model.decode_latent",
                        lambda vae, latent, checkpoint_blocks: vae.decode(latent).sample)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_zero_readout_matches_source_and_packet_remains_decodable(checkpointed):
    model = context_model().eval()
    model.checkpoint_blocks = checkpointed
    image = torch.rand(2, 3, 256, 512)*2-1
    with torch.no_grad():
        result = model(image, RateControl(.4), capture=True)
        original, _, _, _ = c_sana_forward(unwrap_sana_transformer(model.DiT), result.coarse,
            model._fixed_text_condition(2, image.device), model.self_flow_timestep(2, image.device))
        actual, _ = model.predict(result.coarse, result.scale_info['log_step'])
        torch.testing.assert_close(original, actual, rtol=0, atol=0)
        assert result.feature.shape == (2, 32, 32)
        for gate in model.context_readout.gates:
            gate.fill_(.1)
        native = model(image, RateControl(.4), capture=True)
        assert not torch.equal(native.image, result.image)
        model.context_mode = 'off'
        off = model(image, RateControl(.4))
        torch.testing.assert_close(off.image, result.image, rtol=0, atol=0)
        model.context_mode = 'native'
        model.codec.update(force=True)
        packet = model.compress(image[:1], RateControl(.4))
        decoded = model.decompress(packet.to_bytes())
        direct = model(image[:1], RateControl(.4)).image
        torch.testing.assert_close(decoded, direct, rtol=0, atol=1e-5)


def test_sf_whitelist_and_RD_codec_gradient_are_independent():
    model = context_model().train()
    model.checkpoint_blocks = True
    teacher = ContextTeacher(model)
    image = torch.rand(4, 3, 256, 256)*2-1
    result = model(image, RateControl(.2), capture=True)
    rng = torch.get_rng_state().clone()
    views = model.context_views(result, RateControl(.2), teacher, settings(), 10000, 0, 0, quality_b=1.)
    assert torch.equal(rng, torch.get_rng_state())
    assert not views.target.requires_grad and not views.mixed.requires_grad
    assert len(views.indices) == 2 and views.mixed.shape[0] == 2
    native = views.native_loss(result.feature)
    accumulate_sf(native, model.sf_parameters(), retain_graph=True)
    allowed = {id(p) for p in model.sf_parameters()}
    for p in model.parameters():
        assert p.grad is None or id(p) in allowed
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.context_generator.parameters())
    rd = result.image.square().mean()+result.coarse.square().mean()
    expected = torch.autograd.grad(rd, tuple(model.codec.parameters()), retain_graph=True, allow_unused=True)
    rd.backward()
    for p, g in zip(model.codec.parameters(), expected):
        if g is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.codec.router.parameters())
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.context_readout.parameters())
    model.zero_grad(set_to_none=True)
    accumulate_sf(model.mixed_context_loss(views, 4), model.sf_parameters())
    assert all(p.grad is None or id(p) in allowed for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    equal = model.context_views(result, RateControl(.2), teacher, settings(), 10000, 0, 0, quality_b=.2)
    assert equal.mixed is None and equal.indices.numel() == 0
    assert equal.native_loss(result.feature).item() > 0  # same-quality SF stays enabled


def test_two_full_analyses_and_coding_aligned_half_batch_mix():
    model = context_model()
    calls, observed = [], []
    class Codec(nn.Module):
        def analyze(self, clean, q):
            calls.append(q.value)
            return SimpleNamespace(clean=clean, q=q.value)
        def quantize_analysis(self, source, q, **kwargs):
            assert source.q == q.value
            return {'mean': source.clean+q.value, 'delta': torch.ones(4,1,4,4)*2}
    def teacher(latent, text, time, logstep):
        observed.append(latent.clone())
        return torch.ones(4,16,4)
    model.codec = Codec()
    a = torch.zeros(4,4,8,8,requires_grad=True)
    result = SimpleNamespace(coarse=a, clean=torch.ones_like(a), scale_info={'log_step': torch.zeros(4,1,4,4)})
    v = model.context_views(result, RateControl(.4), teacher, settings(), 10000, 0, 0, quality_b=1.)
    assert calls == [1.] and v.mixed.shape == (2,4,8,8)
    assert torch.all((v.mixed[:,0] != 0).sum((1,2)) == 16)
    blocks = v.mixed[:,0].reshape(2,4,2,4,2)
    assert torch.equal(blocks, blocks[:,:,0:1,:,0:1].expand_as(blocks))
    assert torch.all(v.mixed_log_step.flatten(1).count_nonzero(1) == 4)
    assert all(torch.equal(x, torch.full_like(x,2.)) for x in observed)


def test_per_image_view_average_and_ema_new_parameters():
    s,t = torch.randn(4,16,32),torch.randn(4,16,32)
    v = ContextViews(t,torch.tensor([1,3]),torch.zeros(2,4,8,8),None,.6)
    d = context_distance(s,t)
    torch.testing.assert_close(v.native_loss(s),(d[0]+.5*d[1]+d[2]+.5*d[3])/4)
    model = context_model()
    ema = DeploymentEMA(model,decay=.5)
    before = clone_state(ema.state)
    teacher = ContextTeacher(model)
    with torch.no_grad():
        model.context_generator.mlp[0].weight.add_(1)
        model.context_readout.gates[0].add_(1)
    ema.update(model)
    teacher.update_context_student(model,.5)
    torch.testing.assert_close(ema.state['context_readout']['gates.0'],torch.tensor(.5))
    torch.testing.assert_close(teacher.context_readout.gates[0],torch.tensor(.5))
    torch.testing.assert_close(ema.state['context_generator']['mlp.0.weight'],before['context_generator']['mlp.0.weight']+.5)


def test_real_trainer_whitelisted_backwards_and_exact_resume(tmp_path):
    make_images(tmp_path)
    config = small_config(tmp_path,'joint')
    config['model']['backend'] = 'C'
    config['self_flow'] = settings()['self_flow']
    config['quality'] = settings()['quality']
    config['objective'] = dict(base_weight=.1,repaired_weight=0.)
    config['train'].update(steps=3,batch_size=2,accumulation_steps=1,effective_batch=2,router_ramp_steps=0)
    config['image_loss'] = dict(mse_weight=4.,lpips_weight=0.,dists_weight=0.)
    state = clone_state(context_model().export_state(),cpu=True)
    initial = tmp_path/'backend_init.pt'
    save_atomic(initial,dict(method=METHOD,schema=SCHEMA,phase='backend_init',backend='shared',step=0,raw=state,config=config,assets={'test':'context'}))
    args = SimpleNamespace(resume=None,init_checkpoint=str(initial),max_updates=None)
    def factory(cfg,device):
        model=context_model().to(device)
        model.vae.requires_grad_(False)
        model.checkpoint_blocks=True
        return model
    with patch.object(trainer,'make_model',side_effect=factory), patch.object(trainer,'asset_identity',return_value={'test':'context'}), redirect_stdout(io.StringIO()):
        trainer.train(config,args,0,1,torch.device('cpu'))
        split=deepcopy(config)
        split['train']['output']=str(tmp_path/'split')
        args.max_updates=2
        trainer.train(split,args,0,1,torch.device('cpu'))
        args.resume,args.init_checkpoint,args.max_updates=str(tmp_path/'split/paused.pt'),None,None
        trainer.train(split,args,0,1,torch.device('cpu'))
    full=read_checkpoint(tmp_path/'joint/complete.pt')
    resumed=read_checkpoint(tmp_path/'split/complete.pt')
    for kind in ('raw','ema'):
        for part,values in full[kind].items():
            for key,value in values.items():
                assert torch.equal(value,resumed[kind][part][key]),(kind,part,key)
    for key,value in full['teacher'].items():
        assert torch.equal(value,resumed['teacher'][key]),key
    assert resumed['context_sf']['teacher_initialized']
    logs=[json.loads(line) for line in (tmp_path/'split/train.jsonl').read_text().splitlines()]
    assert all(log['sf_weight']==1 for log in logs)


def test_migration_keeps_shared_moments_and_initializes_only_new_context():
    from test_c_sf_migration import source_state, TinyC
    from scripts.migrate_c_sf_batch8 import parameter_layout
    from scripts.migrate_context_sf import migrate
    class TinyContext(TinyC):
        def __init__(self):
            super().__init__()
            self.context_generator=nn.Linear(3,3)
            self.context_readout=ContextReadout(2)
        def export_state(self):
            return dict(super().export_state(),context_generator=self.context_generator.state_dict(),
                        context_readout=self.context_readout.state_dict())
        def load_bt_state(self,state):
            super().load_bt_state(state)
            self.context_generator.load_state_dict(state['context_generator'])
            self.context_readout.load_state_dict(state['context_readout'])
    source,old_layout,_=source_state()
    model=TinyContext()
    migrated=migrate(source,model,settings(),SimpleNamespace(fingerprint='same'),{},'pure-C10K')
    assert migrated['context_sf']==dict(protocol='context_sf_v1',teacher_initialized=False)
    assert migrated['step']==migrated['micro_seen']==10000
    assert migrated['migration']['preserved_global_samples']==80000
    assert migrated['scheduler']==source['scheduler']
    for kind in ('raw','ema'):
        for part,state in source[kind].items():
            for key,value in state.items():
                assert torch.equal(value,migrated[kind][part][key])
    _,_,new_layout,_=parameter_layout(model,settings())
    old={name:source['optimizer']['state'][ident] for group,(_,params) in zip(source['optimizer']['param_groups'],old_layout)
         for ident,(name,_) in zip(group['params'],params)}
    for group,(_,params) in zip(migrated['optimizer']['param_groups'],new_layout):
        for ident,(name,_) in zip(group['params'],params):
            if name.startswith(('context_generator.','context_readout.')):
                assert ident not in migrated['optimizer']['state']
            else:
                assert all(torch.equal(v,migrated['optimizer']['state'][ident][key]) for key,v in old[name].items())
