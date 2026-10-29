"""Native output, codec gradients, detached mixed views and exact resume."""
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

from models.bt_self_flow.checkpoint import METHOD, SCHEMA, clone_state, read_checkpoint, save_atomic
from models.bt_self_flow.codec import BTLatentCodec
from models.bt_self_flow.experiment import load_config
from models.bt_self_flow.native_relational import (CNativeRelationalCodec, MIGRATION,
    relation_distance, spatial_relation)
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.quantization_residual import CResidualTeacher
from models.sana_self_flow import unwrap_sana_transformer
from test_bt_backend import tiny_model
from test_bt_training import small_config
from test_quantization_residual_sf import make_images
import train_bt_self_flow as trainer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def relational_model():
    model = tiny_model("C")
    model.__class__ = CNativeRelationalCodec
    model.codec = BTLatentCodec(ch_emd=4, channel=16, channel_out=4,
                               precision_conditioning=True, rate_likelihood="noise")
    return model


def settings():
    return load_config(ROOT/"configs/bt_self_flow/c_native_relational.json")


@pytest.fixture(autouse=True)
def tiny_ae_decode(monkeypatch):
    # TinyAE is a single convolution, without DC-AE's checkpointing API.
    # Keep the production DiT capture/prefix checkpointing intact.
    monkeypatch.setattr("models.bt_self_flow.model.decode_latent",
                        lambda vae, latent, checkpoint_blocks: vae.decode(latent).sample)


def test_relation_ignores_channel_basis_scale_and_common_offset_but_not_content():
    h = torch.randn(3, 16, 8, requires_grad=True)
    rotation = torch.linalg.qr(torch.randn(8, 8))[0]
    target = spatial_relation((h.detach() @ rotation)*3+torch.randn(3, 1, 8))
    same = relation_distance(spatial_relation(h), target)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=2e-6, rtol=0)
    wrong = relation_distance(spatial_relation(h), target.roll(1, 0))
    shuffled = relation_distance(spatial_relation(h[:, torch.randperm(16)]), target)
    assert wrong.mean() > .1 and shuffled.mean() > .1
    wrong.mean().backward()
    assert h.grad.norm() > 0 and torch.isfinite(h.grad).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert spatial_relation(h).dtype == torch.float32


@pytest.mark.parametrize("checkpointed", [False, True])
def test_capture_preserves_native_C_and_parameter_layout(checkpointed):
    model = relational_model().eval()
    before = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
    model.checkpoint_blocks = checkpointed
    image = torch.rand(2, 3, 256, 256)*2-1
    with torch.no_grad():
        native = model(image, RateControl(.4), capture=False)
        captured = model(image, RateControl(.4), capture=True)
    torch.testing.assert_close(captured.image, native.image, rtol=0, atol=0)
    assert captured.feature is not None and captured.feature.shape[:2] == (2, 64)
    assert model.quality_conditioner is None and model.representation_projector is None
    assert before == [(n, tuple(p.shape)) for n, p in model.named_parameters()]
    assert set(model.export_state()) == {"codec", "distortion_scale", "dit", "vae", "text"}


def test_native_sf_trains_codec_but_mixed_sf_trains_only_shared_prefix():
    model = relational_model().train()
    model.checkpoint_blocks = True
    teacher = CResidualTeacher(model.DiT, model.text_proj, model.teacher_layer)
    image = torch.rand(2, 3, 256, 256)*2-1
    result = model(image, RateControl(.2), capture=True)
    rng = torch.get_rng_state().clone()
    views = model.relational_views(result, RateControl(.2), teacher, settings(), 10000, 0, 0, quality_b=1.)
    assert torch.equal(rng, torch.get_rng_state())
    assert not views.target.requires_grad and not views.mixed.requires_grad
    native = relation_distance(spatial_relation(result.feature), views.target).mean()
    native.backward()
    for part in (model.codec.g_a, model.codec.g_s, model.codec.router):
        assert any(p.grad is not None and p.grad.norm() > 0 for p in part.parameters())
    blocks = unwrap_sana_transformer(model.DiT).transformer_blocks
    assert all(p.grad is None for p in blocks[model.student_layer+1:].parameters())
    assert all(p.grad is None for p in model.vae.parameters())
    model.zero_grad(set_to_none=True)
    model.mixed_relation(views).backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in blocks[:model.student_layer+1].parameters())
    assert all(p.grad is None for p in model.codec.parameters())
    assert all(p.grad is None for p in blocks[model.student_layer+1:].parameters())
    assert all(p.grad is None for p in model.vae.parameters())
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())


def test_two_native_analyses_all_images_mixed_and_equal_quality_skips_replay():
    model = relational_model()
    calls, observed = [], []
    class Codec(nn.Module):
        def analyze(self, clean, q):
            calls.append(q.value)
            return SimpleNamespace(clean=clean, q=q.value)
        def quantize_analysis(self, source, q, **kwargs):
            assert source.q == q.value
            return {"mean": source.clean+q.value}
    def teacher(latent, *unused):
        observed.append(latent.clone())
        return latent.flatten(2).transpose(1, 2)
    model.codec = Codec()
    a = torch.zeros(4, 4, 8, 8, requires_grad=True)
    result = SimpleNamespace(coarse=a, clean=torch.ones_like(a, requires_grad=True))
    v = model.relational_views(result, RateControl(.4), teacher, settings(), 10000, 0, 0, quality_b=1.)
    assert calls == [1.] and v.mixed.shape == a.shape and v.native_weight == .5
    assert torch.all((v.mixed[:, 0] != 0).sum((1, 2)) == 16)
    assert torch.equal(v.mixed[:, :1].expand_as(v.mixed), v.mixed)
    assert all(torch.equal(x, torch.full_like(x, 2.)) for x in observed)
    calls.clear()
    v = model.relational_views(result, RateControl(.4), teacher, settings(), 10000, 0, 0, quality_b=.4)
    assert not calls and v.mixed is None and v.native_weight == 1.


def test_migration_preserves_all_C_parameters_moments_and_sample_progress():
    from test_c_sf_migration import source_state, TinyC
    from scripts.migrate_native_relational import migrate
    source, _, _ = source_state()
    result = migrate(source, TinyC(), settings(), SimpleNamespace(fingerprint="same"), {}, "pure-C10K")
    assert result["migration"]["protocol"] == MIGRATION
    assert result["micro_seen"] == result["step"] == 10000
    assert result["migration"]["preserved_global_samples"] == 80000
    assert result["native_relational"]["teacher_initialized"] is False
    assert "teacher" not in result and result["scaler"]["scale"] == 64.
    for kind in ("raw", "ema"):
        for part, values in source[kind].items():
            for key, value in values.items():
                torch.testing.assert_close(result[kind][part][key], value, rtol=0, atol=0)
    assert len(result["optimizer"]["param_groups"]) == len(source["optimizer"]["param_groups"])
    for old, new in zip(source["optimizer"]["state"].values(), result["optimizer"]["state"].values()):
        for key in old:
            torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    assert result["scheduler"] == source["scheduler"]


def test_real_trainer_sequential_backwards_and_exact_teacher_resume(tmp_path):
    make_images(tmp_path)
    config = small_config(tmp_path, "joint")
    config["model"]["backend"] = "C"
    config["self_flow"] = dict(settings()["self_flow"], warmup=0, ramp=2)
    config["quality"] = settings()["quality"]
    config["objective"] = dict(base_weight=.1, repaired_weight=0.)
    config["train"].update(steps=3, batch_size=2, accumulation_steps=1, effective_batch=2,
                           router_ramp_steps=0)
    config["image_loss"] = dict(mse_weight=4., lpips_weight=0., dists_weight=0.)
    state = clone_state(relational_model().export_state(), cpu=True)
    initial = tmp_path/"backend_init.pt"
    save_atomic(initial, dict(method=METHOD, schema=SCHEMA, phase="backend_init", backend="shared",
        step=0, raw=state, config=config, assets={"test": "native_relation"}))
    args = SimpleNamespace(resume=None, init_checkpoint=str(initial), max_updates=None)
    def factory(cfg, device):
        model = relational_model().to(device)
        model.vae.requires_grad_(False)
        model.checkpoint_blocks = True
        return model
    with patch.object(trainer, "make_model", side_effect=factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "native_relation"}), \
         redirect_stdout(io.StringIO()):
        trainer.train(config, args, 0, 1, torch.device("cpu"))
        split = deepcopy(config)
        split["train"]["output"] = str(tmp_path/"split")
        args.max_updates = 2
        trainer.train(split, args, 0, 1, torch.device("cpu"))
        args.resume, args.init_checkpoint, args.max_updates = str(tmp_path/"split/paused.pt"), None, None
        trainer.train(split, args, 0, 1, torch.device("cpu"))
    full = read_checkpoint(tmp_path/"joint/complete.pt")
    resumed = read_checkpoint(tmp_path/"split/complete.pt")
    for kind in ("raw", "ema"):
        for part, values in full[kind].items():
            for key, value in values.items():
                assert torch.equal(value, resumed[kind][part][key]), (kind, part, key)
    for key, value in full["teacher"].items():
        assert torch.equal(value, resumed["teacher"][key]), key
    logs = [json.loads(line) for line in (tmp_path/"split/train.jsonl").read_text().splitlines()]
    assert logs[-1]["sf_weight"] == 1.
    assert resumed["native_relational"]["teacher_initialized"] is True
