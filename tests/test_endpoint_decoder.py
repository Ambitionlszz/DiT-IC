"""Endpoint SF preserves the native receiver and captures a live AE feature."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from models.bt_self_flow.decoder import (
    decode_latent, decode_latent_prefix, decode_latent_with_endpoint_feature,
    decode_latent_with_precision_mu_adapter,
)
from models.bt_self_flow.endpoint_model import EndpointAblationCodec
from models.bt_self_flow.model import BTCodec
from models.bt_self_flow.packet import RateControl
from test_pixel_adapter import SMALL_PIXEL, activate_output, real_small_ae
from test_precision_mu_pixel_adapter import precision_model


torch.set_num_threads(1)


def tiny_endpoint_model(*, sf=True, d=True, device="cpu", checkpoint_blocks=True, train_rgb_head=False):
    def initialize(self, *args, **kwargs):
        base = precision_model()
        del base.pixel_adapter
        self.__dict__.update(base.__dict__)
        self.checkpoint_blocks = checkpoint_blocks

    pixel = dict(SMALL_PIXEL, seed=903, checkpoint_blocks=checkpoint_blocks) if d else None
    with patch.object(BTCodec, "__init__", initialize):
        return EndpointAblationCodec(endpoint_config={"enabled": sf}, pixel_config=pixel,
                                    train_rgb_head=train_rgb_head).to(device)


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_real_lora_ae_capture_matches_native_and_retains_same_prefix(checkpoint_blocks):
    torch.manual_seed(54)
    ae = real_small_ae().train()
    latent = torch.randn(1, 4, 2, 3, requires_grad=True)
    with torch.no_grad():
        expected = ae.decode(latent).sample
    image, feature = decode_latent_with_endpoint_feature(ae, latent, checkpoint_blocks=checkpoint_blocks)
    prefix = decode_latent_prefix(ae, latent, checkpoint_blocks=checkpoint_blocks)
    assert feature.shape == (1, 8, 8, 12)
    assert feature.requires_grad
    torch.testing.assert_close(image, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(feature, prefix, atol=0, rtol=0)
    (feature.square().mean()+image.square().mean()).backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum()>0
    assert any(p.grad is not None for p in ae.parameters() if p.requires_grad)


def test_feature_and_image_gradients_match_with_and_without_checkpoint_recomputation():
    torch.manual_seed(56)
    a, b = real_small_ae().train(), None
    b = deepcopy(a)
    x = torch.randn(1, 4, 2, 2, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    image_a, feature_a = decode_latent_with_endpoint_feature(a, x, checkpoint_blocks=False)
    image_b, feature_b = decode_latent_with_endpoint_feature(b, y, checkpoint_blocks=True)
    (image_a.square().mean()+feature_a.abs().mean()).backward()
    (image_b.square().mean()+feature_b.abs().mean()).backward()
    torch.testing.assert_close(image_a, image_b, atol=0, rtol=0)
    torch.testing.assert_close(feature_a, feature_b, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=2e-6)
    for (name_a, p_a), (name_b, p_b) in zip(a.named_parameters(), b.named_parameters()):
        assert name_a == name_b
        if p_a.grad is not None:
            torch.testing.assert_close(p_a.grad, p_b.grad, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_d_capture_preserves_original_nonzero_adapter_and_its_context(checkpoint_blocks):
    model = tiny_endpoint_model(checkpoint_blocks=checkpoint_blocks).train()
    activate_output(model.pixel_adapter)
    latent = torch.randn(1, 4, 2, 3, requires_grad=True)
    mu = torch.randn_like(latent)
    delta = torch.exp(torch.randn(1, 1, 1, 2))
    precision = model.codec.precision_embedding(delta.log())
    native, _ = decode_latent_with_precision_mu_adapter(model.vae, latent,
        model.pixel_adapter, mu, precision, checkpoint_blocks, False)
    captured, feature = decode_latent_with_endpoint_feature(model.vae, latent,
        model.pixel_adapter, (mu, precision), checkpoint_blocks)
    torch.testing.assert_close(captured, native, atol=0, rtol=0)
    (captured.square().mean()+feature.square().mean()).backward()
    assert latent.grad is not None and latent.grad.abs().sum()>0
    assert model.pixel_adapter.pixel_output.weight.grad.abs().sum()>0


@pytest.mark.parametrize("d", [False, True])
def test_receiver_captures_actual_forward_feature_and_has_no_added_inference_parameters(d):
    model = tiny_endpoint_model(d=d).eval()
    mu = torch.randn(1, 4, 4, 6)
    delta = torch.exp(torch.randn(1, 1, 2, 3))
    if d:
        activate_output(model.pixel_adapter)
    base = model.vae.get_base_model()
    calls = []
    hook = base.decoder.up_blocks[3].register_forward_hook(lambda m, a, value: calls.append(value))
    with torch.no_grad():
        plain = model.decode_received(mu, delta, capture=False)
        calls.clear()
        captured = model.decode_received(mu, delta, capture=True)
        assert len(calls) == 1  # one student AE forward, no auxiliary replay
        assert captured["feature"] is calls[0]
    hook.remove()
    for name in ("prediction", "latent", "image"):
        torch.testing.assert_close(plain[name], captured[name], atol=2e-6, rtol=2e-6)
    assert plain["feature"] is None
    with torch.no_grad():
        torch.testing.assert_close(captured["feature"], model.endpoint_feature(captured["latent"]),
                                   atol=0, rtol=0)
    assert not hasattr(model, "spatial_projector")
    assert not hasattr(model, "spatial_settings")
    assert "spatial_projector" not in model.export_state()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    model.sf_enabled = False
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_d_switch_changes_only_image_and_not_the_endpoint_feature():
    model = tiny_endpoint_model().eval()
    activate_output(model.pixel_adapter)
    mu, delta = torch.randn(1, 4, 4, 4), torch.ones(1, 1, 2, 2)
    with torch.no_grad():
        on = model.decode_received(mu, delta, capture=True)
        with model.decoder_switches(d=False):
            off = model.decode_received(mu, delta, capture=True)
    assert model.d_enabled
    assert torch.equal(on["latent"], off["latent"])
    assert torch.equal(on["feature"], off["feature"])
    assert not torch.equal(on["image"], off["image"])
    with pytest.raises(ValueError, match="absent"):
        with tiny_endpoint_model(d=False).decoder_switches(d=True):
            pass


def test_codec_forward_uses_single_native_prediction_and_exposes_graph_to_sana():
    model = tiny_endpoint_model().train()
    image = torch.randn(1, 3, 256, 256)
    before = model.prediction_calls
    result = model(image, RateControl(.4), capture=True)
    assert model.prediction_calls-before == 1
    assert result.feature.shape == (1, 8, 32, 32)
    assert result.feature.requires_grad
    parameters = [p for p in model.DiT.parameters() if p.requires_grad]
    grads = torch.autograd.grad(result.feature.square().mean(), parameters, allow_unused=True)
    assert any(g is not None and g.abs().sum()>0 for g in grads)


def test_shared_state_roundtrip_and_strict_d_projector_compatibility():
    model = tiny_endpoint_model()
    state = model.export_state()
    other = tiny_endpoint_model()
    other.load_bt_state(state)
    for part, values in state.items():
        for name, value in values.items():
            assert torch.equal(value, other.export_state()[part][name])
    native = {part: values for part, values in state.items() if part != "pixel_adapter"}
    with pytest.raises(ValueError, match="missing"):
        other.load_bt_state(native)
    other.load_bt_state(native, initialize_pixel_adapter=True)
    with pytest.raises(ValueError, match="absent"):
        tiny_endpoint_model(d=False).load_bt_state(state)
    for code in ("qk_source_code", "precision_adaln_version"):
        bad = deepcopy(state)
        bad["pixel_adapter"][code].add_(1)
        with pytest.raises(ValueError, match="incompatible"):
            other.load_bt_state(bad)
    with pytest.raises(ValueError, match="projector"):
        other.load_bt_state(dict(state, spatial_projector={}))


def test_prefix_rejects_tiling_or_wrong_decoder_and_target_can_be_no_grad():
    ae = real_small_ae()
    latent = torch.randn(1, 4, 2, 3, requires_grad=True)
    with torch.no_grad():
        target = decode_latent_prefix(ae, latent)
    assert not target.requires_grad
    ae.get_base_model().use_tiling = True
    with pytest.raises(ValueError, match="whole-image"):
        decode_latent_prefix(ae, latent)
    ae.get_base_model().use_tiling = False
    ae.get_base_model().decoder.up_blocks = torch.nn.ModuleList(list(ae.get_base_model().decoder.up_blocks)[:5])
    with pytest.raises(ValueError, match="six-stage"):
        decode_latent_prefix(ae, latent)
