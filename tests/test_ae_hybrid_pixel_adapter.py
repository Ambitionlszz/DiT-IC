"""Hybrid D: live AE conditioning, sequential SA/CA and receiver gradients."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.bt_self_flow.endpoint_model import EndpointAblationCodec, HYBRID_D_METHOD
from models.bt_self_flow.endpoint_self_flow import EndpointEMATeacher, sf_parameters
from models.bt_self_flow.endpoint_training import endpoint_backward
from models.bt_self_flow.losses import ImageLoss, reconstruction_objective
from models.bt_self_flow.model import BTCodec, pad_image
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.pixel_adapter import (
    AEConditionedHybridPixelAdapter, HybridRegionAttention, SharedPixelStateAdaLN,
    rotary_position,
)
from test_pixel_adapter import SMALL_PIXEL, activate_output
from test_precision_mu_pixel_adapter import precision_model
from test_endpoint_training import endpoint_config


torch.set_num_threads(1)


def small_adapter(**kwargs):
    settings = dict(pixel_dim=4, patch_size=4, context_dim=16,
                    attention_dim=16, heads=2, checkpoint_blocks=False)
    settings.update(kwargs)
    return AEConditionedHybridPixelAdapter(8, 8, **settings)


def hybrid_endpoint(*, sf=True, checkpoint_blocks=False):
    def initialize(self, *args, **kwargs):
        original = precision_model()
        del original.pixel_adapter
        self.__dict__.update(original.__dict__)
        self.checkpoint_blocks = checkpoint_blocks

    with patch.object(BTCodec, "__init__", initialize):
        return EndpointAblationCodec(
            endpoint_config={"enabled": sf}, pixel_method=HYBRID_D_METHOD,
            pixel_config=dict(SMALL_PIXEL, seed=903, checkpoint_blocks=checkpoint_blocks))


def inputs(requires_grad=False):
    return [torch.randn(*shape).requires_grad_(requires_grad)
            for shape in ((1, 8, 16, 24), (1, 8, 8, 12), (1, 128, 2, 3))]


def test_default_parameter_budget_and_nonzero_internal_initialization():
    # Meta construction counts the full deployment architecture without a
    # high-resolution forward or GPU allocation.
    with torch.device("meta"):
        model = AEConditionedHybridPixelAdapter()
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_478_000
    assert len(model.blocks) == 2
    for block in model.blocks:
        assert sum(parameter.numel() for parameter in block.modulation.parameters()) == 25_056
    small = small_adapter()
    assert not small.pixel_output.weight.count_nonzero()
    assert not small.pixel_output.bias.count_nonzero()
    assert not small.precision_projection.weight.count_nonzero()
    for block in small.blocks:
        assert block.modulation.output[0].weight.count_nonzero()
        assert block.modulation.output[1].weight.count_nonzero()
        for convolution in block.modulation.spatial:
            source = torch.randn(1, 4, 5, 7)
            torch.testing.assert_close(convolution(source), source, atol=0, rtol=0)
    for norm in (module for module in small.modules() if isinstance(module, nn.LayerNorm)):
        assert norm.eps == 1e-6
        assert torch.equal(norm.weight, torch.ones_like(norm.weight))
        assert torch.equal(norm.bias, torch.zeros_like(norm.bias))


def test_condition_projects_precision_on_native_grid_and_aligns_coding_cells():
    torch.manual_seed(71)
    model = small_adapter()
    content = torch.randn(1, 8, 8, 12)
    precision = torch.randn(1, 128, 2, 3)
    initial = model.condition(content, precision, (4, 6))
    torch.testing.assert_close(initial, model.condition(content, precision * 100, (4, 6)), atol=0, rtol=0)
    nn.init.normal_(model.precision_projection.weight, std=.03)
    received = []
    hook = model.precision_projection.register_forward_pre_hook(lambda module, args: received.append(args[0].shape))
    actual = model.condition(content, precision, (4, 6))
    hook.remove()
    assert received == [torch.Size((1, 2, 3, 128))]
    projected_content = model.content_norm(model.content_projection(F.avg_pool2d(content, 2).movedim(1, -1)))
    projected_precision = model.precision_projection(precision.movedim(1, -1))
    aligned = projected_precision.repeat_interleave(2, 1).repeat_interleave(2, 2)
    expected = F.silu(projected_content + aligned).reshape(1, 24, 16)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, initial)
    assert not torch.equal(actual, model.condition(content.flip(-1), precision, (4, 6)))


def test_adaln_state_reads_neighbors_across_region_boundary():
    generator = SharedPixelStateAdaLN(pixel_dim=4, context_dim=16, patch_size=2)
    # Two horizontal 2x2 regions. Position x=1 must read x=2, across their
    # boundary, rather than wrapping within an independently processed patch.
    grid = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4).expand(1, 4, 2, 4)
    packed = grid.reshape(1, 4, 1, 2, 2, 2).permute(0, 2, 4, 3, 5, 1).reshape(1, 2, 4, 4)
    with torch.no_grad():
        generator.spatial[0].weight.zero_()
        generator.spatial[0].weight[:, 0, 1, 2] = 1
        generator.state[0].weight.zero_()
        generator.state[0].weight[0, 0] = 1
        generator.output[0].weight.zero_()
        generator.output[0].bias.zero_()
        generator.output[0].weight[0, 0] = 1
    shift, scale, gate = generator(packed, torch.zeros(1, 2, 64), (1, 2), 0)
    assert shift[0, 0, 1, 0] == F.silu(grid[0, 0, 0, 2])
    assert shift[0, 0, 3, 0] == F.silu(grid[0, 0, 1, 2])
    assert not scale.count_nonzero() and not gate.count_nonzero()


def test_cross_attention_queries_use_sa_updated_state_and_expand_only_updates():
    torch.manual_seed(72)
    attention = HybridRegionAttention(16, 2)
    tokens, content = torch.randn(1, 6, 16), torch.randn(1, 6, 16)
    position = rotary_position(8, 2, 3, "cpu")
    observed = {}
    handles = []
    for name, module in (("query", attention.cross_attn.q),
                         ("key", attention.cross_attn.k), ("value", attention.cross_attn.v)):
        handles.append(module.register_forward_pre_hook(
            lambda module, args, name=name: observed.__setitem__(name, args[0].detach().clone())))
    update = attention(tokens, content, position)
    for handle in handles:
        handle.remove()
    self_update = attention.self_attn(attention.self_norm(tokens), attention.self_norm(tokens), position)
    expected_query = attention.cross_norm(tokens + self_update)
    torch.testing.assert_close(observed["query"], expected_query, atol=0, rtol=0)
    assert torch.equal(observed["key"], content) and torch.equal(observed["value"], content)
    cross_update = attention.cross_attn(expected_query, content, position)
    torch.testing.assert_close(update, self_update + cross_update, atol=0, rtol=0)
    assert not torch.equal(observed["query"], attention.cross_norm(tokens))

    model = small_adapter()
    seen = []
    hook = model.blocks[0].expand_from_attn.register_forward_hook(lambda *args: seen.append(1))
    model(*inputs())
    hook.remove()
    assert seen == [1]


def test_zero_output_then_real_updates_open_all_branches_and_condition_gradients():
    torch.manual_seed(73)
    model = small_adapter()
    values = inputs(requires_grad=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        for value in values:
            value.grad = None
        result, captured = model(*values, return_features=True)
        if step == 0:
            assert torch.equal(result, values[0])
        assert captured["token_hw"] == (4, 6)
        assert captured["pixel_tokens"].shape == (1, 24, 16, 4)
        result.square().mean().backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            if step >= 1:
                assert parameter.grad.abs().sum() > 0, name
        if step == 0:
            assert model.blocks[0].attn.self_attn.q.weight.grad.count_nonzero() == 0
        if step >= 1:
            assert values[1].grad.abs().sum() > 0
        if step >= 2:
            assert values[2].grad.abs().sum() > 0
        optimizer.step()


@pytest.mark.parametrize("bf16", [False, True])
def test_checkpoint_preserves_forward_all_parameter_and_input_gradients(bf16):
    torch.manual_seed(74)
    plain = small_adapter()
    activate_output(plain)
    nn.init.normal_(plain.precision_projection.weight, std=.02)
    checked = deepcopy(plain)
    checked.checkpoint_blocks = True
    source = inputs()
    results = []
    for model in (plain, checked):
        values = [value.clone().requires_grad_() for value in source]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16, cache_enabled=False):
            result = model(*values)
        result.square().mean().backward()
        results.append((result.detach(), *[value.grad for value in values]))
    for left, right in zip(*results):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    for (name, left), (_, right) in zip(plain.named_parameters(), checked.named_parameters()):
        assert left.grad is not None and torch.isfinite(left.grad).all(), name
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_receiver_uses_one_live_ae_condition_with_sf_off_or_capture_disabled(checkpoint_blocks):
    torch.manual_seed(75)
    model = hybrid_endpoint(sf=False, checkpoint_blocks=checkpoint_blocks).train()
    activate_output(model.pixel_adapter)
    mu, delta = torch.randn(1, 4, 2, 4, requires_grad=True), torch.ones(1, 1, 1, 2)
    captured, conditions = [], []
    base = model.vae.get_base_model()
    handles = [base.decoder.up_blocks[3].register_forward_hook(lambda module, args, value: captured.append(value)),
               model.pixel_adapter.register_forward_pre_hook(lambda module, args: conditions.append(args))]
    result = model.decode_received(mu, delta, capture=False)
    assert len(captured) == len(conditions) == 1
    assert conditions[0][1] is captured[0]
    assert conditions[0][1].requires_grad
    assert result["feature"] is None
    torch.testing.assert_close(conditions[0][2], model.codec.precision_embedding(delta.log()), atol=0, rtol=0)
    captured[0].retain_grad()
    result["image"].square().mean().backward()
    assert captured[0].grad is not None and captured[0].grad.abs().sum() > 0
    assert mu.grad is not None and mu.grad.abs().sum() > 0
    assert model.pixel_adapter.content_projection.weight.grad.abs().sum() > 0
    assert model.pixel_adapter.blocks[1].attn.cross_attn.k.weight.grad.abs().sum() > 0
    for handle in handles:
        handle.remove()


def test_zero_initialized_receiver_matches_c_and_d_does_not_change_sf_feature():
    model = hybrid_endpoint().eval()
    mu, delta = torch.randn(1, 4, 2, 4), torch.ones(1, 1, 1, 2)
    with torch.no_grad():
        initial = model.decode_received(mu, delta, capture=True)
        with model.decoder_switches(d=False):
            native = model.decode_received(mu, delta, capture=True)
        for name in ("prediction", "latent", "image", "feature"):
            torch.testing.assert_close(initial[name], native[name], atol=0, rtol=0)
        activate_output(model.pixel_adapter)
        active = model.decode_received(mu, delta, capture=True)
    assert torch.equal(active["feature"], native["feature"])
    assert torch.equal(active["latent"], native["latent"])
    assert not torch.equal(active["image"], native["image"])


def test_real_packet_highest_quality_and_fresh_receiver_need_no_sender_or_sf_capture():
    torch.manual_seed(76)
    model = hybrid_endpoint(sf=False).eval()
    activate_output(model.pixel_adapter)
    receiver = hybrid_endpoint(sf=False).eval()
    receiver.load_bt_state(model.export_state())
    image, control = torch.rand(1, 3, 65, 97) * 2 - 1, RateControl(1.)
    with torch.no_grad():
        packet = model.compress(image, control)
        expected = model(pad_image(image), control).image[..., :65, :97]
    conditions, captures = [], []
    base = receiver.vae.get_base_model()
    handles = [base.decoder.up_blocks[3].register_forward_hook(lambda module, args, value: captures.append(value)),
               receiver.pixel_adapter.register_forward_pre_hook(lambda module, args: conditions.append(args))]
    before = receiver.prediction_calls
    with patch.object(receiver, "encode_clean_latent", side_effect=AssertionError("receiver accessed source image")), \
         patch.object(receiver.codec.router, "forward", side_effect=AssertionError("receiver accessed sender planner")):
        actual = receiver.decompress(packet.to_bytes())
    for handle in handles:
        handle.remove()
    assert receiver.prediction_calls - before == 1
    assert len(conditions) == len(captures) == 1
    assert conditions[0][1] is captures[0]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_actual_endpoint_sf_adds_sana_gradients_but_preserves_hybrid_d_rd_gradients():
    torch.manual_seed(77)
    model = hybrid_endpoint(sf=True, checkpoint_blocks=True).train()
    activate_output(model.pixel_adapter)
    nn.init.normal_(model.pixel_adapter.precision_projection.weight, std=.02)
    reference = deepcopy(model)
    teacher = EndpointEMATeacher(model)
    image, control = torch.randn(1, 3, 256, 256), RateControl(.4)
    config = endpoint_config()
    image_loss = ImageLoss(lpips_weight=0., dists_weight=0., mse_weight=4.)
    rng = torch.get_rng_state().clone()
    result, _, metrics = endpoint_backward(
        model, teacher, image, control, image_loss, config, 0, 0, 1,
        torch.amp.GradScaler("cuda", enabled=False), torch.bfloat16, full_precision=True)
    # The native codec estimates training likelihoods using sampled noise.
    # Match that noise before attributing gradient differences to SF.
    torch.set_rng_state(rng)
    baseline = reference(image, rate_control=control)
    rd, _ = reconstruction_objective(
        baseline, image, control, reference.distortion_scale.s_data, image_loss,
        objective=config["objective"], quality=config["quality"])
    rd.backward()
    assert metrics["sf_active"] == 1 and metrics["sf"] > 0
    assert result.feature.requires_grad
    allowed = {id(parameter) for parameter in sf_parameters(model)}
    changed = []
    for (name, parameter), (other, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other
        if not parameter.requires_grad:
            assert parameter.grad is None and expected.grad is None, name
        elif id(parameter) in allowed:
            if parameter.grad is not None and not torch.equal(parameter.grad, expected.grad):
                changed.append(name)
        else:
            assert parameter.grad is not None and expected.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected.grad, atol=0, rtol=0, msg=name)
    assert changed and all(name.startswith("DiT.") for name in changed)
    assert model.pixel_adapter.content_projection.weight.grad.abs().sum() > 0
    assert model.pixel_adapter.precision_projection.weight.grad.abs().sum() > 0


def test_versioned_state_rejects_legacy_or_changed_hybrid_architecture():
    model = hybrid_endpoint()
    state = model.export_state()
    other = hybrid_endpoint()
    other.load_bt_state(state)
    for part, values in state.items():
        for name, value in values.items():
            assert torch.equal(value, other.export_state()[part][name]), (part, name)
    with pytest.raises(ValueError, match="hybrid"):
        other.load_bt_state(precision_model().export_state())
    for name in ("ae_hybrid_version", "hybrid_architecture"):
        invalid = deepcopy(state)
        invalid["pixel_adapter"][name].add_(1)
        with pytest.raises(ValueError, match="incompatible"):
            other.load_bt_state(invalid)
    native = {part: values for part, values in state.items() if part != "pixel_adapter"}
    with pytest.raises(ValueError, match="missing"):
        other.load_bt_state(native)
    other.load_bt_state(native, initialize_pixel_adapter=True)
    with pytest.raises(RuntimeError, match="incompatible"):
        small_adapter().load_state_dict({})


@pytest.mark.parametrize("which", ["content", "precision", "pixels"])
def test_bad_spatial_fields_are_rejected_without_silent_resampling(which):
    model = small_adapter()
    features, content, precision = inputs()
    if which == "content":
        content = content[..., :-1]
    elif which == "precision":
        precision = torch.randn(1, 128, 3, 1)
    else:
        features = features[..., :-1]
    with pytest.raises(ValueError):
        model(features, content, precision)
