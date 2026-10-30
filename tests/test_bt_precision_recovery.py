from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.training_precision import can_retry_fp32, clip_grad_norm_safe_
from test_bt_training import make_tiny_stage, small_config


def test_bf16_retry_does_not_require_a_grad_scaler():
    assert can_retry_fp32(torch.device("cuda"), 1, False, False)
    assert can_retry_fp32(torch.device("cuda"), 1, False, True)
    assert not can_retry_fp32(torch.device("cuda"), 1, True, False)
    assert not can_retry_fp32(torch.device("cuda"), 2, False, False)
    assert not can_retry_fp32(torch.device("cpu"), 1, False, False)


@pytest.mark.parametrize("values", [[[3e20, 4e20]], [[1e19], [1e19], [1e19], [1e19]]])
def test_finite_gradient_norm_overflow_retains_correct_clipping(values):
    parameters = [torch.nn.Parameter(torch.zeros(len(value))) for value in values]
    original = [torch.tensor(value) for value in values]
    for parameter, gradient in zip(parameters, original):
        parameter.grad = gradient.clone()
    reference_norm = torch.linalg.vector_norm(torch.cat(original).double())
    with pytest.raises(RuntimeError, match="non-finite"):
        torch.nn.utils.clip_grad_norm_(parameters, 4., error_if_nonfinite=True)
    norm = clip_grad_norm_safe_(parameters, 4.)
    assert norm.dtype == torch.float64
    torch.testing.assert_close(norm, reference_norm)
    for parameter, gradient in zip(parameters, original):
        torch.testing.assert_close(parameter.grad, (gradient.double()*4./reference_norm).float())
    torch.testing.assert_close(torch.linalg.vector_norm(torch.cat([p.grad for p in parameters])), torch.tensor(4.))


def test_regular_gradient_clipping_is_unchanged():
    original = torch.nn.Parameter(torch.zeros(2))
    actual = torch.nn.Parameter(torch.zeros(2))
    original.grad = torch.tensor([6., 8.])
    actual.grad = original.grad.clone()
    reference_norm = torch.nn.utils.clip_grad_norm_([original], 4., error_if_nonfinite=True)
    norm = clip_grad_norm_safe_([actual], 4.)
    assert torch.equal(norm, reference_norm)
    assert torch.equal(actual.grad, original.grad)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_gradient_still_requires_recovery(value):
    parameter = torch.nn.Parameter(torch.zeros(1))
    parameter.grad = torch.tensor([value])
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_grad_norm_safe_([parameter], 4.)
    assert not torch.isfinite(clip_grad_norm_safe_([parameter], 4., error_if_nonfinite=False))


def prepare_images(root):
    (root/"images").mkdir()
    for index in range(3):
        pixels = np.random.default_rng(903+index).integers(0, 256, (270, 270, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root/f"images/{index}.png")


def assert_same_weights(reference, actual):
    assert actual["step"] == reference["step"]
    assert actual["micro_seen"] == reference["micro_seen"]
    assert actual["scheduler"] == reference["scheduler"]
    for weights in ("raw", "ema"):
        for part, values in reference[weights].items():
            for name, value in values.items():
                assert torch.equal(value, actual[weights][part][name]), f"{weights}.{part}.{name}"


@pytest.mark.parametrize("fault", ["nan_gradient", "finite_norm_overflow"])
def test_numerical_failure_replays_exact_batches_and_weights(tmp_path, fault):
    prepare_images(tmp_path)
    args = SimpleNamespace(resume=None, init_checkpoint=None, max_updates=None)
    with patch.object(trainer, "make_model", side_effect=make_tiny_stage), \
         patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}):
        trainer.train(small_config(tmp_path), args, 0, 1, torch.device("cpu"))
    reference = read_checkpoint(tmp_path/"codec/complete.pt")
    config = small_config(tmp_path)
    config["train"]["output"] = str(tmp_path/"replayed")
    calls = [0]

    def factory(cfg, device):
        model = make_tiny_stage(cfg, device)
        parameter = next(p for p in model.codec.parameters() if p.requires_grad)

        def corrupt_once(gradient):
            calls[0] += 1
            return gradient*float("nan") if fault == "nan_gradient" and calls[0] == 4 else gradient

        parameter.register_hook(corrupt_once)
        return model

    # Exercise the CUDA BF16 policy on CPU while GradScaler stays disabled.
    def bf16_policy(device, world, already_retrying, scaler_enabled):
        assert not scaler_enabled
        return can_retry_fp32(torch.device("cuda"), world, already_retrying, scaler_enabled)

    clip_calls = [0]

    def legacy_clip(*args, **kwargs):
        clip_calls[0] += 1
        if fault == "finite_norm_overflow" and clip_calls[0] == 2:
            raise RuntimeError("The total norm is non-finite")
        return clip_grad_norm_safe_(*args, **kwargs)

    with patch.object(trainer, "make_model", side_effect=factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}), \
         patch.object(trainer, "can_retry_fp32", side_effect=bf16_policy), \
         patch.object(trainer, "clip_grad_norm_safe_", side_effect=legacy_clip):
        trainer.train(config, args, 0, 1, torch.device("cpu"))
    actual = read_checkpoint(tmp_path/"replayed/complete.pt")
    assert actual["attempts"] == reference["attempts"]+1
    assert actual["full_precision_retry"] is False
    assert_same_weights(reference, actual)
    paused = read_checkpoint(tmp_path/"replayed/paused.pt")
    assert paused["step"] == 1
    assert paused["micro_seen"] == 2
    assert paused["full_precision_retry"] is True
    # The saved retry must reproduce the same result after a process restart.
    args.resume = str(tmp_path/"replayed/paused.pt")
    with patch.object(trainer, "make_model", side_effect=make_tiny_stage), \
         patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}), \
         patch.object(trainer, "full_precision_recompute", wraps=trainer.full_precision_recompute) as recompute:
        trainer.train(config, args, 0, 1, torch.device("cpu"))
    assert recompute.call_args_list[0].args[1] is (fault == "nan_gradient")
    resumed = read_checkpoint(tmp_path/"replayed/complete.pt")
    assert resumed["full_precision_retry"] is False
    assert_same_weights(reference, resumed)


def test_persistent_bad_gradient_stops_before_optimizer_or_ema_update(tmp_path):
    prepare_images(tmp_path)
    config = small_config(tmp_path)
    calls = [0]

    def factory(cfg, device):
        model = make_tiny_stage(cfg, device)
        parameter = next(p for p in model.codec.parameters() if p.requires_grad)

        def corrupt(gradient):
            calls[0] += 1
            return gradient*float("inf")

        parameter.register_hook(corrupt)
        return model

    def bf16_policy(device, world, already_retrying, scaler_enabled):
        return can_retry_fp32(torch.device("cuda"), world, already_retrying, scaler_enabled)

    args = SimpleNamespace(resume=None, init_checkpoint=None, max_updates=None)
    with patch.object(trainer, "make_model", side_effect=factory), \
         patch.object(trainer, "asset_identity", return_value={"test": "tiny-frozen-AE"}), \
         patch.object(trainer, "can_retry_fp32", side_effect=bf16_policy):
        with pytest.raises(FloatingPointError, match="last completed update"):
            trainer.train(config, args, 0, 1, torch.device("cpu"))
    paused = read_checkpoint(tmp_path/"codec/paused.pt")
    assert paused["step"] == 0
    assert paused["micro_seen"] == 0
    assert paused["attempts"] == 2
    assert not paused["optimizer"]["state"]
    assert paused["scheduler"]["last_epoch"] == 0
    assert calls[0] == 4
    assert len(list((tmp_path/"codec").glob("nonfinite_*.json"))) == 2
    for values in paused["raw"].values():
        assert all(torch.isfinite(value).all() for value in values.values() if value.is_floating_point())
