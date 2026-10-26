"""An explicit SF coefficient change must preserve continuation state."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from models.bt_self_flow.experiment import load_config, training_contract
from models.bt_self_flow.residual_training import residual_weight
from scripts.set_residual_sf_weight import with_fixed_weight


def active_checkpoint():
    config = load_config("configs/bt_self_flow/c_sf_ablation_b8/c_sf.json")
    dataset = SimpleNamespace(fingerprint="same-images")
    return dict(config=config, contract=training_contract(config, dataset, 2), step=13123,
                micro_seen=13123, attempts=13123, rep_weight=.6413,
                residual_sf=dict(alpha=.6413, protocol="independent_training_crops_mu_gradient_v1"),
                raw={"residual": {"active": torch.tensor(True), "mean": torch.randn(4)}},
                ema={"residual": {"mean": torch.randn(4)}}, teacher={"lora": torch.randn(4)},
                optimizer={"state": {0: {"exp_avg": torch.randn(4), "step": torch.tensor(13123)}}},
                aux_optimizer={"state": {0: {"exp_avg": torch.randn(4)}}},
                scheduler={"last_epoch": 13123}, scaler={"scale": 64.},
                rng_by_rank=[{"torch": torch.get_rng_state()}, {"torch": torch.get_rng_state()}],
                validation_best={"step": 12000, "score": .4683}, full_precision_retry=False)


def test_reweight_preserves_training_state_and_records_the_boundary():
    source = active_checkpoint()
    source_contract = deepcopy(source["contract"])
    changed = with_fixed_weight(source, 1., "/new/joint", {"reason": "user requested fixed weight"})
    for key in ("raw", "ema", "teacher", "optimizer", "aux_optimizer", "scheduler", "scaler",
                "rng_by_rank", "micro_seen", "attempts", "step", "validation_best"):
        assert changed[key] is source[key]
    assert source["rep_weight"] == .6413 and "weight" not in source["config"]["self_flow"]
    assert source["contract"] == source_contract
    assert changed["rep_weight"] == changed["config"]["self_flow"]["weight"] == changed["contract"]["self_flow"]["weight"] == 1.
    assert changed["residual_sf"]["calibrated_alpha"] == .6413
    assert changed["residual_sf"]["weight_history"][-1]["after_step"] == 13123
    expected = deepcopy(source_contract)
    expected["self_flow"]["weight"] = 1.
    assert changed["contract"] == expected


@pytest.mark.parametrize("weight", [0., -1., float("nan"), float("inf"), True])
def test_invalid_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="weight"):
        with_fixed_weight(active_checkpoint(), weight, "/new/joint", {})


def test_incomplete_or_inconsistent_source_is_rejected():
    source = active_checkpoint()
    source["contract"]["train"]["batch_size"] = 8
    with pytest.raises(ValueError, match="contract"):
        with_fixed_weight(source, 1., "/new/joint", {})
    source = active_checkpoint()
    source["teacher"] = {}
    with pytest.raises(ValueError, match="active"):
        with_fixed_weight(source, 1., "/new/joint", {})


def test_fixed_weight_is_explicit_and_calibrated_recipes_stay_compatible():
    original = load_config("configs/bt_self_flow/c_sf_ablation_b8/c_sf.json")
    fixed = load_config("configs/bt_self_flow/c_sf_ablation_b8/c_sf_weight1.json")
    assert residual_weight(original, .6413) == .6413
    assert residual_weight(fixed, .6413) == 1.
    assert training_contract(original, SimpleNamespace(fingerprint="same"), 2) != training_contract(
        fixed, SimpleNamespace(fingerprint="same"), 2)
