"""Endpoint ablation isolation, initialization ancestry and non-launching plans."""

from copy import deepcopy
import json
from pathlib import Path
import random
import sys

import pytest
import torch

from models.bt_self_flow.experiment import validate_endpoint_self_flow
from scripts import launch_endpoint_sf as launcher


def common_initialization():
    return dict(phase="backend_init", backend="shared", step=0,
                config=deepcopy(launcher.ablation_configs()["c"]),
                raw={"codec": {}, "distortion_scale": {"calibrated": torch.tensor(True)}},
                calibration={"rate_span_pass": True, "quality_span_pass": True},
                assets={"test": "assets"})


def test_primary_factorial_and_target_controls_change_only_declared_fields():
    configs = launcher.ablation_configs()
    assert tuple(configs) == launcher.PRIMARY_VARIANTS + launcher.CONTROL_VARIANTS
    reference = launcher.shared_signature(configs["c"])
    for name, config in configs.items():
        assert launcher.shared_signature(config) == reference, name
        assert config["self_flow"] == {"enabled": False, "mixed_sampling": False}
        assert "spatial_self_flow" not in config
        assert set(config["endpoint_self_flow"]) == {
            "enabled", "coefficient", "epsilon", "decoder_capture_block", "target"}
        assert config["endpoint_self_flow"]["coefficient"] == 1.
        assert config["train"]["steps"] == 200000
        assert tuple(config["train"][key] for key in (
            "world_size", "batch_size", "accumulation_steps", "effective_batch")) == (2, 4, 1, 8)
    # SF changes training supervision only: the paired deployment architectures
    # retain the exact same C, SANA, AE, D settings and trainable rank settings.
    for baseline, sf in (("c", "c_sf"), ("c_d", "c_d_sf"),
                         ("c_d", "c_d_same_rate_sf"), ("c_d", "c_d_gt_sf")):
        assert configs[baseline]["model"] == configs[sf]["model"]
        assert configs[baseline]["decoder"] == configs[sf]["decoder"]
    assert configs["c_d_sf"]["endpoint_self_flow"]["target"] == "adjacent_ema"
    assert configs["c_d_same_rate_sf"]["endpoint_self_flow"]["target"] == "same_rate_ema"
    assert configs["c_d_gt_sf"]["endpoint_self_flow"]["target"] == "clean_latent"


@pytest.mark.parametrize("change", [
    {"coefficient": .1}, {"coefficient": True}, {"epsilon": 0},
    {"epsilon": float("nan")}, {"target": "random_high"},
    {"cosine": True}, {"projector_enabled": True}, {"student_layer": 8},
    {"bypass_probability": .1}, {"decoder_capture_block": 2},
])
def test_endpoint_validator_rejects_legacy_and_extra_mechanisms(change):
    config = deepcopy(launcher.ablation_configs()["c_d_sf"])
    config["endpoint_self_flow"].update(change)
    with pytest.raises(ValueError):
        validate_endpoint_self_flow(config)


def test_endpoint_validator_rejects_spatial_mixture_and_incorrect_batch_contract():
    config = deepcopy(launcher.ablation_configs()["c_sf"])
    config["spatial_self_flow"] = {"enabled": False}
    with pytest.raises(ValueError):
        validate_endpoint_self_flow(config)
    config.pop("spatial_self_flow")
    config["train"].update(world_size=1, batch_size=8)
    with pytest.raises(ValueError):
        validate_endpoint_self_flow(config)


def test_student_quality_sequence_matches_variants_and_does_not_consume_global_rng():
    from train_bt_self_flow import sample_control
    configs = launcher.ablation_configs(seed=904)
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    reference = {}
    for rank in (0, 1):
        reference[rank] = [sample_control(configs["c"], step, rank, (4, 16, 16),
                                          torch.device("cpu")).value for step in range(64)]
        for config in configs.values():
            actual = [sample_control(config, step, rank, (4, 16, 16),
                                     torch.device("cpu")).value for step in range(64)]
            assert actual == reference[rank]
    assert len(set(reference[0])) == 6
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_initialization_rejects_trained_d_sf_and_uncalibrated_ancestors():
    initial = common_initialization()
    launcher.validate_initialization(initial)
    for part in ("pixel_adapter", "spatial_projector", "projector"):
        checkpoint = deepcopy(initial)
        checkpoint["raw"][part] = {}
        with pytest.raises(ValueError, match="pure-C"):
            launcher.validate_initialization(checkpoint)
    for field in ("endpoint_self_flow", "self_flow", "spatial_self_flow"):
        checkpoint = deepcopy(initial)
        checkpoint["config"].setdefault(field, {})["enabled"] = True
        with pytest.raises(ValueError, match="pure-C"):
            launcher.validate_initialization(checkpoint)
    for field in ("rate_span_pass", "quality_span_pass"):
        checkpoint = deepcopy(initial)
        checkpoint["calibration"][field] = False
        with pytest.raises(ValueError, match="calibration"):
            launcher.validate_initialization(checkpoint)
    initial["raw"]["distortion_scale"]["calibrated"] = torch.tensor(False)
    with pytest.raises(ValueError, match="calibration"):
        launcher.validate_initialization(initial)


def test_shared_identity_pins_common_initialization_and_source_without_dry_run_writes(tmp_path):
    path = tmp_path / "family" / "ablation_identity.json"
    identity = {"initialization_sha256": "common", "source_sha256": "code", "shared_config_sha256": "cfg"}
    launcher.check_shared_identity(path, identity)
    assert not path.exists()
    launcher.check_shared_identity(path, identity, write=True)
    launcher.check_shared_identity(path, identity)
    for field in identity:
        changed = dict(identity, **{field: "different"})
        with pytest.raises(ValueError, match="different common initialization"):
            launcher.check_shared_identity(path, changed)


def test_frozen_source_detects_post_snapshot_modification(tmp_path):
    source = tmp_path / "live"
    source.mkdir()
    (source / "train.py").write_text("value = 1\n")
    identity = {"sha256": "test", "files": {"train.py": launcher.sha256_file(source / "train.py")}}
    frozen = tmp_path / "frozen"
    launcher.freeze_source(identity, frozen, source)
    launcher.freeze_source(identity, frozen, source)
    (frozen / "train.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="frozen source was modified"):
        launcher.freeze_source(identity, frozen, source)


def test_launcher_default_plan_never_probes_cuda_starts_processes_or_writes_runs(monkeypatch, tmp_path, capsys):
    checkpoint = common_initialization()
    monkeypatch.setattr(launcher, "read_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(launcher, "check_stage_model", lambda *args: None)
    monkeypatch.setattr(launcher, "asset_identity", lambda config: checkpoint["assets"])
    monkeypatch.setattr(launcher, "sha256_file", lambda path: "common")
    monkeypatch.setattr(launcher, "source_identity", lambda: {"sha256": "code", "files": {}})
    def forbidden(*args, **kwargs):
        raise AssertionError("default dry-run attempted a process or filesystem mutation")
    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher.subprocess, "call", forbidden)
    monkeypatch.setattr(launcher, "freeze_source", forbidden)
    run_root = tmp_path / "new_runs"
    monkeypatch.setattr(sys, "argv", ["launch_endpoint_sf.py", "--variant", "c_d_sf",
                                      "--init-checkpoint", str(tmp_path / "backend_init.pt"),
                                      "--run-root", str(run_root), "--gpus", "2,3"])
    launcher.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] and plan["devices"] == ["2", "3"]
    assert plan["effective_batch"] == 8 and "--nproc-per-node=2" in plan["command"]
    assert not run_root.exists()


@pytest.mark.parametrize("gpus", ["0", "0,0", "0,1,2", "0,"])
def test_launcher_requires_exactly_two_distinct_devices(monkeypatch, gpus):
    monkeypatch.setattr(sys, "argv", ["launch_endpoint_sf.py", "--variant", "c",
                                      "--init-checkpoint", "unused.pt", "--gpus", gpus])
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2
