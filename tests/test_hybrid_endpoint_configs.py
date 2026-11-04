"""Additive AE-conditioned hybrid family routing and launch isolation."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from models.bt_self_flow.experiment import (AE_HYBRID_PIXEL_METHOD, make_model,
                                             validate_decoder)
from scripts import launch_endpoint_sf as launcher


def test_hybrid_recipes_preserve_original_c_sf_optimizer_and_sample_contract():
    legacy = launcher.ablation_configs()
    hybrid = launcher.ablation_configs(family="hybrid")
    assert tuple(hybrid) == launcher.PRIMARY_VARIANTS + launcher.CONTROL_VARIANTS
    for name, config in hybrid.items():
        assert launcher.shared_signature(config) == launcher.shared_signature(legacy[name])
        d = launcher.VARIANTS[name][0]
        assert config["decoder"]["method"] == (AE_HYBRID_PIXEL_METHOD if d else "native")
        assert "/hybrid/" in config["train"]["output"]
    for baseline, sf in (("c", "c_sf"), ("c_d", "c_d_sf"),
                         ("c_d", "c_d_same_rate_sf"), ("c_d", "c_d_gt_sf")):
        assert hybrid[baseline]["decoder"] == hybrid[sf]["decoder"]


@pytest.mark.parametrize("change", [{"depth": 4}, {"patch_size": 8},
                                     {"context_dim": 128}, {"delta_ref": 1.},
                                     {"use_quantization": True}])
def test_hybrid_validator_rejects_undeclared_architecture_or_legacy_options(change):
    config = deepcopy(launcher.ablation_configs(family="hybrid")["c_d"])
    config["decoder"]["pixel_adapter"].update(change)
    with pytest.raises(ValueError):
        validate_decoder(config)


def test_hybrid_requires_endpoint_decode_family_even_with_sf_off():
    config = deepcopy(launcher.ablation_configs(family="hybrid")["c_d"])
    del config["endpoint_self_flow"]
    with pytest.raises(ValueError, match="endpoint decoder family"):
        validate_decoder(config)


def test_factory_selects_hybrid_and_preserves_sf_off_capture_owner(monkeypatch):
    from models.bt_self_flow import endpoint_model
    seen = {}

    class TinyCodec(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            seen.update(kwargs)

        def set_codec_frozen(self, frozen):
            seen["codec_frozen"] = frozen

    monkeypatch.setattr(endpoint_model, "EndpointAblationCodec", TinyCodec)
    config = launcher.ablation_configs(family="hybrid")["c_d"]
    make_model(config, torch.device("cpu"))
    assert seen["pixel_method"] == AE_HYBRID_PIXEL_METHOD
    assert seen["pixel_config"] == config["decoder"]["pixel_adapter"]
    assert not seen["endpoint_config"]["enabled"]
    assert seen["codec_frozen"] is False


def _stub_plan(monkeypatch, checkpoint):
    monkeypatch.setattr(launcher, "read_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(launcher, "check_stage_model", lambda *args: None)
    monkeypatch.setattr(launcher, "asset_identity", lambda config: {"test": "assets"})
    monkeypatch.setattr(launcher, "sha256_file", lambda path: "common")
    monkeypatch.setattr(launcher, "source_identity", lambda: {"sha256": "code", "files": {}})

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted a launch, CUDA probe or source write")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher.subprocess, "call", forbidden)
    monkeypatch.setattr(launcher, "freeze_source", forbidden)


def test_hybrid_plan_uses_separate_namespace_and_identity_without_writes(monkeypatch, tmp_path, capsys):
    checkpoint = dict(phase="backend_init", backend="shared", step=0,
                      config=launcher.ablation_configs()["c"], assets={"test": "assets"},
                      raw={"codec": {}, "distortion_scale": {"calibrated": torch.tensor(True)}},
                      calibration={"rate_span_pass": True, "quality_span_pass": True})
    _stub_plan(monkeypatch, checkpoint)
    identities = []
    monkeypatch.setattr(launcher, "check_shared_identity",
                        lambda path, expected: identities.append((Path(path), expected)))
    root = tmp_path / "runs"
    monkeypatch.setattr(sys, "argv", ["launch_endpoint_sf.py", "--family", "hybrid",
                                      "--variant", "c_d_sf", "--init-checkpoint", "init.pt",
                                      "--run-root", str(root)])
    launcher.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["family"] == "hybrid" and plan["dry_run"]
    assert Path(plan["output"]) == root / "hybrid" / "seed_903" / "runs" / "c_d_sf"
    assert identities[0][0] == root / "hybrid" / "ablation_identity.json"
    assert identities[0][1]["decoder_family"] == "hybrid"
    assert not root.exists()


def test_hybrid_launcher_rejects_resume_from_old_v4_decoder(monkeypatch, tmp_path):
    checkpoint = dict(phase="joint", config=launcher.ablation_configs()["c_d_sf"],
                      source_sha256="same_common_initializer")
    _stub_plan(monkeypatch, checkpoint)
    monkeypatch.setattr(sys, "argv", ["launch_endpoint_sf.py", "--family", "hybrid",
                                      "--variant", "c_d_sf", "--resume", "old_v4.pt",
                                      "--run-root", str(tmp_path / "runs")])
    with pytest.raises(ValueError, match="preserve the endpoint decoder"):
        launcher.main()
