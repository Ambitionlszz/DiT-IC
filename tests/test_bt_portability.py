from copy import deepcopy
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from bt_runtime import configure_caches, relocate_config


def test_relocation_preserves_recipe_and_custom_data(tmp_path, monkeypatch):
    config = {"data": {"train": ["/data2/ls/data/Flickr2K/Flickr2K_png", "/custom/images"],
                       "valid": ["/data2/ls/data/Kodak"], "crop": 512},
              "model": {"sana_path": "/data2/ls/checkpoints/SANA", "text_path": "/old/text.pt",
                        "codec": {"delta_coarse": 8., "delta_fine": 1.}},
              "train": {"batch_size": 32, "accumulation_steps": 1, "effective_batch": 64,
                        "rate_weight": 3.}}
    original = deepcopy(config)
    monkeypatch.setenv("BT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BT_SANA_PATH", str(tmp_path / "SANA"))
    monkeypatch.setenv("BT_TEXT_PATH", str(tmp_path / "text.pt"))
    relocated = relocate_config(config)
    assert config == original
    assert relocated["data"]["train"] == [str(tmp_path / "data/Flickr2K/Flickr2K_png"), "/custom/images"]
    assert relocated["data"]["valid"] == [str(tmp_path / "data/Kodak")]
    assert relocated["model"]["sana_path"] == str(tmp_path / "SANA")
    assert relocated["model"]["text_path"] == str(tmp_path / "text.pt")
    assert relocated["train"] == config["train"]
    assert relocated["model"]["codec"] == config["model"]["codec"]
    assert relocate_config(relocated) == relocated


def test_no_overrides_preserves_existing_v100_config(monkeypatch):
    for name in ("BT_DATA_ROOT", "BT_SANA_PATH", "BT_TEXT_PATH"):
        monkeypatch.delenv(name, raising=False)
    config = {"data": {"train": ["/data2/ls/data/LSDIR"]}, "model": {"sana_path": "/old/SANA"}}
    assert relocate_config(config) == config


def test_inherited_lowrate_configs_resolve_before_manifest(tmp_path, monkeypatch):
    from scripts.launch_bt_c_ablation import resolved_configs
    monkeypatch.setenv("BT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BT_SANA_PATH", str(tmp_path / "SANA"))
    monkeypatch.setenv("BT_TEXT_PATH", str(tmp_path / "text.pt"))
    configs = resolved_configs(tmp_path / "run")
    for config in configs.values():
        assert config["data"]["valid"] == [str(tmp_path / "data/Kodak")]
        assert all(path.startswith(str(tmp_path / "data")) for path in config["data"]["train"])
        assert config["model"]["sana_path"] == str(tmp_path / "SANA")
        assert config["train"]["rate_weight"] == 3.


def test_nested_data_root_is_relocated_once(monkeypatch):
    from models.bt_self_flow.experiment import load_config
    monkeypatch.setenv("BT_DATA_ROOT", "/data2/ls/data/portable")
    repo = Path(__file__).resolve().parents[1]
    config = load_config(repo / "configs/bt_self_flow/lowrate_c_no_latent.yaml")
    assert config["data"]["valid"] == ["/data2/ls/data/portable/Kodak"]


def test_gpu_check_rejects_v100_wheel_on_5090_before_allocating():
    from scripts.check_bt_gpu import check_gpu
    cuda = SimpleNamespace(get_device_capability=lambda device: (12, 0),
                           get_device_name=lambda device: "NVIDIA GeForce RTX 5090",
                           get_arch_list=lambda: ["sm_70", "sm_80", "sm_90"])
    torch = SimpleNamespace(device=lambda value: value, cuda=cuda)
    with pytest.raises(RuntimeError, match="needs sm_120"):
        check_gpu(torch, 0, "5090")


def test_cache_overrides_preserve_explicit_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BT_CACHE_ROOT", str(tmp_path))
    env = {"TORCH_HOME": "/existing/torch"}
    configure_caches(env)
    assert env == {"TORCH_HOME": "/existing/torch", "HF_HOME": str(tmp_path / "huggingface")}


def test_shell_uses_selected_interpreter_and_run_root(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    fake_python = tmp_path / "python"
    fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$TORCH_HOME" "$HF_HOME" "$@"\n')
    fake_python.chmod(0o755)
    env = dict(os.environ, BT_PYTHON=str(fake_python), BT_RUN_ROOT=str(tmp_path / "run"),
               BT_CACHE_ROOT=str(tmp_path / "cache"))
    env.pop("TORCH_HOME", None)
    env.pop("HF_HOME", None)
    result = subprocess.run(["bash", str(repo / "scripts/run_bt_self_flow.sh"), "C", "0"],
                            env=env, text=True, capture_output=True, check=True)
    assert str(tmp_path / "run/backend_init.pt") in result.stdout
    assert str(tmp_path / "cache/torch") in result.stdout
    assert str(tmp_path / "cache/huggingface") in result.stdout
    assert "/data2/ls" not in result.stdout
