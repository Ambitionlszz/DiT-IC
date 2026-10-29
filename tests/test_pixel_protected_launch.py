import json
from pathlib import Path

import pytest

from scripts.launch_pixel_protected import inspect_reports, same_process, verify_failed_resident, verify_holder


def fake_process(root, pid, parent, command, gpu=None):
    directory = root / str(pid)
    directory.mkdir()
    fields = ["S", str(parent)] + ["0"] * 17 + ["12345"]
    (directory / "stat").write_text(f"{pid} (a process with spaces) " + " ".join(fields))
    (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in command) + b"\0")
    (directory / "environ").write_bytes(f"CUDA_VISIBLE_DEVICES={gpu}\0".encode() if gpu else b"")


def test_only_exact_authorized_holder_on_multislot_controller(tmp_path):
    fake_process(tmp_path, 10, 1, ["python", "-c", "import watch_two_idle_gpus", "--slots", "2"])
    fake_process(tmp_path, 11, 10, ["python", "/repo/scripts/train_depart.py", "--gb", "28"], "GPU-5")
    identity = verify_holder(11, "GPU-5", tmp_path)
    assert identity["pid"] == 11 and identity["parent"] == 10 and identity["start"] == "12345"
    with pytest.raises(ValueError, match="different physical GPU"):
        verify_holder(11, "GPU-1", tmp_path)
    fake_process(tmp_path, 12, 10, ["python", "/repo/train_bt_self_flow.py"], "GPU-5")
    with pytest.raises(ValueError, match="not a live memory holder"):
        verify_holder(12, "GPU-5", tmp_path)


def test_adapter_only_and_small_batch_cannot_open_training_gate(tmp_path):
    report = tmp_path / "report.json"
    passed = dict(status="passed", batch_size=8, steps=[{}, {}],
                  scope="pretrained_sana_dc_ae_codec_perceptual_rd")
    report.write_text(json.dumps(passed))
    assert inspect_reports([report])[0] == passed
    for changes in (dict(batch_size=1), dict(status="failed"), dict(steps=[{}]),
                    dict(scope="production_width_pixel_adapter_only")):
        report.write_text(json.dumps({**passed, **changes}))
        with pytest.raises(ValueError, match="complete batch8"):
            inspect_reports([report])


def test_process_identity_tolerates_scheduling_but_rejects_pid_reuse():
    identity = dict(pid=11, parent=10, start="12345", command=["python", "train_depart.py"], state="R")
    assert same_process({**identity, "state": "S"}, identity)
    for changes in (dict(start="99999"), dict(parent=1), dict(command=["python", "train.py"]), dict(pid=12)):
        assert not same_process({**identity, **changes}, identity)


def test_recovery_rejects_running_training_and_unrelated_resident(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    fake_process(tmp_path, 20, 1, ["python", "/repo/scripts/launch_pixel_protected.py", "--root", str(previous)], "GPU-5")
    record = dict(pid=20, gpu_uuid="GPU-5", phase="failed_reserved")
    (previous / "status.json").write_text(json.dumps(record))
    assert verify_failed_resident(20, "GPU-5", previous, tmp_path)["pid"] == 20
    (previous / "status.json").write_text(json.dumps({**record, "phase": "training"}))
    with pytest.raises(ValueError, match="failed protected run"):
        verify_failed_resident(20, "GPU-5", previous, tmp_path)


def test_launcher_has_no_empty_cache_or_controller_shutdown():
    source = (Path(__file__).resolve().parents[1] / "scripts/launch_pixel_protected.py").read_text()
    assert "torch.cuda.empty_cache(" not in source
    assert 'os.kill(holder["pid"], signal.SIGTERM)' in source
    assert 'os.kill(holder["parent"]' not in source
    assert '"--guard"' in source and '"failed_reserved"' in source
