from copy import deepcopy
import json

import pytest

from scripts.resume_pixel_protected import (REPO, parent_is_alive, repartition_microbatches, resume_configuration,
                                           stage_resume_checkpoints, verify_reservation)


def joint_checkpoint():
    return dict(phase="joint", backend="C", step=8000, contract=dict(world_size=1),
                raw=dict(pixel_adapter={}), ema=dict(pixel_adapter={}),
                config=dict(decoder=dict(method="sana_conditioned_pixel_adapter_v1", pixel_adapter=dict(depth=2)),
                            self_flow=dict(enabled=False), train=dict(steps=200000, batch_size=8,
                            accumulation_steps=1, effective_batch=8, output="previous")))


def test_resume_preserves_settings_and_source_checkpoint(tmp_path):
    source = joint_checkpoint()
    before = deepcopy(source)
    result = resume_configuration(source, tmp_path)
    assert source == before
    result["train"]["output"] = "previous"
    assert result == source["config"]


@pytest.mark.parametrize("change", ("phase", "backend", "pixel", "self_flow", "batch", "world", "completed"))
def test_resume_rejects_incompatible_experiments(tmp_path, change):
    source = joint_checkpoint()
    if change == "phase":
        source["phase"] = "codec"
    elif change == "backend":
        source["backend"] = "D"
    elif change == "pixel":
        source["ema"].pop("pixel_adapter")
    elif change == "self_flow":
        source["config"]["self_flow"]["enabled"] = True
    elif change == "batch":
        source["config"]["train"]["batch_size"] = 3
    elif change == "world":
        source["contract"]["world_size"] = 2
    else:
        source["step"] = 200000
    with pytest.raises(ValueError, match="resumable"):
        resume_configuration(source, tmp_path)


def test_reservation_matches_exact_tool_status_and_gpu(tmp_path):
    status = tmp_path / "reservation.json"
    command = ["python", str(REPO / "scripts/reserve_gpu_memory.py"), "--status", str(status)]
    process = tmp_path / "12"
    process.mkdir()
    fields = ["S", "1"] + ["0"] * 17 + ["12345"]
    (process / "stat").write_text("12 (reservation) " + " ".join(fields))
    (process / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in command) + b"\0")
    (process / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=GPU-4\0")
    status.write_text(json.dumps(dict(pid=12, ready=True, parent_pid=1, gpu="GPU-4")))
    assert verify_reservation(status, "GPU-4", tmp_path)["pid"] == 12
    with pytest.raises(ValueError, match="unrelated"):
        verify_reservation(status, "GPU-5", tmp_path)
    (tmp_path / "12/cmdline").write_bytes(b"python\0train_bt_self_flow.py\0")
    with pytest.raises(ValueError, match="unrelated"):
        verify_reservation(status, "GPU-4", tmp_path)


def test_migration_restores_best_checkpoint_before_trainer_initializes(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    source = previous / "0008000.pt"
    source.write_bytes(b"latest recovery")
    best = previous / "0006000.pt"
    best.write_bytes(b"previous best")
    output = tmp_path / "new"
    checkpoint = dict(validation_best=dict(checkpoint=best.name))
    stage_resume_checkpoints(checkpoint, source, output)
    assert (output / source.name).read_bytes() == source.read_bytes()
    assert (output / best.name).read_bytes() == best.read_bytes()
    stage_resume_checkpoints(checkpoint, source, output)
    with pytest.raises(ValueError, match="invalid"):
        stage_resume_checkpoints(dict(validation_best=dict(checkpoint="../bad.pt")), source, output)


def test_four_by_two_preserves_weights_optimizer_rng_and_next_image_position(tmp_path):
    source = joint_checkpoint()
    source["micro_seen"] = 8000
    source["contract"]["train"] = deepcopy(source["config"]["train"])
    source.update(optimizer=object(), aux_optimizer=object(), scheduler=object(), rng_by_rank=[object()])
    migrated = repartition_microbatches(source, 4)
    result = resume_configuration(migrated, tmp_path)
    assert result["train"]["batch_size"] == 4 and result["train"]["accumulation_steps"] == 2
    assert migrated["micro_seen"] == 16000 and source["micro_seen"] == 8000
    assert migrated["step"] == source["step"] == 8000
    for part in ("raw", "ema", "optimizer", "aux_optimizer", "scheduler", "rng_by_rank"):
        assert migrated[part] is source[part]
    assert source["config"]["train"]["batch_size"] == source["contract"]["train"]["batch_size"] == 8
    assert migrated["contract"]["train"]["batch_size"] == 4


def test_cpu_guard_detects_pid_reuse_zombies_and_exit(tmp_path):
    process = tmp_path / "12"
    process.mkdir()
    fields = ["S", "1"] + ["0"] * 17 + ["12345"]
    (process / "stat").write_text("12 (training) " + " ".join(fields))
    (process / "cmdline").write_bytes(b"python\0training.py\0")
    assert parent_is_alive(12, "12345", tmp_path)
    assert not parent_is_alive(12, "other-start", tmp_path)
    fields[0] = "Z"
    (process / "stat").write_text("12 (training) " + " ".join(fields))
    assert not parent_is_alive(12, "12345", tmp_path)
    assert not parent_is_alive(99, "12345", tmp_path)
