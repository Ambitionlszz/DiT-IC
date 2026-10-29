"""Keep batch-8 continuation settings matched and checkpoint preparation inert."""

from copy import deepcopy
from itertools import islice
import json
import random
from unittest.mock import patch

import pytest
import torch
from torch import nn

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, save_atomic
from models.bt_self_flow.data import CounterSampler, counter_seed
from models.bt_self_flow import experiment
from models.bt_self_flow.experiment import load_config
from models.bt_self_flow.residual_training import balanced_quality
from models.bt_self_flow.training_precision import training_amp_dtype
from scripts.launch_quantization_residual import check_effective_batch
from scripts.prepare_c_sf_ablation import CONFIGS, audit_source, paired_configs, prepare


def test_matched_configs_use_original_c_factory_only_in_control():
    configs = paired_configs()

    class FakeModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def set_codec_frozen(self, frozen):
            assert frozen is False

    with patch.object(experiment, "BTCodec", side_effect=FakeModel) as pure, \
         patch("models.bt_self_flow.quantization_model.CQuantizationResidualCodec", side_effect=FakeModel) as residual:
        experiment.make_model(configs["pure_c"], torch.device("cpu"))
        assert pure.call_count == 1 and residual.call_count == 0
        experiment.make_model(configs["c_sf"], torch.device("cpu"))
        assert pure.call_count == residual.call_count == 1


def test_independent_quality_matches_original_c_on_both_ranks_and_branches():
    configs = paired_configs()
    original = load_config(CONFIGS.parent/"precision_v2/c_joint.yaml")
    values = original["quality"]["values"]
    draws = []
    for step in range(10000, 10200):
        expected = random.Random(counter_seed(original["train"]["seed"], step, 13)).choice(values)
        draws.append(expected)
        for config in [original, *configs.values()]:
            for rank in (0, 1):
                control = trainer.sample_control(config, step, rank, (4, 8, 8), "cpu")
                assert control.low == expected and control.high is None
    assert any(a == b for a, b in zip(draws, draws[1:]))
    assert set(draws) == set(values)
    assert any(len(set(draws[start:start+6])) < 6 for start in range(0, 192, 6))


def test_older_residual_configs_keep_balanced_quality_default():
    config = load_config(CONFIGS.parent/"quantization_residual/c_sf_b4a2_stable.json")
    assert "sampling" not in config["quality"]
    for step in range(30):
        assert trainer.sample_control(config, step, 0, (4, 8, 8), "cpu").low == balanced_quality(config, step)


def test_single_rank_global_data_sequence_partitions_unchanged_at_10k():
    original = iter(CounterSampler(123, 903, rank=0, world_size=1, batch_size=8,
                                   micro_batches_seen=10000, stream=23))
    ranks = [iter(CounterSampler(123, 903, rank=rank, world_size=2, batch_size=4,
                                micro_batches_seen=10000, stream=23)) for rank in (0, 1)]
    for _ in range(20):
        assert list(islice(original, 8)) == list(islice(ranks[0], 4))+list(islice(ranks[1], 4))


@pytest.mark.parametrize("capability,requested,expected", [
    ((7, 0), "auto", torch.float16),
    ((7, 0), "fp16", torch.float16),
    ((8, 0), "auto", torch.bfloat16),
    ((8, 0), "fp16", torch.float16),
    ((8, 0), "bf16", torch.bfloat16),
])
def test_amp_precision_is_explicit_even_on_native_bf16_gpu(capability, requested, expected):
    with patch("torch.cuda.get_device_capability", return_value=capability), \
         patch("torch.cuda.is_bf16_supported", return_value=True):
        assert training_amp_dtype(torch.device("cuda:0"), requested) == expected


def test_emulated_bf16_is_rejected_and_cpu_never_queries_cuda():
    with patch("torch.cuda.get_device_capability", return_value=(7, 0)), \
         patch("torch.cuda.is_bf16_supported", return_value=True):
        with pytest.raises(ValueError, match="native BF16"):
            training_amp_dtype(torch.device("cuda:0"), "bf16")
    with patch("torch.cuda.get_device_capability", side_effect=AssertionError("CUDA queried on CPU")):
        assert training_amp_dtype(torch.device("cpu")) == torch.bfloat16


def test_supervisor_accepts_batch8_and_preserves_old_batch16_support():
    config = paired_configs()["c_sf"]
    assert check_effective_batch(config, 2) == 8
    old = load_config(CONFIGS.parent/"quantization_residual/c_sf_b4a2_stable.json")
    assert check_effective_batch(old, 2) == 16
    with pytest.raises(ValueError, match="effective batch"):
        check_effective_batch(config, 1)


def source_checkpoint(tmp_path, **changes):
    config = deepcopy(paired_configs()["pure_c"])
    config["train"].update(batch_size=8, amp_dtype="bf16")
    config["quality"].pop("sampling")
    source = dict(method=METHOD, schema=SCHEMA, phase="joint", backend="C", step=10000,
                  config=config, contract={"world_size": 1}, raw={}, ema={}, optimizer={}, aux_optimizer={},
                  scheduler={}, scaler={}, rng_by_rank=[{}], micro_seen=10000, attempts=10000, assets={})
    source.update(changes)
    path = tmp_path/"pure_c_10k.pt"
    save_atomic(path, source)
    return path


def test_preparation_waits_without_creating_training_outputs(tmp_path, monkeypatch):
    training_root = tmp_path/"training"
    monkeypatch.setenv("C_SF_B8_ROOT", str(training_root))
    output = tmp_path/"prepared"
    plan = prepare(output)
    assert plan["status"] == "waiting_for_checkpoint" and plan["launch_ready"] is False
    assert plan["source"] is None
    assert not training_root.exists()
    for name in ("pure_c", "c_sf"):
        assert json.loads((output/f"{name}.resolved.json").read_text())["train"]["effective_batch"] == 8


def test_source_metadata_audit_does_not_claim_conversion_or_launch_readiness(tmp_path):
    source = source_checkpoint(tmp_path)
    plan = prepare(tmp_path/"prepared", source)
    assert plan["status"] == "awaiting_state_migration"
    assert plan["source"]["metadata_errors"] == []
    assert plan["source"]["converted"] is plan["launch_ready"] is False
    assert plan["source"]["world_size"] == 1 and plan["source"]["effective_batch"] == 8
    assert len(plan["source"]["sha256"]) == 64


@pytest.mark.parametrize("changes,message", [
    ({"step": 9000}, "step 10000"),
    ({"raw": {"residual": {}}}, "original pure C"),
    ({"contract": {"world_size": 2}}, "effective batch 8"),
])
def test_source_audit_flags_wrong_step_model_and_batch(tmp_path, changes, message):
    source = source_checkpoint(tmp_path, **changes)
    report = audit_source(source, paired_configs()["pure_c"])
    assert any(message in error for error in report["metadata_errors"])


@pytest.mark.parametrize("override,message", [
    ({"quality": {"sampling": "typo"}}, "quality.sampling"),
    ({"train": {"amp_dtype": "typo"}}, "train.amp_dtype"),
])
def test_invalid_sampling_and_precision_fail_before_training(tmp_path, override, message):
    path = tmp_path/"config.json"
    path.write_text(json.dumps(dict(base=str(CONFIGS/"pure_c.json"), **override)))
    with pytest.raises(ValueError, match=message):
        load_config(path)
