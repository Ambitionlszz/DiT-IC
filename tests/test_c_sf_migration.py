"""Preserve learned C state while changing rank layout and adding a dormant head."""

from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.bt_self_flow.checkpoint import capture_rng, clone_state
from scripts.migrate_c_sf_batch8 import check_dataset, migrate_progress, migrate_state, new_rank_rng, parameter_layout, remap_optimizer
from models.bt_self_flow.data import CounterSampler
from scripts.prepare_c_sf_ablation import paired_configs
from scripts.launch_c_sf_ablation import next_segment
from scripts.launch_quantization_residual import ResidualRun


class TinyC(nn.Module):
    def __init__(self, residual=False):
        super().__init__()
        if residual:
            # Register the new group first to test actual name mapping rather
            # than accidentally relying on unchanged serialized integer IDs.
            self.residual_head = nn.Linear(3, 3)
            self.residual_head.register_buffer("active", torch.tensor(False))
        self.codec = nn.Linear(3, 3)
        self.codec.register_parameter("quantiles", nn.Parameter(torch.zeros(3)))
        self.DiT = nn.Linear(3, 3)

    def export_state(self):
        state = dict(codec=self.codec.state_dict(), dit=self.DiT.state_dict())
        if hasattr(self, "residual_head"):
            state["residual"] = self.residual_head.state_dict()
        return state

    def load_bt_state(self, state):
        self.codec.load_state_dict(state["codec"])
        self.DiT.load_state_dict(state["dit"])
        if hasattr(self, "residual_head"):
            self.residual_head.load_state_dict(state["residual"])


def source_state():
    torch.manual_seed(7)
    config = paired_configs()["pure_c"]
    config["train"].update(batch_size=2, accumulation_steps=2)
    model = TinyC()
    groups, aux, layout, aux_layout = parameter_layout(model, config)
    optimizer = torch.optim.AdamW(groups, betas=(.9, .999), weight_decay=0)
    auxiliary = torch.optim.AdamW(aux, lr=.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.)
    for _ in range(2):
        optimizer.zero_grad()
        auxiliary.zero_grad()
        model.DiT(model.codec(torch.ones(2, 3))).square().mean().backward()
        model.codec.quantiles.sum().backward()
        optimizer.step()
        auxiliary.step()
        scheduler.step()
    raw = clone_state(model.export_state(), cpu=True)
    ema = clone_state(raw)
    for state in ema.values():
        for value in state.values():
            if value.is_floating_point():
                value.add_(.01)
    source = dict(raw=raw, ema=ema, optimizer=optimizer.state_dict(), aux_optimizer=auxiliary.state_dict(),
                  scheduler=scheduler.state_dict(), scaler={}, config=config, step=10000, micro_seen=20000,
                  contract=dict(world_size=2), attempts=10000, rng_by_rank=[capture_rng(), new_rank_rng(904)],
                  source_sha256="warmup", validation_best={}, assets={})
    return source, layout, aux_layout


@pytest.mark.parametrize("name", ["pure_c", "c_sf"])
def test_migration_preserves_shared_weights_and_named_moments(name):
    source, layout, aux_layout = source_state()
    config = paired_configs()[name]
    model = TinyC(residual=name == "c_sf")
    migrated = migrate_state(source, layout, aux_layout, model, config, SimpleNamespace(fingerprint="same"), {}, "10k")
    assert migrated["step"] == migrated["micro_seen"] == 10000
    assert migrated["attempts"] == 10000 and migrated["validation_best"] is None
    assert migrated["contract"]["world_size"] == 2
    assert len(migrated["rng_by_rank"]) == 2
    for rank in (0, 1):
        assert torch.equal(migrated["rng_by_rank"][rank]["torch"], source["rng_by_rank"][rank]["torch"])
    assert migrated["migration"]["preserved_global_samples"] == 80000
    assert migrated["migration"]["source_micro_seen"] == 20000
    assert migrated["scaler"]["scale"] == 64.
    for kind in ("raw", "ema"):
        for component, values in source[kind].items():
            for key, value in values.items():
                assert torch.equal(migrated[kind][component][key], value)
    _, _, new_layout, _ = parameter_layout(model, config)
    expected = {parameter: source["optimizer"]["state"][identity]
                for group, (_, parameters) in zip(source["optimizer"]["param_groups"], layout)
                for identity, (parameter, _) in zip(group["params"], parameters)}
    for group, (label, parameters) in zip(migrated["optimizer"]["param_groups"], new_layout):
        for identity, (parameter, _) in zip(group["params"], parameters):
            actual = migrated["optimizer"]["state"].get(identity)
            if label == "residual":
                assert actual is None
            else:
                assert all(torch.equal(actual[key], value) for key, value in expected[parameter].items())
    # Loading a real optimizer/scheduler exercises the migrated group IDs.
    groups, aux, _, _ = parameter_layout(model, config)
    optimizer = torch.optim.AdamW(groups)
    optimizer.load_state_dict(migrated["optimizer"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.)
    scheduler.load_state_dict(migrated["scheduler"])
    assert len(scheduler.base_lrs) == len(groups)
    auxiliary = torch.optim.AdamW(aux)
    auxiliary.load_state_dict(migrated["aux_optimizer"])
    assert auxiliary.state[model.codec.quantiles]["step"] == 2
    if name == "c_sf":
        assert not migrated["raw"]["residual"]["active"]
        assert migrated["rep_weight"] == 0 and "teacher" not in migrated


def test_both_branches_get_identical_scaler_and_rank_rng_policy():
    source, layout, aux_layout = source_state()
    states = [migrate_state(source, layout, aux_layout, TinyC(residual=name == "c_sf"), config,
                            SimpleNamespace(fingerprint="same"), {}, "10k")
              for name, config in paired_configs().items()]
    assert states[0]["scaler"] == states[1]["scaler"]
    for rank in (0, 1):
        assert torch.equal(states[0]["rng_by_rank"][rank]["torch"], states[1]["rng_by_rank"][rank]["torch"])


@pytest.mark.parametrize("world,batch,accumulation,micro", [(2, 2, 2, 20000), (1, 8, 1, 10000)])
def test_counter_migration_preserves_the_next_global_sample_and_crop_stream(world, batch, accumulation, micro):
    source, _, _ = source_state()
    source["contract"]["world_size"] = world
    source["config"]["train"].update(batch_size=batch, accumulation_steps=accumulation)
    source["micro_seen"] = micro
    source["rng_by_rank"] = source["rng_by_rank"][:world]
    target = paired_configs()["pure_c"]
    target_micro, samples, rng, _ = migrate_progress(source, target)
    assert target_micro == 10000 and samples == 80000 and len(rng) == 2
    def next_update(local_batch, num_ranks, start_micro, count):
        streams = [iter(CounterSampler(100003, 903, rank, num_ranks, local_batch, start_micro))
                   for rank in range(num_ranks)]
        return [next(streams[rank]) for _ in range(count) for rank in range(num_ranks)
                for _ in range(local_batch)]
    assert next_update(batch, world, micro, accumulation) == next_update(4, 2, target_micro, 1)


def test_migration_rejects_partial_accumulation_or_missing_rank_rng():
    source, _, _ = source_state()
    source["micro_seen"] += 1
    with pytest.raises(ValueError, match="update boundary"):
        migrate_progress(source, paired_configs()["pure_c"])
    source["micro_seen"] -= 1
    source["rng_by_rank"].pop()
    with pytest.raises(ValueError, match="RNG rank"):
        migrate_progress(source, paired_configs()["pure_c"])


@pytest.mark.parametrize("invalid", ["shape", "missing", "nonfinite"])
def test_invalid_optimizer_state_is_rejected(invalid):
    source, layout, _ = source_state()
    stored = deepcopy(source["optimizer"])
    if invalid == "shape":
        stored["state"][0]["exp_avg"] = torch.ones(9)
    elif invalid == "missing":
        stored["param_groups"].pop()
    else:
        stored["state"][0]["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError):
        remap_optimizer(stored, layout, layout, source["optimizer"])


def test_matched_ablation_alternates_two_gpu_segments_and_resumes_partial_segment():
    assert next_segment(dict(pure_c=10000, c_sf=10000), 200000, 2000) == ("c_sf", 12000)
    assert next_segment(dict(pure_c=10000, c_sf=12000), 200000, 2000) == ("pure_c", 12000)
    assert next_segment(dict(pure_c=11000, c_sf=12000), 200000, 2000) == ("pure_c", 12000)
    assert next_segment(dict(pure_c=12000, c_sf=12000), 200000, 2000) == ("c_sf", 14000)
    assert next_segment(dict(pure_c=200000, c_sf=200000), 200000, 2000) is None


def test_segment_ceiling_is_forwarded_as_a_pause_without_changing_config(tmp_path):
    run = ResidualRun(tmp_path, ["0", "1"], "python")
    run.max_updates = 12000
    command = run.joint_command(tmp_path/"config.json", "source.pt", tmp_path/"0010000.pt")
    assert command[-2:] == ["--max-updates", "12000"]
    assert "--resume" in command and "--init-checkpoint" not in command


def test_dataset_relocation_requires_the_same_source_file_list(tmp_path):
    paths = [tmp_path/"new/a.png", tmp_path/"new/nested/b.png"]
    fingerprint = hashlib.sha256('/old/data/a.png\n/old/data/nested/b.png'.encode()).hexdigest()
    source = dict(config=dict(data=dict(train=['/old/data'])), contract=dict(dataset_fingerprint=fingerprint))
    config = dict(data=dict(train=[str(tmp_path/'new')]))
    dataset = SimpleNamespace(paths=paths)
    assert check_dataset(source, config, dataset) == fingerprint
    dataset.paths = paths[:-1]
    with pytest.raises(ValueError, match="file list"):
        check_dataset(source, config, dataset)
