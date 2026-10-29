"""The requested batch increase preserves optimizer state and sample position."""
from copy import deepcopy
from pathlib import Path
import unittest

import torch

from scripts.migrate_joint_batch import migrate_batch
from models.bt_self_flow.data import CounterSampler
from models.bt_self_flow.experiment import load_config


class BatchMigrationTest(unittest.TestCase):
    def fixture(self):
        config = load_config(Path(__file__).resolve().parents[1]/"configs/bt_self_flow/joint_sf_after20k.yaml")
        contract = deepcopy(config)
        contract["world_size"] = 2
        old = deepcopy(contract)
        old["train"].update(batch_size=2, effective_batch=8)
        state = dict(phase="joint", backend="D", step=472, micro_seen=944,
                     contract=old, raw={"weight": torch.tensor([1.])}, ema={"weight": torch.tensor([2.])},
                     optimizer={"state": {0: {"step": 472, "exp_avg": torch.tensor([3.])}}},
                     aux_optimizer={"step": 472}, scheduler={"last_epoch": 472},
                     scaler={"scale": 65536.}, rng_by_rank=[{"rank": 0}, {"rank": 1}], attempts=472)
        return state, config, contract

    def test_preserves_state_and_global_next_images(self):
        state, config, contract = self.fixture()
        migrated, report = migrate_batch(state, config, contract)
        self.assertEqual(migrated["micro_seen"], 472)
        self.assertEqual(report["global_images_seen"], 3776)
        for key in ("raw", "ema", "optimizer", "aux_optimizer", "scheduler", "scaler", "rng_by_rank"):
            self.assertIs(migrated[key], state[key])
        self.assertEqual(migrated["step"], state["step"])
        self.assertEqual(state["contract"]["train"]["batch_size"], 2)
        # A new rank-local batch regroups two old global microbatches. Across
        # both ranks it must consume exactly the next eight images/crops.
        old_keys, new_keys = [], []
        for rank in range(2):
            old_sampler = iter(CounterSampler(10000, 903, rank, 2, 2, 944, 23))
            new_sampler = iter(CounterSampler(10000, 903, rank, 2, 4, 472, 23))
            old_keys.extend(next(old_sampler) for _ in range(4))
            new_keys.extend(next(new_sampler) for _ in range(4))
        self.assertCountEqual(new_keys, old_keys)

    def test_rejects_unrelated_changes_or_an_unaligned_cursor(self):
        state, config, contract = self.fixture()
        changed = deepcopy(contract)
        changed["optimizer"]["backend_lr"] *= 2
        with self.assertRaisesRegex(ValueError, "other than batch"):
            migrate_batch(state, config, changed)
        state["micro_seen"] += 1
        with self.assertRaisesRegex(ValueError, "not aligned"):
            migrate_batch(state, config, contract)

    def test_rejects_migration_after_teacher_activation(self):
        state, config, contract = self.fixture()
        state["teacher"] = {}
        with self.assertRaisesRegex(ValueError, "RD-only"):
            migrate_batch(state, config, contract)


if __name__ == "__main__":
    unittest.main()
