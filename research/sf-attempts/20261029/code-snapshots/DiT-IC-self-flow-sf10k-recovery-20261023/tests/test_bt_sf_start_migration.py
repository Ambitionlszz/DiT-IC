from copy import deepcopy
from pathlib import Path
import unittest

import torch

from models.bt_self_flow.experiment import load_config
from scripts.migrate_sf_start import migrate_start


class SFStartMigrationTest(unittest.TestCase):
    def fixture(self):
        config = load_config(Path(__file__).resolve().parents[1]/"configs/bt_self_flow/joint_sf_after20k.yaml")
        contract = deepcopy(config)
        contract["world_size"] = 2
        old = deepcopy(contract)
        old["self_flow"]["start_step"] = 20000
        state = dict(phase="joint", backend="D", contract=old, step=8000, micro_seen=15528,
                     raw={"weight": torch.ones(2)}, ema={"weight": torch.ones(2)*2},
                     optimizer={"step": 8000}, aux_optimizer={"step": 8000}, scheduler={"step": 8000},
                     scaler={"scale": 65536.}, rng_by_rank=[{}, {}], attempts=8000)
        return state, config, contract

    def test_changes_only_boundary_and_preserves_all_training_state(self):
        state, config, contract = self.fixture()
        migrated, report = migrate_start(state, config, contract)
        for key in state.keys()-{"contract"}:
            self.assertIs(migrated[key], state[key])
        self.assertEqual(state["contract"]["self_flow"]["start_step"], 20000)
        self.assertEqual(migrated["contract"]["self_flow"]["start_step"], 10000)
        self.assertEqual(report["first_sf_update"], 10001)

    def test_rejects_other_changes_and_an_active_teacher(self):
        state, config, contract = self.fixture()
        changed = deepcopy(contract)
        changed["train"]["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "other than"):
            migrate_start(state, config, changed)
        state["teacher"] = {}
        with self.assertRaisesRegex(ValueError, "pure-RD"):
            migrate_start(state, config, contract)


if __name__ == "__main__":
    unittest.main()
