import unittest
from pathlib import Path

import yaml

from config_compat import normalize_pure_sana_config


ROOT = Path(__file__).resolve().parents[1]


class AuxEDConfigTest(unittest.TestCase):
    def test_lrd1p0_run_starts_from_zero(self):
        path = ROOT / "configs" / "auxed_lrd1p0.yaml"
        with path.open(encoding="utf-8") as handle:
            config = normalize_pure_sana_config(yaml.safe_load(handle))

        params = config["model"]["params"]
        train = config["train"]
        self.assertTrue(params["use_aux_encoder"])
        self.assertTrue(params["use_aux_decoder"])
        self.assertIsNone(params["codec_path"])
        self.assertFalse(train["resume"])
        self.assertEqual(train["lrd"], 1.0)
        self.assertEqual(train["rate_weight"], 1.0)
        self.assertEqual(
            params["loss_params"]["rate_weight"],
            1.0,
        )
        self.assertEqual(train["exp_name"], "auxed_lrd1p0_bs8_200k")
        self.assertNotIn("baseline", train["exp_name"])

    def test_main_entropy_lrd0p5_run_starts_from_zero(self):
        path = ROOT / "configs" / "auxed_lrd0p5_main_entropy.yaml"
        with path.open(encoding="utf-8") as handle:
            config = normalize_pure_sana_config(yaml.safe_load(handle))

        params = config["model"]["params"]
        train = config["train"]
        optimizer = config["optimizer"]
        self.assertTrue(params["use_aux_encoder"])
        self.assertTrue(params["use_aux_decoder"])
        self.assertIsNone(params["codec_path"])
        self.assertFalse(train["resume"])
        self.assertEqual(train["accumulation_steps"], 1)
        self.assertEqual(train["lrd"], 0.5)
        self.assertEqual(train["rate_weight"], 0.5)
        self.assertEqual(
            params["loss_params"]["rate_weight"],
            0.5,
        )
        self.assertEqual(optimizer["entropy_lr"], 0.001)
        self.assertEqual(
            train["exp_name"],
            "auxed_lrd0p5_main_entropy_bs8_200k",
        )


if __name__ == "__main__":
    unittest.main()
