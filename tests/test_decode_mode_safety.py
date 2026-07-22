import tempfile
import unittest
from pathlib import Path

import yaml

from config_compat import (
    configure_frozen_decode_ablation,
    normalize_pure_sana_config,
)
from decode_modes import validate_decode_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class DecodeModeSafetyTest(unittest.TestCase):
    def test_invalid_decode_mode_is_rejected(self):
        config = {"model": {"params": {"decode_mode": "gs_residul"}}}
        with self.assertRaisesRegex(ValueError, "decode_mode must be one of"):
            normalize_pure_sana_config(config)

    def test_inference_requires_matching_fusion_state(self):
        with self.assertRaisesRegex(ValueError, "has no g_s_fusion weights"):
            validate_decode_checkpoint(
                "gs_residual",
                False,
                training=False,
                allow_missing_gs_fusion=False,
                source="checkpoint",
            )
        with self.assertRaisesRegex(ValueError, "contains g_s_fusion weights"):
            validate_decode_checkpoint(
                "sana",
                True,
                training=False,
                allow_missing_gs_fusion=False,
                source="checkpoint",
            )

    def test_missing_fusion_requires_explicit_training_warm_start(self):
        with self.assertRaisesRegex(ValueError, "has no g_s_fusion weights"):
            validate_decode_checkpoint(
                "gs_residual",
                False,
                training=True,
                allow_missing_gs_fusion=False,
                source="checkpoint",
            )
        validate_decode_checkpoint(
            "gs_residual",
            False,
            training=True,
            allow_missing_gs_fusion=True,
            source="checkpoint",
        )

    def test_frozen_ablation_uses_common_checkpoint_and_freezes_codec(self):
        config = {
            "model": {"params": {"decode_mode": "gs_residual"}},
            "train": {"resume": False},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "pure.pt"
            checkpoint.touch()
            configured = configure_frozen_decode_ablation(
                config,
                "gs_residual",
                str(checkpoint),
            )

        params = configured["model"]["params"]
        self.assertTrue(params["freeze_codec"])
        self.assertFalse(params["use_merge"])
        self.assertTrue(params["allow_missing_gs_fusion_for_init"])
        self.assertEqual(params["codec_path"], str(checkpoint.resolve()))

    def test_frozen_ablation_rejects_config_mode_mismatch(self):
        config = {
            "model": {"params": {"decode_mode": "sana"}},
            "train": {"resume": False},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "pure.pt"
            checkpoint.touch()
            with self.assertRaisesRegex(ValueError, "does not match"):
                configure_frozen_decode_ablation(
                    config,
                    "gs_residual",
                    str(checkpoint),
                )

    def test_paired_configs_only_change_decode_mode_and_experiment_name(self):
        baseline = yaml.safe_load(
            (ROOT / "configs" / "baseline_lrd0p4.yaml").read_text(
                encoding="utf-8"
            )
        )
        gs_residual = yaml.safe_load(
            (ROOT / "configs" / "pure_sana_ablation_gs_residual.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(
            baseline["model"]["params"].get("freeze_codec", False)
        )
        self.assertFalse(
            gs_residual["model"]["params"].get("freeze_codec", False)
        )

        baseline["model"]["params"]["decode_mode"] = "paired"
        gs_residual["model"]["params"]["decode_mode"] = "paired"
        baseline["train"]["exp_name"] = "paired"
        gs_residual["train"]["exp_name"] = "paired"
        self.assertEqual(baseline, gs_residual)

if __name__ == "__main__":
    unittest.main()
