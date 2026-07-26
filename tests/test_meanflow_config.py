import ast
import tempfile
import unittest
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from config_compat import (
    configure_frozen_meanflow_ablation,
    normalize_pure_sana_config,
)


ROOT = Path(__file__).resolve().parents[1]


class MeanFlowConfigTest(unittest.TestCase):
    def test_only_adaln_meanflow_5a_is_accepted(self):
        config = {
            "model": {
                "params": {
                    "decode_mode": "sana",
                    "meanflow_params": {
                        "enabled": True,
                        "mode": "meanflow",
                    },
                }
            }
        }
        normalized = normalize_pure_sana_config(config)
        meanflow = normalized["model"]["params"]["meanflow_params"]
        self.assertEqual(meanflow["conditioning_mode"], "adaln")
        self.assertEqual(meanflow["objective"], "original")
        self.assertEqual(meanflow["equal_time_ratio"], 0.5)
        self.assertEqual(meanflow["endpoint_ratio"], 0.0)

        config["model"]["params"]["meanflow_params"][
            "conditioning_mode"
        ] = "text_token"
        with self.assertRaisesRegex(ValueError, "only.*adaln"):
            normalize_pure_sana_config(config)

        config["model"]["params"]["meanflow_params"] = {
            "enabled": True,
            "mode": "velocity_control",
        }
        with self.assertRaisesRegex(ValueError, "only MeanFlow 5A"):
            normalize_pure_sana_config(config)

    def test_codec_observable_objective_and_endpoint_are_validated(self):
        config = {
            "model": {
                "params": {
                    "decode_mode": "sana",
                    "meanflow_params": {
                        "enabled": True,
                        "mode": "meanflow",
                        "objective": "codec_observable",
                        "equal_time_ratio": 0.5,
                        "endpoint_ratio": 0.25,
                    },
                }
            }
        }
        normalized = normalize_pure_sana_config(config)
        meanflow = normalized["model"]["params"]["meanflow_params"]
        self.assertEqual(meanflow["objective"], "codec_observable")
        self.assertEqual(meanflow["endpoint_ratio"], 0.25)

        config["model"]["params"]["meanflow_params"][
            "objective"
        ] = "unknown"
        with self.assertRaisesRegex(ValueError, "objective must be"):
            normalize_pure_sana_config(config)

        config["model"]["params"]["meanflow_params"].update(
            {
                "objective": "original",
                "endpoint_ratio": 0.25,
            }
        )
        with self.assertRaisesRegex(ValueError, "supported only"):
            normalize_pure_sana_config(config)

        config["model"]["params"]["meanflow_params"].update(
            {
                "objective": "codec_observable",
                "equal_time_ratio": 0.8,
                "endpoint_ratio": 0.3,
            }
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            normalize_pure_sana_config(config)

    def test_meanflow_rejects_an_active_gan_schedule(self):
        config = {
            "model": {
                "params": {
                    "meanflow_params": {"enabled": True},
                    "loss_params": {"disc_start": 10},
                }
            },
            "train": {"max_steps": 10},
        }
        with self.assertRaisesRegex(ValueError, "must be no-GAN"):
            normalize_pure_sana_config(config)

    @unittest.skipIf(yaml is None, "PyYAML unavailable")
    def test_meanflow_config_matches_pure_with_equivalent_batch(self):
        baseline = yaml.safe_load(
            (ROOT / "configs" / "baseline_lrd0p4.yaml").read_text(
                encoding="utf-8"
            )
        )
        meanflow = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "pure_sana_meanflow_adaln.yaml"
            ).read_text(encoding="utf-8")
        )
        meanflow["model"]["params"].pop("meanflow_params")
        meanflow["train"]["exp_name"] = baseline["train"]["exp_name"]
        baseline_effective_batch = (
            baseline["train"]["global_batch_size"]
            * baseline["train"]["accumulation_steps"]
        )
        meanflow_effective_batch = (
            meanflow["train"]["global_batch_size"]
            * meanflow["train"]["accumulation_steps"]
        )
        self.assertEqual(meanflow_effective_batch, baseline_effective_batch)
        for config in (baseline, meanflow):
            config["train"].pop("global_batch_size")
            config["train"].pop("accumulation_steps")
        self.assertEqual(meanflow, baseline)

    @unittest.skipIf(yaml is None, "PyYAML unavailable")
    def test_codec_observable_config_only_changes_meanflow_settings(self):
        original = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "pure_sana_meanflow_adaln.yaml"
            ).read_text(encoding="utf-8")
        )
        codec_observable = yaml.safe_load(
            (
                ROOT
                / "configs"
                / "pure_sana_co_meanflow_adaln.yaml"
            ).read_text(encoding="utf-8")
        )
        original_meanflow = original["model"]["params"].pop(
            "meanflow_params"
        )
        codec_observable_meanflow = codec_observable[
            "model"
        ]["params"].pop("meanflow_params")
        codec_observable["train"]["exp_name"] = original["train"][
            "exp_name"
        ]

        self.assertEqual(original, codec_observable)
        self.assertNotIn("objective", original_meanflow)
        self.assertEqual(
            codec_observable_meanflow["objective"],
            "codec_observable",
        )
        self.assertEqual(
            codec_observable_meanflow["endpoint_ratio"],
            0.25,
        )

    def test_codec_constructor_only_adds_meanflow_arguments(self):
        source = (ROOT / "models" / "DiT_IC.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        codec_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Codec"
        )
        constructor = next(
            node
            for node in codec_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        arguments = [
            argument.arg
            for argument in constructor.args.args
        ]
        pure_sana_arguments = [
            "self",
            "device",
            "model_type",
            "dit_path",
            "codec_path",
            "use_ema",
            "use_merge",
            "training",
            "DiT_params",
            "vae_params",
            "codec_param",
            "loss_params",
            "text_encoder_path",
            "freeze_codec",
            "decode_mode",
        ]
        self.assertEqual(
            arguments[: len(pure_sana_arguments)],
            pure_sana_arguments,
        )
        self.assertEqual(
            arguments[len(pure_sana_arguments) :],
            ["meanflow_params", "require_pure_init"],
        )
        self.assertEqual(constructor.args.kwarg.arg, "kwargs")

    def test_frozen_comparison_uses_one_pure_checkpoint(self):
        config = normalize_pure_sana_config(
            {
                "model": {
                    "params": {
                        "decode_mode": "sana",
                        "loss_params": {"disc_start": 100},
                    }
                },
                "train": {
                    "max_steps": 10,
                    "exp_name": "probe",
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "pure.pt"
            checkpoint.touch()
            configured = configure_frozen_meanflow_ablation(
                config,
                "meanflow",
                str(checkpoint),
            )

        params = configured["model"]["params"]
        self.assertTrue(params["freeze_codec"])
        self.assertTrue(params["require_pure_init"])
        self.assertFalse(params["use_merge"])
        self.assertTrue(params["meanflow_params"]["enabled"])
        self.assertEqual(
            params["meanflow_params"]["objective"],
            "original",
        )
        self.assertEqual(
            params["meanflow_params"]["conditioning_mode"],
            "adaln",
        )
        self.assertEqual(
            Path(
                configured["experiment"]["frozen_meanflow_ablation"][
                    "init_checkpoint"
                ]
            ).resolve(),
            checkpoint.resolve(),
        )
        self.assertFalse(configured["train"]["resume"])

        co_config = normalize_pure_sana_config(
            {
                "model": {
                    "params": {
                        "decode_mode": "sana",
                        "loss_params": {"disc_start": 100},
                    }
                },
                "train": {
                    "max_steps": 10,
                    "exp_name": "probe",
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            co_checkpoint = Path(temp_dir) / "pure.pt"
            co_checkpoint.touch()
            co_configured = configure_frozen_meanflow_ablation(
                co_config,
                "co_meanflow",
                str(co_checkpoint),
            )

        co_meanflow = co_configured["model"]["params"][
            "meanflow_params"
        ]
        self.assertTrue(co_meanflow["enabled"])
        self.assertEqual(
            co_meanflow["objective"],
            "codec_observable",
        )
        self.assertEqual(co_meanflow["endpoint_ratio"], 0.25)

    def test_text_token_implementation_is_not_migrated(self):
        source = (
            ROOT / "models" / "compression_meanflow.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("text_token", source)
        self.assertNotIn("condition_caption", source)
        self.assertNotIn("class IntervalAdapter", source)

    def test_trainer_adds_the_weighted_meanflow_objective(self):
        source = (ROOT / "train_nogan_ddp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "meanflow_loss = model.module.latest_meanflow_loss",
            source,
        )
        self.assertIn("loss = loss + meanflow_loss", source)
        self.assertIn(
            "static_graph=not model.meanflow_enabled",
            source,
        )
        self.assertIn(
            "find_unused_parameters=model.meanflow_enabled",
            source,
        )

    def test_model_wires_training_inference_and_checkpoint_state(self):
        source = (ROOT / "models" / "DiT_IC.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("self.meanflow.compute_loss(", source)
        self.assertIn("denoised = self._meanflow_endpoint(", source)
        self.assertIn('"state_dict_meanflow"', source)
        self.assertIn("self.meanflow.load_state_dict(meanflow_state)", source)
        self.assertIn(
            "expected_mode_id = int(self.meanflow.training_mode_id.item())",
            source,
        )
        self.assertIn(
            "MeanFlow objective does not match configured",
            source,
        )
        self.assertNotIn("condition_caption(", source)

if __name__ == "__main__":
    unittest.main()
