import unittest

from config_compat import normalize_pure_sana_config, prepare_inference_config
from eval.testing_utils import parse_args_testing


class LegacyConfigCompatibilityTest(unittest.TestCase):
    def test_legacy_rate_and_entropy_names_are_mapped(self):
        config = {
            "model": {
                "model_type": "Codec",
                "params": {
                    "use_aux_encoder": False,
                    "use_aux_decoder": False,
                    "use_variance_guide": False,
                    "use_self_dist": False,
                    "use_text_condition": True,
                    "loss_params": {"kl_weight": 0.1, "disc_start": 18000},
                },
            },
            "train": {"lrd": 0.4},
            "optimizer": {"aux_lr": 0.001},
        }

        normalized = normalize_pure_sana_config(config)

        self.assertEqual(normalized["train"]["rate_weight"], 0.4)
        self.assertEqual(
            normalized["model"]["params"]["loss_params"]["rate_weight"],
            0.4,
        )
        self.assertEqual(normalized["optimizer"]["entropy_lr"], 0.001)

    def test_removed_feature_cannot_be_reenabled(self):
        config = {
            "model": {
                "params": {
                    "use_variance_guide": True,
                    "use_text_condition": True,
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "removed DiT-IC features"):
            normalize_pure_sana_config(config)

    def test_training_yaml_values_are_not_overridden_by_cli_defaults(self):
        args = parse_args_testing(["--config_path", "legacy.yaml"])
        self.assertIsNone(args.codec_path)
        self.assertIsNone(args.use_ema)
        self.assertIsNone(args.use_merge)

        config = {
            "model": {
                "params": {
                    "codec_path": "/legacy/best_ckpt.pt",
                    "use_ema": True,
                    "use_merge": False,
                    "use_text_condition": True,
                }
            }
        }
        prepared = prepare_inference_config(
            config,
            codec_path=args.codec_path,
            use_ema=args.use_ema,
            use_merge=args.use_merge,
        )
        params = prepared["model"]["params"]
        self.assertEqual(params["codec_path"], "/legacy/best_ckpt.pt")
        self.assertTrue(params["use_ema"])
        self.assertFalse(params["training"])

    def test_freeze_codec_can_be_declared_in_train_section(self):
        config = {
            "model": {"params": {"use_text_condition": True}},
            "train": {"freeze_codec": True},
        }
        normalized = normalize_pure_sana_config(config)
        self.assertTrue(normalized["model"]["params"]["freeze_codec"])


if __name__ == "__main__":
    unittest.main()
