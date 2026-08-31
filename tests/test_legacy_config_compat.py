import unittest

from config_compat import normalize_pure_sana_config, prepare_inference_config
from eval.testing_utils import parse_args_testing


class LegacyConfigCompatibilityTest(unittest.TestCase):
    def test_baseline_rejects_active_meanflow_config(self):
        config = {
            "model": {
                "params": {
                    "meanflow_params": {
                        "enabled": True,
                        "mode": "meanflow",
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "baseline branch"):
            normalize_pure_sana_config(config)

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
        self.assertEqual(normalized["optimizer"]["aux_lr"], 0.001)
        self.assertEqual(normalized["optimizer"]["entropy_lr"], 0.001)

    def test_current_entropy_lr_is_mapped_to_official_aux_lr(self):
        config = {
            "model": {"params": {"use_text_condition": True}},
            "optimizer": {"entropy_lr": 0.001},
        }

        normalized = normalize_pure_sana_config(config)

        self.assertEqual(normalized["optimizer"]["aux_lr"], 0.001)

    def test_conflicting_entropy_learning_rates_are_rejected(self):
        config = {
            "model": {"params": {"use_text_condition": True}},
            "optimizer": {
                "aux_lr": 0.001,
                "entropy_lr": 0.002,
            },
        }

        with self.assertRaisesRegex(
            ValueError,
            "Conflicting entropy learning rates",
        ):
            normalize_pure_sana_config(config)

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

    def test_conflicting_rate_weight_aliases_are_rejected(self):
        config = {
            "model": {
                "params": {
                    "use_text_condition": True,
                    "loss_params": {"rate_weight": 0.5},
                }
            },
            "train": {"lrd": 0.4, "rate_weight": 0.5},
        }
        with self.assertRaisesRegex(ValueError, "Conflicting rate weights"):
            normalize_pure_sana_config(config)

    def test_training_rate_weight_overrides_legacy_kl_default(self):
        config = {
            "model": {
                "params": {
                    "use_text_condition": True,
                    "loss_params": {"kl_weight": 0.1},
                }
            },
            "train": {"lrd": 0.4},
        }
        normalized = normalize_pure_sana_config(config)
        self.assertEqual(normalized["train"]["rate_weight"], 0.4)
        self.assertEqual(
            normalized["model"]["params"]["loss_params"]["rate_weight"],
            0.4,
        )

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

    def test_multirate_plain_resolves_six_rate_codec_contract(self):
        config = {
            "model": {
                "params": {
                    "freeze_codec": True,
                    "train_text_projection": True,
                    "vae_params": {
                        "vae_lora": True,
                        "train_vae_lora": False,
                    },
                    "codec_param": {"variable_rate": True},
                    "loss_params": {"rate_weight": 0.0},
                }
            },
            "quality_control": {"enabled": True, "num_levels": 6},
            "train": {"rate_weight": 0.0},
        }
        normalized = normalize_pure_sana_config(config)
        codec = normalized["model"]["params"]["codec_param"]
        self.assertTrue(codec["variable_rate"])
        self.assertEqual(codec["num_quality_levels"], 6)

    def test_multirate_plain_requires_decoder_only_trainable_vae_lora(self):
        config = {
            "model": {
                "params": {
                    "freeze_codec": True,
                    "train_text_projection": True,
                    "vae_params": {
                        "vae_lora": True,
                        "train_vae_lora": True,
                    },
                    "loss_params": {"rate_weight": 0.0},
                }
            },
            "quality_control": {"enabled": True, "num_levels": 6},
            "train": {"rate_weight": 0.0},
        }
        with self.assertRaisesRegex(ValueError, "decoder-only VAE LoRA"):
            normalize_pure_sana_config(config)

    def test_multirate_plain_allows_decoder_only_trainable_vae_lora(self):
        config = {
            "model": {
                "params": {
                    "freeze_codec": True,
                    "train_text_projection": True,
                    "vae_params": {
                        "vae_lora": True,
                        "train_vae_lora": True,
                        "decoder_lora_only": True,
                    },
                    "loss_params": {"rate_weight": 0.0},
                }
            },
            "quality_control": {"enabled": True, "num_levels": 6},
            "train": {"rate_weight": 0.0},
        }
        normalized = normalize_pure_sana_config(config)
        self.assertTrue(
            normalized["model"]["params"]["vae_params"]["decoder_lora_only"]
        )

    def test_multirate_plain_requires_trainable_text_projection(self):
        config = {
            "model": {
                "params": {
                    "freeze_codec": True,
                    "train_text_projection": False,
                    "vae_params": {
                        "vae_lora": False,
                        "train_vae_lora": False,
                    },
                    "loss_params": {"rate_weight": 0.0},
                }
            },
            "quality_control": {"enabled": True, "num_levels": 6},
            "train": {"rate_weight": 0.0},
        }
        with self.assertRaisesRegex(ValueError, "train_text_projection=True"):
            normalize_pure_sana_config(config)

    def test_end_to_end_variable_rate_requires_unfrozen_codec_and_phase(self):
        config = {
            "model": {
                "params": {
                    "freeze_codec": False,
                    "use_text_condition": True,
                    "codec_param": {"variable_rate": True},
                    "loss_params": {"rate_weight": 0.0},
                }
            },
            "quality_control": {
                "enabled": True,
                "num_levels": 6,
                "rate_weights": [16, 8, 4, 2, 1, 0.5],
            },
            "train": {"rate_weight": 0.0, "e2e_phase": "nogan"},
        }
        normalized = normalize_pure_sana_config(config)
        self.assertEqual(
            normalized["quality_control"]["rate_weights"],
            [16, 8, 4, 2, 1, 0.5],
        )

        config["model"]["params"]["freeze_codec"] = True
        with self.assertRaisesRegex(ValueError, "freeze_codec=False"):
            normalize_pure_sana_config(config)

        config["model"]["params"]["freeze_codec"] = False
        config["train"]["e2e_phase"] = "invalid"
        with self.assertRaisesRegex(ValueError, "e2e_phase"):
            normalize_pure_sana_config(config)


if __name__ == "__main__":
    unittest.main()
