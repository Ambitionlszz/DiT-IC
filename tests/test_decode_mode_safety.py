import unittest

from config_compat import normalize_pure_sana_config
from decode_modes import validate_decode_checkpoint


class DecodeModeSafetyTest(unittest.TestCase):
    def test_only_plain_sana_decode_is_supported(self):
        config = {"model": {"params": {"decode_mode": "gs_residual"}}}
        with self.assertRaisesRegex(ValueError, "decode_mode must be one of"):
            normalize_pure_sana_config(config)

    def test_plain_sana_rejects_gs_residual_checkpoint_state(self):
        with self.assertRaisesRegex(ValueError, "contains g_s_fusion weights"):
            validate_decode_checkpoint(
                "sana",
                True,
                training=False,
                allow_missing_gs_fusion=False,
                source="checkpoint",
            )

    def test_plain_sana_accepts_plain_checkpoint_state(self):
        validate_decode_checkpoint(
            "sana",
            False,
            training=False,
            allow_missing_gs_fusion=False,
            source="checkpoint",
        )


if __name__ == "__main__":
    unittest.main()
