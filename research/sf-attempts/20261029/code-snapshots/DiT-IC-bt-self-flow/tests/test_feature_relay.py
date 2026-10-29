import unittest

import torch

from models.feature_relay import DecoderFeatureRelay


class DecoderFeatureRelayTest(unittest.TestCase):
    def _make_relay(self, mode):
        return DecoderFeatureRelay(
            mode=mode,
            stage_channels=(16, 16, 8),
            stage_block_indices=(3, 2, 1),
            source_channels=20,
            hidden_channels=8,
        )

    def test_capacity_control_and_bitstream_relay_have_equal_parameters(self):
        decoder_self = self._make_relay("decoder_self")
        bitstream = self._make_relay("bitstream")
        self.assertEqual(
            sum(parameter.numel() for parameter in decoder_self.parameters()),
            sum(parameter.numel() for parameter in bitstream.parameters()),
        )

    def test_zero_initialized_sft_is_exact_identity(self):
        hidden = torch.randn(2, 16, 12, 10)
        bridge = torch.randn(2, 20, 3, 3)
        for mode in ("decoder_self", "bitstream"):
            relay = self._make_relay(mode)
            output = relay(3, hidden, bridge)
            self.assertTrue(torch.equal(output, hidden))

    def test_control_ignores_bridge_and_bgfr_uses_it(self):
        hidden = torch.randn(1, 16, 8, 8)
        bridge_a = torch.randn(1, 20, 2, 2)
        bridge_b = torch.randn(1, 20, 2, 2)

        decoder_self = self._make_relay("decoder_self")
        bitstream = self._make_relay("bitstream")
        with torch.no_grad():
            decoder_self.heads["3"].out_conv.weight.fill_(0.01)
            bitstream.heads["3"].out_conv.weight.fill_(0.01)

        self.assertTrue(
            torch.equal(
                decoder_self(3, hidden, bridge_a),
                decoder_self(3, hidden, bridge_b),
            )
        )
        self.assertFalse(
            torch.equal(
                bitstream(3, hidden, bridge_a),
                bitstream(3, hidden, bridge_b),
            )
        )

    def test_contract_contains_no_explicit_rate_condition(self):
        contract = self._make_relay("bitstream").contract()
        self.assertEqual(contract["source"], "codec_g_s_preprojection_320ch")
        self.assertNotIn("rate", contract)
        self.assertNotIn("quality", contract)


if __name__ == "__main__":
    unittest.main()
