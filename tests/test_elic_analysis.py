import tempfile
import unittest
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn

try:
    import compressai.layers  # noqa: F401
except ModuleNotFoundError:
    compressai = types.ModuleType("compressai")
    layers = types.ModuleType("compressai.layers")

    class AttentionBlock(nn.Identity):
        def __init__(self, channels):
            super().__init__()

    layers.AttentionBlock = AttentionBlock
    compressai.layers = layers
    sys.modules["compressai"] = compressai
    sys.modules["compressai.layers"] = layers

from models.elic_analysis import (
    ELICAnalysisEncoder,
    load_elic_analysis_encoder,
)


class ELICAnalysisEncoderTest(unittest.TestCase):
    def test_loader_extracts_and_freezes_historical_g_a_weights(self):
        reference = ELICAnalysisEncoder()
        checkpoint = {
            f"g_a.{key}": value.clone()
            for key, value in reference.state_dict().items()
        }
        checkpoint["g_s.unused"] = torch.ones(1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "elic.pt"
            torch.save(checkpoint, path)
            loaded = load_elic_analysis_encoder(path)

        for key, value in reference.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value)
        self.assertFalse(loaded.training)
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))


if __name__ == "__main__":
    unittest.main()
