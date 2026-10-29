from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import eval_bt_self_flow as evaluator
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, save_atomic
from models.bt_self_flow.losses import ImageLoss
from test_bt_backend import tiny_model
from test_bt_training import small_config


class EvaluationTest(unittest.TestCase):
    def test_packet_report_and_source_free_decode_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"images").mkdir()
            Image.fromarray(np.random.default_rng(906).integers(0, 256, (5, 9, 3), dtype=np.uint8)).save(root/"images/small.png")
            config = small_config(root, "joint")
            model = tiny_model("D").eval()
            save_atomic(root/"checkpoint.pt", {"method": METHOD, "schema": SCHEMA, "phase": "joint", "backend": "D",
                                             "step": 2, "config": config, "raw": model.export_state(),
                                             "ema": model.export_state(), "assets": {"test": "tiny"}})
            decoding = [False]
            def factory(cfg, device):
                result = tiny_model("D").to(device).eval()
                if decoding[0]:
                    def forbidden(*args, **kwargs):
                        raise AssertionError("standalone decode used sender-only input")
                    result.encode_clean_latent = forbidden
                    result.codec.router.forward = forbidden
                return result
            args = SimpleNamespace(checkpoint=str(root/"checkpoint.pt"), device="cpu", weights="ema",
                                   input=[str(root/"images")], limit=1, output=str(root/"eval"), decode_packet=None,
                                   qualities=[.2, .5, .8], check_forward=True, save_images=True)
            with patch.object(evaluator, "make_model", side_effect=factory), \
                 patch.object(evaluator, "asset_identity", return_value={"test": "tiny"}), \
                 patch.object(evaluator, "ImageLoss", side_effect=lambda: ImageLoss(lpips_weight=0, dists_weight=0)) as metrics, \
                 redirect_stdout(io.StringIO()):
                evaluator.evaluate(args)
                summary = json.loads((root/"eval/summary.json").read_text())
                rows = [json.loads(line) for line in (root/"eval/images.jsonl").read_text().splitlines()]
                self.assertEqual(len(summary["summary"]), 3)
                self.assertEqual(len(rows), 3)
                for row in rows:
                    self.assertEqual(row["nfe"], 1)
                    self.assertAlmostEqual(row["bpp"], 8*row["bytes"]/(5*9))
                    self.assertEqual(sum(row["bytes_by_component"].values()), row["bytes"])
                    self.assertAlmostEqual(sum(row["byte_fractions"].values()), 1.)
                    self.assertIn("bpp_coding_gap", row)
                    self.assertIn("repair_latent_rms", row)
                for point in summary["summary"].values():
                    self.assertAlmostEqual(sum(point["byte_fractions"].values()), 1.)
                packet = next((root/"eval").rglob("*.btic"))
                expected = np.array(Image.open(packet.with_suffix(".png")))
                (root/"images").rename(root/"unavailable_source")
                decoding[0] = True
                args.input, args.decode_packet, args.output = None, str(packet), str(root/"decoded.png")
                evaluator.evaluate(args)
                self.assertEqual(metrics.call_count, 1)  # packet-only decode builds no perceptual network
                actual = np.array(Image.open(root/"decoded.png"))
                self.assertTrue(np.array_equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
