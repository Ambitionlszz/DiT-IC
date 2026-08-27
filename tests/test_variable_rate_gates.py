import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "variable_rate_gate", ROOT / "scripts" / "variable_rate_gate.py"
)
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def write_direct_summary(path, bpp, psnr=22.0):
    path.write_text(
        json.dumps(
            {
                "checkpoint": "/tmp/lic.pt",
                "weights": "ema",
                "qualities": {
                    f"q{index}": {
                        "bpp": value,
                        "psnr": psnr + index,
                        "lpips": 0.2 - index * 0.01,
                        "dists": 0.1 - index * 0.005,
                        "ms_ssim": 0.7 + index * 0.01,
                    }
                    for index, value in enumerate(bpp)
                },
            }
        ),
        encoding="utf-8",
    )


class VariableRateGateTests(unittest.TestCase):
    def test_gate_a_selects_only_a_valid_real_bpp_ladder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last = root / "last.json"
            best = root / "best.json"
            report = root / "gate_a.json"
            write_direct_summary(last, [0.012, 0.020, 0.030, 0.042, 0.053, 0.065])
            write_direct_summary(best, [0.012, 0.020, 0.030, 0.042, 0.053, 0.065], psnr=23.0)

            GATE.select_lic(
                Namespace(
                    last_checkpoint=root / "last.pt",
                    last_summary=last,
                    last_eval_dir=root / "last",
                    best_checkpoint=root / "best.pt",
                    best_summary=best,
                    best_eval_dir=root / "best",
                    output=report,
                    min_bpp_span=0.02,
                    q0_max_bpp=0.03,
                    q5_min_bpp=0.045,
                    q5_max_bpp=0.08,
                )
            )
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["selected"]["name"], "best")

    def test_gate_a_rejects_non_monotonic_real_bpp(self):
        metrics = {
            f"q{index}": {"bpp": value}
            for index, value in enumerate([0.012, 0.020, 0.019, 0.042, 0.053, 0.065])
        }
        failures = GATE.check_rate_ladder(metrics, 0.02, 0.03, 0.045, 0.08)
        self.assertTrue(any("strictly increasing" in failure for failure in failures))

    def test_gate_b_requires_byte_identical_frozen_codec_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lic_eval = root / "lic"
            p0_root = root / "p0"
            bpp = [0.012, 0.020, 0.030, 0.042, 0.053, 0.065]
            summary = root / "lic_summary.json"
            write_direct_summary(summary, bpp)
            for index, value in enumerate(bpp):
                quality = f"q{index}"
                (lic_eval / quality / "bin").mkdir(parents=True)
                (p0_root / quality / "bin").mkdir(parents=True)
                (lic_eval / quality / "bin" / "image.bin").write_bytes(b"packet")
                (p0_root / quality / "bin" / "image.bin").write_bytes(b"packet")
                metrics_dir = p0_root / quality / "metrics" / "run"
                metrics_dir.mkdir(parents=True)
                (metrics_dir / "summary_metrics.json").write_text(
                    json.dumps(
                        {
                            "avg_bpp": value,
                            "psnr": 22.0,
                            "lpips": 0.2,
                            "dists": 0.1,
                            "ms_ssim": 0.7,
                        }
                    ),
                    encoding="utf-8",
                )

            report = root / "gate_b.json"
            GATE.verify_p0(
                Namespace(
                    lic_summary=summary,
                    lic_eval_dir=lic_eval,
                    p0_root=p0_root,
                    output=report,
                )
            )
            self.assertTrue(json.loads(report.read_text(encoding="utf-8"))["accepted"])


if __name__ == "__main__":
    unittest.main()
