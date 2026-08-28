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


def write_direct_summary(
    path,
    bpp,
    checkpoint,
    psnr=22.0,
    weights="ema",
    image_manifest_sha256="1" * 64,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "summary_schema_version": 1,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": GATE.sha256(checkpoint),
                "weights": weights,
                "image_manifest_sha256": image_manifest_sha256,
                "num_images": 24,
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
            last_checkpoint = root / "last.pt"
            best_checkpoint = root / "best.pt"
            last_checkpoint.write_bytes(b"last checkpoint")
            best_checkpoint.write_bytes(b"best checkpoint")
            last = root / "last" / "summary.json"
            best = root / "best" / "summary.json"
            report = root / "gate_a.json"
            write_direct_summary(
                last,
                [0.012, 0.020, 0.030, 0.042, 0.053, 0.065],
                last_checkpoint,
            )
            write_direct_summary(
                best,
                [0.012, 0.020, 0.030, 0.042, 0.053, 0.065],
                best_checkpoint,
                psnr=23.0,
            )

            GATE.select_lic(
                Namespace(
                    last_checkpoint=last_checkpoint,
                    last_summary=last,
                    last_eval_dir=root / "last",
                    best_checkpoint=best_checkpoint,
                    best_summary=best,
                    best_eval_dir=root / "best",
                    output=report,
                    min_bpp_span=0.02,
                    q0_max_bpp=None,
                    q5_min_bpp=None,
                    q5_max_bpp=None,
                )
            )
            gate_report = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(gate_report["selected"]["name"], "best")
            self.assertIn("matched_mean_psnr", gate_report["candidates"][0])
            self.assertEqual(gate_report["candidates"][0]["weights"], "ema")

    def test_gate_a_rejects_a_summary_for_another_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "last.pt"
            other_checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"last checkpoint")
            other_checkpoint.write_bytes(b"best checkpoint")
            summary = root / "last" / "summary.json"
            write_direct_summary(
                summary,
                [0.012, 0.020, 0.030, 0.042, 0.053, 0.065],
                other_checkpoint,
            )

            with self.assertRaisesRegex(ValueError, "does not match"):
                GATE.select_lic(
                    Namespace(
                        last_checkpoint=checkpoint,
                        last_summary=summary,
                        last_eval_dir=root / "last",
                        best_checkpoint=None,
                        best_summary=None,
                        best_eval_dir=None,
                        output=root / "gate_a.json",
                        min_bpp_span=0.02,
                        q0_max_bpp=0.03,
                        q5_min_bpp=0.045,
                        q5_max_bpp=0.08,
                    )
                )

    def test_gate_a_requires_ema_and_one_shared_image_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            last_checkpoint = root / "last.pt"
            best_checkpoint = root / "best.pt"
            last_checkpoint.write_bytes(b"last checkpoint")
            best_checkpoint.write_bytes(b"best checkpoint")
            last = root / "last" / "summary.json"
            best = root / "best" / "summary.json"
            bpp = [0.012, 0.020, 0.030, 0.042, 0.053, 0.065]
            write_direct_summary(last, bpp, last_checkpoint)
            write_direct_summary(
                best,
                bpp,
                best_checkpoint,
                image_manifest_sha256="2" * 64,
            )

            with self.assertRaisesRegex(ValueError, "same image set"):
                GATE.select_lic(
                    Namespace(
                        last_checkpoint=last_checkpoint,
                        last_summary=last,
                        last_eval_dir=root / "last",
                        best_checkpoint=best_checkpoint,
                        best_summary=best,
                        best_eval_dir=root / "best",
                        output=root / "gate_a.json",
                        min_bpp_span=0.02,
                        q0_max_bpp=0.03,
                        q5_min_bpp=0.045,
                        q5_max_bpp=0.08,
                    )
                )

            write_direct_summary(best, bpp, best_checkpoint, weights="model")
            with self.assertRaisesRegex(ValueError, "requires EMA"):
                GATE.select_lic(
                    Namespace(
                        last_checkpoint=last_checkpoint,
                        last_summary=last,
                        last_eval_dir=root / "last",
                        best_checkpoint=best_checkpoint,
                        best_summary=best,
                        best_eval_dir=root / "best",
                        output=root / "gate_a.json",
                        min_bpp_span=0.02,
                        q0_max_bpp=0.03,
                        q5_min_bpp=0.045,
                        q5_max_bpp=0.08,
                    )
                )

    def test_gate_a_rejects_non_monotonic_real_bpp(self):
        metrics = {
            f"q{index}": {"bpp": value}
            for index, value in enumerate([0.012, 0.020, 0.019, 0.042, 0.053, 0.065])
        }
        failures = GATE.check_rate_ladder(metrics, 0.02, 0.03, 0.045, 0.08)
        self.assertTrue(any("strictly increasing" in failure for failure in failures))

    def test_gate_a_has_no_fixed_endpoint_bounds_by_default(self):
        metrics = {
            f"q{index}": {"bpp": value}
            for index, value in enumerate([0.035, 0.044, 0.053, 0.062, 0.071, 0.080])
        }
        self.assertEqual(GATE.check_rate_ladder(metrics, 0.02), [])

    def test_gate_a_rejects_a_quality_reversal(self):
        metrics = {
            f"q{index}": {
                "psnr": 22.0 + index,
                "ms_ssim": 0.70 + index * 0.01,
                "lpips": 0.20 - index * 0.01,
                "dists": 0.10 - index * 0.005,
            }
            for index in range(6)
        }
        metrics["q3"]["lpips"] = metrics["q2"]["lpips"] + 0.01
        failures = GATE.check_quality_ladder(metrics)
        self.assertTrue(any("lpips" in failure for failure in failures))

    def test_gate_b_requires_byte_identical_frozen_codec_bins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lic_eval = root / "lic"
            p0_root = root / "p0"
            bpp = [0.012, 0.020, 0.030, 0.042, 0.053, 0.065]
            summary = root / "lic_summary.json"
            checkpoint = root / "lic.pt"
            checkpoint.write_bytes(b"checkpoint")
            write_direct_summary(summary, bpp, checkpoint)
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
