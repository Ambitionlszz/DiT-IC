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
    schema=1,
    protocol=None,
    dataset_content_sha256=None,
    num_images=24,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "summary_schema_version": schema,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": GATE.sha256(checkpoint),
        "weights": weights,
        "image_manifest_sha256": image_manifest_sha256,
        "num_images": num_images,
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
    if protocol is not None:
        summary["metric_protocol"] = protocol
        for values in summary["qualities"].values():
            values["ms_ssim_rgb"] = values["ms_ssim"]
            values["ms_ssim_y"] = values["ms_ssim"] + .01
    if dataset_content_sha256 is not None:
        summary["dataset_content_sha256"] = dataset_content_sha256
    path.write_text(json.dumps(summary), encoding="utf-8")


def canonical_protocol(domain="float"):
    return {
        "version": "image_quality_rgb_y_v2",
        "pixel_domain": domain,
        "ms_ssim_alias": "ms_ssim_rgb",
        "ms_ssim_rgb": {"implementation": "pytorch_msssim", "data_range": 1},
        "ms_ssim_y": {"implementation": "pyiqa.MS_SSIM", "color_space": "yiq"},
    }


def two_candidates(root, **kwargs):
    bpp = [0.012, 0.020, 0.030, 0.042, 0.053, 0.065]
    for name in ("last", "best"):
        checkpoint = root/f"{name}.pt"
        checkpoint.write_bytes(name.encode())
        write_direct_summary(root/name/"summary.json", bpp, checkpoint, **kwargs)
    return Namespace(
        last_checkpoint=root/"last.pt", last_summary=root/"last/summary.json", last_eval_dir=root/"last",
        best_checkpoint=root/"best.pt", best_summary=root/"best/summary.json", best_eval_dir=root/"best",
        output=root/"gate_a.json", min_bpp_span=.02, q0_max_bpp=None, q5_min_bpp=None, q5_max_bpp=None,
    )


def frozen_codec_evaluations(root, modern=False):
    lic_eval, plain_root = root/"lic", root/"p0"
    bpp = [.012, .020, .030, .042, .053, .065]
    checkpoint = root/"lic.pt"
    checkpoint.write_bytes(b"checkpoint")
    options = {"schema": 2, "protocol": canonical_protocol(), "dataset_content_sha256": "3"*64} if modern else {}
    summary_path = lic_eval/"summary.json"
    write_direct_summary(summary_path, bpp, checkpoint, num_images=1, **options)
    for index, value in enumerate(bpp):
        quality = f"q{index}"
        for directory, name in ((lic_eval/quality/"bin", "image.bin"), (plain_root/quality/"bin", "image")):
            directory.mkdir(parents=True)
            (directory/name).write_bytes(b"packet")
        metrics_dir = plain_root/quality/"metrics/run"
        metrics_dir.mkdir(parents=True)
        values = {"avg_bpp": value, "psnr": 22., "lpips": .2, "dists": .1, "ms_ssim": .7}
        if modern:
            values.update(metric_protocol=canonical_protocol(), dataset_content_sha256="3"*64,
                          ms_ssim_rgb=.7, ms_ssim_y=.71)
        (metrics_dir/"summary_metrics.json").write_text(json.dumps(values))
    return Namespace(lic_summary=summary_path, lic_eval_dir=lic_eval, p0_root=plain_root, output=root/"gate_b.json")


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
            self.assertEqual(gate_report["candidates"][0]["metric_protocol"], GATE.LEGACY_METRIC_PROTOCOL)

    def test_gate_a_accepts_matching_v2_protocol_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = two_candidates(root, schema=2, protocol=canonical_protocol(), dataset_content_sha256="3"*64)
            GATE.select_lic(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["candidates"][0]["dataset_content_sha256"], "3"*64)
            self.assertEqual(report["candidates"][0]["metric_protocol"], canonical_protocol())

    def test_gate_a_rejects_mixed_legacy_and_v2_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = two_candidates(root)
            write_direct_summary(args.best_summary, [.012, .020, .030, .042, .053, .065],
                                 args.best_checkpoint, schema=2, protocol=canonical_protocol(),
                                 dataset_content_sha256="3"*64)
            with self.assertRaisesRegex(ValueError, "metric protocols differ"):
                GATE.select_lic(args)

    def test_gate_a_rejects_different_domains_and_changed_image_contents(self):
        for field, value, expected in (
            ("pixel_domain", "uint8_roundtrip", "metric protocols differ"),
            ("dataset_content_sha256", "4"*64, "same image set"),
            ("implementation", "other", "metric protocols differ"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = two_candidates(root, schema=2, protocol=canonical_protocol(), dataset_content_sha256="3"*64)
                summary = json.loads(args.best_summary.read_text())
                if field == "pixel_domain":
                    summary["metric_protocol"][field] = value
                elif field == "implementation":
                    summary["metric_protocol"]["ms_ssim_rgb"][field] = value
                else:
                    summary[field] = value
                args.best_summary.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError, expected):
                    GATE.select_lic(args)

    def test_v2_requires_protocol_content_hash_and_consistent_alias(self):
        for failure in ("metric_protocol", "dataset_content_sha256", "ms_ssim_alias"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = two_candidates(root, schema=2, protocol=canonical_protocol(), dataset_content_sha256="3"*64)
                summary = json.loads(args.last_summary.read_text())
                if failure == "ms_ssim_alias":
                    summary["qualities"]["q0"]["ms_ssim_rgb"] += .02
                else:
                    del summary[failure]
                args.last_summary.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError, "metric_protocol|dataset_content_sha256|disagrees"):
                    GATE.select_lic(args)

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
            args = frozen_codec_evaluations(root)
            GATE.verify_p0(args)
            report = json.loads(args.output.read_text())
            self.assertTrue(report["accepted"])
            self.assertEqual(report["metric_protocol"], GATE.LEGACY_METRIC_PROTOCOL)

    def test_gate_b_accepts_matching_v2_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            args = frozen_codec_evaluations(Path(directory), modern=True)
            GATE.verify_p0(args)
            report = json.loads(args.output.read_text())
            self.assertTrue(report["accepted"])
            self.assertEqual(report["metric_protocol"], canonical_protocol())

    def test_gate_b_rejects_two_empty_bitstream_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            args = frozen_codec_evaluations(Path(directory))
            (args.lic_eval_dir/"q0/bin/image.bin").unlink()
            (args.p0_root/"q0/bin/image").unlink()
            with self.assertRaisesRegex(ValueError, "no bitstream files"):
                GATE.verify_p0(args)
            self.assertFalse(args.output.exists())

    def test_gate_b_rejects_identical_partially_deleted_bitstream_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            args = frozen_codec_evaluations(Path(directory), modern=True)
            summary = json.loads(args.lic_summary.read_text())
            summary["num_images"] = 2
            args.lic_summary.write_text(json.dumps(summary))
            for quality in GATE.QUALITY_NAMES:
                (args.lic_eval_dir/quality/"bin/second.bin").write_bytes(b"second packet")
                (args.p0_root/quality/"bin/second").write_bytes(b"second packet")
            (args.lic_eval_dir/"q0/bin/second.bin").unlink()
            (args.p0_root/"q0/bin/second").unlink()
            with self.assertRaisesRegex(ValueError, "file count.*is 1, but num_images is 2"):
                GATE.verify_p0(args)
            self.assertFalse(args.output.exists())

    def test_gate_b_checks_plain_report_image_count_when_available(self):
        with tempfile.TemporaryDirectory() as directory:
            args = frozen_codec_evaluations(Path(directory))
            path = args.p0_root/"q0/metrics/run/summary_metrics.json"
            summary = json.loads(path.read_text())
            summary["num_images"] = 2
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "file count.*is 1, but num_images is 2"):
                GATE.verify_p0(args)

    def test_gate_b_preserves_nonempty_legacy_reports_without_image_count(self):
        with tempfile.TemporaryDirectory() as directory:
            args = frozen_codec_evaluations(Path(directory))
            summary = json.loads(args.lic_summary.read_text())
            del summary["num_images"]
            args.lic_summary.write_text(json.dumps(summary))
            GATE.verify_p0(args)
            report = json.loads(args.output.read_text())
            self.assertTrue(report["accepted"])
            self.assertEqual(report["qualities"]["q0"]["bitstream_files"], 1)

    def test_gate_b_rejects_changed_protocol_missing_metadata_and_image_contents(self):
        for failure in ("pixel_domain", "legacy", "dataset_content_sha256", "ms_ssim_rgb", "bitstream"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                args = frozen_codec_evaluations(Path(directory), modern=True)
                path = args.p0_root/"q0/metrics/run/summary_metrics.json"
                summary = json.loads(path.read_text())
                if failure == "pixel_domain":
                    summary["metric_protocol"]["pixel_domain"] = "uint8_roundtrip"
                elif failure == "legacy":
                    del summary["metric_protocol"]
                elif failure == "dataset_content_sha256":
                    summary["dataset_content_sha256"] = "4"*64
                elif failure == "ms_ssim_rgb":
                    summary["ms_ssim_rgb"] = .5
                else:
                    (args.p0_root/"q0/bin/image").write_bytes(b"changed packet")
                path.write_text(json.dumps(summary))
                with self.assertRaises((ValueError, SystemExit)):
                    GATE.verify_p0(args)

    def test_gate_b_still_rejects_different_bytes_after_suffix_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lic_bin = root / "lic"
            p0_bin = root / "p0"
            lic_bin.mkdir()
            p0_bin.mkdir()
            (lic_bin / "image.bin").write_bytes(b"source packet")
            (p0_bin / "image").write_bytes(b"changed packet")

            self.assertNotEqual(
                GATE.collect_files(lic_bin),
                GATE.collect_files(p0_bin),
            )


if __name__ == "__main__":
    unittest.main()
