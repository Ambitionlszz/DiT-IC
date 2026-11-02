from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import torch

from eval import evaluate as evaluator
from eval.image_pairs import collect_image_pairs


def write_image(path, value=0, size=(8, 6)):
    Image.new("RGB", size, color=(value, value, value)).save(path)


class DirectoryPairingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.recon, self.gt = self.root / "recon", self.root / "gt"
        self.recon.mkdir()
        self.gt.mkdir()

    def test_stem_pairing_is_deterministic_and_allows_changed_extension(self):
        for name in ("b", "a"):
            write_image(self.recon / f"{name}.png")
            write_image(self.gt / f"{name}.JPG")
        pairs = collect_image_pairs(self.recon, self.gt)
        self.assertEqual([pair.reconstruction.stem for pair in pairs], ["a", "b"])
        self.assertEqual([pair.ground_truth.stem for pair in pairs], ["a", "b"])
        self.assertEqual(pairs[0].size, (8, 6))

    def test_equal_length_different_names_cannot_be_paired_positionally(self):
        write_image(self.recon / "a.png")
        write_image(self.gt / "b.png")
        with self.assertRaisesRegex(ValueError, "missing reconstructions=.*b.*unmatched reconstructions=.*a"):
            collect_image_pairs(self.recon, self.gt)

    def test_limit_does_not_hide_dataset_membership_mismatch(self):
        for name in ("a", "b"):
            write_image(self.recon / f"{name}.png")
        write_image(self.gt / "a.png")
        with self.assertRaisesRegex(ValueError, "stems differ"):
            collect_image_pairs(self.recon, self.gt, ntest=1)

    def test_bt_index_prefix_requires_explicit_option(self):
        write_image(self.recon / "00000_kodim01.png")
        write_image(self.gt / "kodim01.png")
        with self.assertRaisesRegex(ValueError, "stems differ"):
            collect_image_pairs(self.recon, self.gt)
        pairs = collect_image_pairs(self.recon, self.gt, strip_recon_index_prefix=True)
        self.assertEqual(pairs[0].ground_truth.name, "kodim01.png")
        (self.recon / "00000_kodim01.png").rename(self.recon / "kodim01.png")
        with self.assertRaisesRegex(ValueError, "Expected an index prefix"):
            collect_image_pairs(self.recon, self.gt, strip_recon_index_prefix=True)

    def test_duplicate_stems_and_duplicate_prefix_targets_are_rejected(self):
        write_image(self.recon / "a.png")
        write_image(self.recon / "a.jpg")
        with self.assertRaisesRegex(ValueError, "Duplicate image stem"):
            collect_image_pairs(self.recon)
        for path in self.recon.iterdir():
            path.unlink()
        write_image(self.recon / "00000_a.png")
        write_image(self.recon / "00001_a.png")
        with self.assertRaisesRegex(ValueError, "Duplicate image stem"):
            collect_image_pairs(self.recon, strip_recon_index_prefix=True)

    def test_empty_missing_and_corrupt_directories_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "No supported images"):
            collect_image_pairs(self.recon)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            collect_image_pairs(self.root / "missing")
        (self.recon / "broken.png").write_bytes(b"invalid image")
        with self.assertRaisesRegex(ValueError, "Cannot read image.*broken.png"):
            collect_image_pairs(self.recon)

    def test_shape_mismatch_fails_before_allocating_metrics(self):
        write_image(self.recon / "a.png")
        write_image(self.gt / "a.png", size=(6, 8))
        with patch.object(evaluator, "_build_metrics") as build:
            with self.assertRaisesRegex(ValueError, "Image size mismatch"):
                evaluator.evaluate(self.recon, self.gt, device="cpu")
            build.assert_not_called()

    def test_paired_patch_size_requirement_fails_before_allocation(self):
        write_image(self.recon / "a.png")
        write_image(self.gt / "a.png")
        with patch.object(evaluator, "_build_metrics") as build:
            with self.assertRaisesRegex(ValueError, "FID/256.*at least 256"):
                evaluator.evaluate(self.recon, self.gt, device="cpu")
            build.assert_not_called()

    def test_limit_must_be_positive_and_limits_validated_dataset(self):
        for name in ("a", "b"):
            write_image(self.recon / f"{name}.png")
        for limit in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                collect_image_pairs(self.recon, ntest=limit)
        self.assertEqual(len(collect_image_pairs(self.recon, ntest=1)), 1)


class DirectoryEvaluatorTest(unittest.TestCase):
    def test_unpaired_evaluation_does_not_access_fid_and_writes_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            write_image(images / "a.png", value=255)
            output = root / "report" / "metrics.json"
            suite = ({"musiq": lambda image: image.mean()}, {}, None, None, None)
            with patch.object(evaluator, "_build_metrics", return_value=suite) as build, \
                    patch.object(evaluator, "structure_metrics") as structure, \
                    redirect_stdout(io.StringIO()):
                result = evaluator.evaluate(images, output_json=output, device="cpu")
            self.assertEqual(result, ["musiq: 1.00000"])
            build.assert_called_once_with(torch.device("cpu"), False)
            structure.assert_not_called()
            report = json.loads(output.read_text())
            self.assertEqual(report["metrics"], {"musiq": 1.0})
            self.assertEqual(report["num_images"], 1)
            self.assertEqual(report["metric_protocol"]["pairing"], "exact_case_sensitive_stem")
            self.assertIsNone(report["images"][0]["ground_truth"])
            self.assertEqual(len(report["dataset_fingerprints"]["reconstruction"]), 64)
            self.assertIsNone(report["dataset_fingerprints"]["ground_truth"])

    def test_paired_rgb_y_outputs_means_and_distribution_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recon, gt = root / "recon", root / "gt"
            recon.mkdir()
            gt.mkdir()
            for name, value in (("a", 0), ("b", 255)):
                write_image(recon / f"{name}.png", value=value, size=(256, 256))
                write_image(gt / f"{name}.png", value=value, size=(256, 256))
            fid = Mock()
            fid.compute.return_value = torch.tensor(17.0)
            kid = Mock()
            kid.compute.return_value = (torch.tensor(.2), torch.tensor(.03))
            patch_fid = Mock(return_value=30)
            def structure(prediction, target):
                self.assertEqual(prediction.dtype, torch.float32)
                self.assertTrue(torch.equal(prediction, target))
                mean = prediction.mean()
                return {"ms_ssim": mean, "ms_ssim_rgb": mean, "ms_ssim_y": mean + .1}
            suite = ({}, {"psnr": lambda pred, target: (pred-target).square().mean()}, fid, kid, patch_fid)
            output = root / "metrics.json"
            with patch.object(evaluator, "_build_metrics", return_value=suite), \
                    patch.object(evaluator, "structure_metrics", side_effect=structure), \
                    redirect_stdout(io.StringIO()):
                result = evaluator.evaluate(recon, gt, output_json=output, device="cpu")
            report = json.loads(output.read_text())
            self.assertIn("ms_ssim: 0.50000", result)
            self.assertEqual(report["metrics"]["ms_ssim_rgb"], .5)
            self.assertAlmostEqual(report["metrics"]["ms_ssim_y"], .6, places=6)
            self.assertEqual(report["metrics"]["fid"], 17.0)
            self.assertEqual(report["metric_protocol"]["fid"]["patch_count"], 60)
            kid.compute.assert_called_once()
            self.assertEqual(patch_fid.call_count, 2)
            self.assertEqual([Path(row["ground_truth"]).name for row in report["images"]], ["a.png", "b.png"])

    def test_insufficient_patch_metrics_are_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_image(root / "a.png", size=(256, 256))
            fid, kid = Mock(), Mock()
            suite = ({}, {}, fid, kid, Mock(return_value=1))
            output = root / "metrics.json"
            with patch.object(evaluator, "_build_metrics", return_value=suite), \
                    patch.object(evaluator, "structure_metrics", return_value={}), \
                    redirect_stdout(io.StringIO()):
                evaluator.evaluate(root, root, output_json=output, device="cpu")
            unavailable = json.loads(output.read_text())["metric_protocol"]["unavailable_metrics"]
            self.assertEqual(set(unavailable), {"fid", "kid_mean", "kid_std"})
            fid.compute.assert_not_called()
            kid.compute.assert_not_called()

    def test_cli_preserves_old_directory_options(self):
        args = evaluator.parse_args(["--recon_dir", "rec", "--gt_dir", "gt",
                                     "--strip-recon-index-prefix", "--output-json", "metrics.json"])
        self.assertEqual((args.recon_dir, args.gt_dir), ("rec", "gt"))
        self.assertTrue(args.strip_recon_index_prefix)


if __name__ == "__main__":
    unittest.main()
