import tempfile
import unittest
from pathlib import Path

from scripts.compare_bitstreams import compare_directories


class CompareBitstreamsTest(unittest.TestCase):
    def test_identical_directories_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline"
            gs_residual = root / "gs_residual"
            baseline.mkdir()
            gs_residual.mkdir()
            (baseline / "kodim01.bin").write_bytes(b"same")
            (gs_residual / "kodim01.bin").write_bytes(b"same")

            self.assertEqual(
                compare_directories(baseline, gs_residual),
                ([], [], []),
            )

    def test_content_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline"
            gs_residual = root / "gs_residual"
            baseline.mkdir()
            gs_residual.mkdir()
            (baseline / "kodim01.bin").write_bytes(b"baseline")
            (gs_residual / "kodim01.bin").write_bytes(b"gs")

            _, _, mismatched = compare_directories(baseline, gs_residual)
            self.assertEqual(mismatched, [Path("kodim01.bin")])


if __name__ == "__main__":
    unittest.main()
