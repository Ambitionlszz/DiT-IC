import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_correction_ownership.py"
SPEC = importlib.util.spec_from_file_location("correction_ownership_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def row(offset, hard_is_closer=True):
    vlm_correct = 0.20 + offset
    vlm_hard = 0.24 + 2 * offset
    vlm_random = (0.30 if hard_is_closer else 0.22) + 2 * offset
    return {
        "real_bpp": 0.05,
        "vlm_correct": vlm_correct,
        "vlm_hard": vlm_hard,
        "vlm_random": vlm_random,
        "lpips_base": 0.30,
        "lpips_correct": 0.20 + offset,
        "lpips_hard": 0.24 + 2 * offset,
        "dists_base": 0.22,
        "dists_correct": 0.12 + offset,
        "dists_hard": 0.16 + 2 * offset,
    }


class CorrectionOwnershipProbeTest(unittest.TestCase):
    def test_derangement_has_no_fixed_positions(self):
        values = PROBE.derangement_indices(8, seed=903)
        self.assertEqual(sorted(values), list(range(8)))
        self.assertTrue(all(index != value for index, value in enumerate(values)))

    def test_promising_summary_requires_aligned_hard_negatives(self):
        rows = [row(index / 1000.0) for index in range(64)]
        summary = PROBE.summarize_rows(rows, min_images=64)
        self.assertEqual(summary["verdict"], "promising")
        self.assertGreaterEqual(
            summary["hard_is_closer_than_random_vlm_rate"],
            0.60,
        )
        self.assertGreaterEqual(summary["spearman_vlm_gap_vs_lpips_gap"], 0.20)
        self.assertGreaterEqual(summary["spearman_vlm_gap_vs_dists_gap"], 0.20)

    def test_small_sample_does_not_claim_viability(self):
        summary = PROBE.summarize_rows([row(0.0)] * 8, min_images=64)
        self.assertEqual(summary["verdict"], "insufficient_sample")


if __name__ == "__main__":
    unittest.main()
