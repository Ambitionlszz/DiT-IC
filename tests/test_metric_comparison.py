from copy import deepcopy

import pytest

from evaluation_protocol import metric_protocol
from scripts.compare_bt_self_flow import compare


def summaries():
    reference = dict(dataset_fingerprint="paths", dataset_content_sha256="a"*64,
                     images=24, metric_protocol=metric_protocol(), weights="ema",
                     step=100, phase="joint", backend="C", checkpoint_sha256="b"*64,
                     summary={str(q): dict(bpp=rate, psnr=20+q, lpips=.3-.01*q,
                                           dists=.2-.01*q, ms_ssim=.7+.01*q,
                                           ms_ssim_rgb=.7+.01*q, ms_ssim_y=.8+.01*q)
                              for q, rate in enumerate((.01, .02, .04))})
    candidate = deepcopy(reference)
    candidate["backend"] = "D"
    return reference, candidate


def test_comparison_preserves_explicit_structure_metrics():
    reference, candidate = summaries()
    result = compare(reference, candidate)
    for row in result["rows"]:
        for metric in ("psnr", "lpips", "dists", "ms_ssim", "ms_ssim_rgb", "ms_ssim_y"):
            assert row[f"D_minus_C_{metric}"] == 0


@pytest.mark.parametrize("change", ["channel", "domain", "pixels"])
def test_comparison_rejects_different_measurement_contracts(change):
    reference, candidate = summaries()
    if change == "channel":
        candidate["metric_protocol"]["ms_ssim_alias"] = "ms_ssim_y"
    elif change == "domain":
        candidate["metric_protocol"] = metric_protocol(pixel_domain="uint8_roundtrip")
    else:
        candidate["dataset_content_sha256"] = "c"*64
    with pytest.raises(ValueError, match="identical"):
        compare(reference, candidate)


def test_comparison_rejects_partial_structure_reports():
    reference, candidate = summaries()
    del candidate["summary"]["0"]["ms_ssim_y"]
    with pytest.raises(ValueError, match="every quality point"):
        compare(reference, candidate)


@pytest.mark.parametrize("digest", [None, "invalid", "x"*64])
def test_modern_comparison_requires_valid_content_identity(digest):
    reference, candidate = summaries()
    for source in (reference, candidate):
        if digest is None:
            del source["dataset_content_sha256"]
        else:
            source["dataset_content_sha256"] = digest
    with pytest.raises(ValueError, match="dataset_content_sha256"):
        compare(reference, candidate)
