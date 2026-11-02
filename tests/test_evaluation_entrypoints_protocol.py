"""Exercise evaluation pixel domains without pretrained assets or a GPU."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from unittest.mock import patch

from codec_packet import CodecPacket
from eval.testing_utils import parse_args_testing
from scripts import eval_variable_rate_lic_direct as direct


def test_compress_metric_domain_argument_preserves_float_default():
    assert parse_args_testing([]).metric_domain == "float"
    assert parse_args_testing(["--metric-domain", "uint8"]).metric_domain == "uint8"


def test_direct_lic_small_input_uses_replicate_padding_without_altering_source(tmp_path):
    pixels = np.random.default_rng(974).integers(0, 256, (5, 9, 3), dtype=np.uint8)
    source = tmp_path / "small.png"
    Image.fromarray(pixels).save(source)
    reference, padded = direct.load_image(source, torch.device("cpu"))
    assert tuple(padded.shape) == (1, 3, 256, 256)
    torch.testing.assert_close(padded[..., :5, :9], reference)
    torch.testing.assert_close(padded[..., -1, -1], reference[..., -1, -1])


def test_direct_lic_float_and_png_domains_match_report_and_saved_pixels(tmp_path):
    torch.set_num_threads(1)
    pixels = np.random.default_rng(971).integers(0, 256, (161, 165, 3), dtype=np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(pixels).save(source)

    class Codec:
        def compress(self, latent, quality_index):
            return CodecPacket([[b"test stream"]], (8, 8), quality_index)

        def decompress(self, packet):
            assert packet.strings == [[b"test stream"]]
            return torch.full((1, 3, 256, 256), .104)

    stage = SimpleNamespace(
        codec=Codec(), encode_clean_latent=lambda image: image,
        vae=SimpleNamespace(config=SimpleNamespace(scaling_factor=1.),
                            decode=lambda latent, **kwargs: (latent,)),
    )
    metrics = {name: lambda prediction, target: (prediction-target).square().mean()
               for name in ("lpips", "dists")}
    for index, domain in enumerate(("float", "uint8")):
        with patch.object(torch.cuda, "synchronize"):
            report = direct.evaluate_quality(stage, [source], tmp_path, index,
                                             metrics, torch.device("cpu"), domain)
        saved = np.asarray(Image.open(tmp_path/f"q{index}/rec/source.png"))
        assert np.all(saved == 141)  # .552*255 = 140.76: PNG conversion rounds.
        reconstruction = saved.astype(np.float64)/255 if domain == "uint8" else np.full(pixels.shape, .552)
        mse = np.mean((reconstruction-pixels.astype(np.float64)/255)**2)
        result = report["per_image"][0]
        assert abs(result["psnr"] + 10*np.log10(mse)) < 1e-5
        assert abs(result["lpips"] - 4*mse) < 1e-6
        assert abs(result["dists"] - mse) < 1e-6
        assert result["ms_ssim"] == result["ms_ssim_rgb"]
        assert "ms_ssim_y" in result
        assert result["metric_domain"] == domain
        assert report["metric_protocol"]["pixel_domain"] == ("uint8_roundtrip" if domain == "uint8" else "float")
        assert report["averages"]["ms_ssim_y"] == result["ms_ssim_y"]
        assert report == json.loads((tmp_path/f"q{index}/metrics.json").read_text())
