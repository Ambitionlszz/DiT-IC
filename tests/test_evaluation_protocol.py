"""Prevent cross-entry metric drift and ambiguous validation selection."""

from copy import deepcopy

import numpy as np
from PIL import Image
import pytest
import torch

from evaluation_protocol import (dataset_content_fingerprint, metric_protocol,
                                 psnr, quantize_image, save_image, structure_metrics)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_both_ssim_definitions_match_their_explicit_reference():
    from pyiqa.archs.ssim_arch import MS_SSIM
    from pytorch_msssim import ms_ssim

    generator = torch.Generator().manual_seed(903)
    target = torch.rand(1, 3, 192, 224, generator=generator)
    prediction = (target + .08 * torch.randn(target.shape, generator=generator)).clamp(0, 1)
    values = structure_metrics(prediction, target)
    expected_rgb = ms_ssim(prediction, target, data_range=1., size_average=True)
    expected_y = MS_SSIM(downsample=False, test_y_channel=True,
                         is_prod=True, color_space="yiq")(prediction, target).mean()
    torch.testing.assert_close(values["ms_ssim_rgb"], expected_rgb, rtol=0, atol=0)
    torch.testing.assert_close(values["ms_ssim_y"], expected_y, rtol=0, atol=0)
    assert values["ms_ssim"].item() == values["ms_ssim_rgb"].item()
    # Channel mixing changes the metric materially for RGB-independent noise.
    assert abs(values["ms_ssim_rgb"].item()-values["ms_ssim_y"].item()) > .001
    native = structure_metrics(prediction*2-1, target*2-1, input_range="minus_one_one")
    for key in values:
        torch.testing.assert_close(native[key], values[key], rtol=1e-6, atol=1e-7)


def test_small_or_invalid_images_are_handled_explicitly():
    image = torch.zeros(1, 3, 160, 200)
    assert structure_metrics(image, image) == {}
    with pytest.raises(ValueError, match="matching NCHW RGB"):
        structure_metrics(image, image[..., :-1])
    with pytest.raises(ValueError, match="input_range"):
        structure_metrics(image, image, input_range="unknown")
    with pytest.raises(ValueError, match="ms_ssim_alias"):
        structure_metrics(image, image, ms_ssim_alias="unknown")


def test_psnr_range_and_exact_match_ceiling_are_shared():
    target = torch.zeros(2, 3, 4, 5)
    prediction = torch.stack([torch.full((3, 4, 5), .1), torch.full((3, 4, 5), .01)])
    torch.testing.assert_close(psnr(prediction, target), torch.tensor(30.))
    torch.testing.assert_close(psnr(prediction*2-1, target*2-1, "minus_one_one"),
                               torch.tensor(30.), rtol=1e-6, atol=1e-5)
    assert psnr(target, target).item() == 120.
    assert psnr(target-1, target).item() == 120.  # Declared image range is clipped.


def test_png_pixels_and_explicit_uint8_metric_domain_are_identical(tmp_path):
    image = torch.tensor([-.1, 0., .5/255, 1.5/255, .501, 1., 1.1]).expand(1, 3, 2, -1)
    output = tmp_path/"round.png"
    save_image(image, output, input_range="zero_one")
    saved = torch.from_numpy(np.array(Image.open(output), copy=True)).permute(2, 0, 1).unsqueeze(0)
    expected = image.clamp(0, 1).mul(255).round().byte()
    assert torch.equal(saved, expected)
    assert torch.equal(quantize_image(image).mul(255).round().byte(), saved)
    native_output = tmp_path/"native.png"
    native = image*2-1
    save_image(native, native_output)
    native_saved = torch.from_numpy(np.array(Image.open(native_output), copy=True)).permute(2, 0, 1).unsqueeze(0)
    # Converting ranges in float32 can move an exact half-integer across its
    # tie. Assert each stored image matches the metrics on its actual inputs.
    assert torch.equal(native_saved, (native.clamp(-1, 1)+1).mul(127.5).round().byte())
    assert torch.equal(quantize_image(native, "minus_one_one").mul(255).round().byte(), native_saved)


def test_protocol_metadata_is_explicit_and_independent():
    floats = metric_protocol()
    assert floats["ms_ssim_alias"] == "ms_ssim_rgb"
    assert floats["ms_ssim_y"]["test_y_channel"] is True
    assert floats["packages"]["pytorch-msssim"] != "not-installed"
    assert len(floats["implementation_sha256"]) == 64
    uint8 = metric_protocol(pixel_domain="uint8_roundtrip")
    assert uint8 != floats
    floats["packages"]["torch"] = "tampered"
    assert metric_protocol()["packages"]["torch"] != "tampered"


def test_content_fingerprint_detects_same_path_and_size_changes(tmp_path):
    first = tmp_path/"first"/"image.png"
    second = tmp_path/"second"/"image.png"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"abc123")
    second.write_bytes(first.read_bytes())
    original = dataset_content_fingerprint([first])
    assert dataset_content_fingerprint([second]) == original
    first.write_bytes(b"abc456")
    assert dataset_content_fingerprint([first]) != original


def test_validation_identity_invalidates_previous_metric_or_pixel_scores(tmp_path):
    from models.bt_self_flow.checkpoint import BestCheckpoint
    from models.bt_self_flow.data import ImageDataset
    from models.bt_self_flow.validation import build_validation_identity

    path = tmp_path/"image.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)
    dataset = ImageDataset([tmp_path], crop=0, training=False)
    identity = build_validation_identity(dataset, actual_rate=True, config={"image_loss": {"mse_weight": 4}})
    scores = {str(q): {"loss": 1.} for q in (0., .2, .4, .6, .8, 1.)}
    best = BestCheckpoint(tmp_path, identity=identity)
    assert best.consider(scores, 2)
    changed_protocol = deepcopy(identity)
    changed_protocol["metric_protocol"] = metric_protocol(pixel_domain="uint8_roundtrip")
    assert BestCheckpoint(tmp_path, restored=best.state, identity=changed_protocol).state is None
    Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)).save(path)
    changed_pixels = build_validation_identity(dataset, actual_rate=True, config={"image_loss": {"mse_weight": 4}})
    assert changed_pixels["dataset_fingerprint"] == identity["dataset_fingerprint"]
    assert BestCheckpoint(tmp_path, restored=best.state, identity=changed_pixels).state is None


def test_bt_validation_reports_rgb_alias_and_explicit_y():
    from models.bt_self_flow.losses import ImageLoss
    from models.bt_self_flow.validation import validate
    from test_bt_backend import tiny_model

    generator = torch.Generator().manual_seed(905)
    dataset = [torch.rand(3, 256, 256, generator=generator)*2-1]
    model = tiny_model("C").eval()
    report = validate(model, dataset, ImageLoss(lpips_weight=0, dists_weight=0), qualities=(.4,))
    assert report["0.4"]["ms_ssim"] == report["0.4"]["ms_ssim_rgb"]
    assert -1 <= report["0.4"]["ms_ssim_y"] <= 1


def test_validation_ssim_mean_counts_only_eligible_images():
    from models.bt_self_flow.losses import ImageLoss
    from models.bt_self_flow.validation import validate
    from test_bt_backend import tiny_model

    generator = torch.Generator().manual_seed(906)
    large = torch.rand(3, 256, 256, generator=generator)*2-1
    small = large[:, :128, :128]
    model = tiny_model("C").eval()
    objective = ImageLoss(lpips_weight=0, dists_weight=0)
    expected = validate(model, [large], objective, qualities=(.4,))["0.4"]
    actual = validate(model, [large, small], objective, qualities=(.4,))["0.4"]
    for metric in ("ms_ssim", "ms_ssim_rgb", "ms_ssim_y"):
        assert actual[metric] == pytest.approx(expected[metric], rel=1e-7)
