"""Shared, explicit image-quality protocol for training validation and evaluation.

``ms_ssim`` always aliases ``ms_ssim_rgb`` in protocol v2. Historical pyiqa
Y-channel scores remain available as ``ms_ssim_y``; old reports are not migrated.
No pretrained weights are constructed by this module.
"""

from copy import deepcopy
from functools import lru_cache
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

import torch


PROTOCOL_VERSION = "image_quality_rgb_y_v2"
PADDING_PROTOCOL = "right/bottom to 256; reflect when possible else replicate; crop to original dimensions"


def _zero_one(image, input_range):
    if input_range not in ("zero_one", "minus_one_one"):
        raise ValueError("input_range must be zero_one or minus_one_one")
    image = image.float()
    if input_range == "minus_one_one":
        image = (image + 1) * .5
    return image.clamp(0, 1)


def quantize_image(image, input_range="zero_one"):
    """Return the exact [0,1] pixels obtained by rounded 8-bit PNG storage."""
    return _zero_one(image, input_range).mul(255).round().div(255)


def psnr(prediction, target, input_range="zero_one"):
    """Mean per-image RGB PSNR, with a common 120 dB exact-match ceiling."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("PSNR requires matching NCHW RGB tensors")
    if prediction.device != target.device or prediction.numel() == 0:
        raise ValueError("PSNR requires nonempty images on the same device")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        difference = _zero_one(prediction, input_range) - _zero_one(target, input_range)
        mse = difference.square().flatten(1).mean(1)
        return (-10 * torch.log10(mse.clamp_min(1e-12))).mean()


def save_image(image, path, input_range="minus_one_one"):
    """Save one RGB image using the same rounding as ``quantize_image``."""
    from PIL import Image

    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("save_image requires a single RGB CHW image")
    pixels = quantize_image(image.detach(), input_range).mul(255).round().byte()
    pixels = pixels.permute(1, 2, 0).cpu().numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


@lru_cache(maxsize=8)
def _y_metric(device_name):
    # Instantiate the architecture directly: no device sentinel in an IQA
    # wrapper, no pretrained weights, and no implicit default color protocol.
    from pyiqa.archs.ssim_arch import MS_SSIM

    return MS_SSIM(downsample=False, test_y_channel=True, is_prod=True,
                   color_space="yiq").eval().to(torch.device(device_name))


def structure_metrics(prediction, target, input_range="zero_one", ms_ssim_alias="rgb"):
    """Compute both structure scores on identical RGB pixels.

    RGB preserves the established BT float32/pytorch-msssim definition. Y
    explicitly preserves the legacy pyiqa YIQ/255-rounding/float64 definition.
    Images too small for the five-scale 11-pixel window return no SSIM fields.
    """
    if ms_ssim_alias not in ("rgb", "y"):
        raise ValueError("ms_ssim_alias must be rgb or y")
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("structure metrics require matching NCHW RGB tensors")
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device")
    if prediction.shape[0] < 1 or min(prediction.shape[-2:]) < 1:
        raise ValueError("structure metrics require nonempty images")
    # Validate the declared range even when SSIM is unavailable for this size.
    if input_range not in ("zero_one", "minus_one_one"):
        raise ValueError("input_range must be zero_one or minus_one_one")
    if min(prediction.shape[-2:]) <= 160:
        return {}
    from pytorch_msssim import ms_ssim

    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = _zero_one(prediction, input_range)
        target = _zero_one(target, input_range)
        rgb = ms_ssim(prediction, target, data_range=1., size_average=True,
                      win_size=11, win_sigma=1.5, K=(.01, .03))
        y = _y_metric(str(prediction.device))(prediction, target).mean()
    return {"ms_ssim": rgb if ms_ssim_alias == "rgb" else y,
            "ms_ssim_rgb": rgb, "ms_ssim_y": y}


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


@lru_cache(maxsize=4)
def _protocol(pixel_domain, ms_ssim_alias):
    if pixel_domain not in ("float", "uint8_roundtrip"):
        raise ValueError("pixel_domain must be float or uint8_roundtrip")
    if ms_ssim_alias not in ("rgb", "y"):
        raise ValueError("ms_ssim_alias must be rgb or y")
    return {
        "version": PROTOCOL_VERSION,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pixel_domain": pixel_domain,
        "color": "RGB",
        "pixel_range": [0, 1],
        "quantization": "none" if pixel_domain == "float" else "round(255*x)/255; ties to even",
        "aggregation": "arithmetic mean of per-image scores; no spatial resize",
        "psnr": {"color": "RGB", "data_range": 1, "precision": "float32",
                 "mse_floor": 1e-12, "exact_match_db": 120,
                 "equivalent_minus_one_one_range": 2},
        "lpips": {"network": "alex", "version": "0.1", "input_range": [-1, 1],
                  "precision": "float32", "mode": "eval"},
        "dists": {"implementation": "pyiqa", "input_range": [0, 1], "precision": "float32"},
        "ms_ssim_alias": "ms_ssim_" + ms_ssim_alias,
        "ms_ssim_rgb": {"implementation": "pytorch_msssim", "data_range": 1,
                        "precision": "float32", "win_size": 11, "win_sigma": 1.5,
                        "weights": [.0448, .2856, .3001, .2363, .1333], "K": [.01, .03]},
        "ms_ssim_y": {"implementation": "pyiqa.MS_SSIM", "test_y_channel": True,
                      "color_space": "yiq", "downsample": False, "is_prod": True,
                      "data_range": 255, "round_internal_pixels": True, "precision": "float64"},
        "ms_ssim_min_side_exclusive": 160,
        "packages": {name: _package_version(name) for name in
                     ("torch", "torchvision", "lpips", "pyiqa", "pytorch-msssim", "torchmetrics", "Pillow")},
    }


def metric_protocol(pixel_domain="float", ms_ssim_alias="rgb"):
    """Return independent JSON-safe protocol metadata, including package versions."""
    return deepcopy(_protocol(pixel_domain, ms_ssim_alias))


def dataset_content_fingerprint(paths):
    """Hash the ordered image contents and stems, independently of the host root.

    This deliberately hashes bytes (including image metadata), rather than just
    paths/file sizes. Use only for evaluation sets, not large training datasets.
    """
    digest = hashlib.sha256(b"ordered-image-bytes-and-stems-v1\0")
    for index, path in enumerate(paths):
        path = Path(path)
        content = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024*1024), b""):
                content.update(chunk)
        digest.update(str(index).encode() + b"\0" + path.stem.encode() + b"\0" + content.digest())
    return digest.hexdigest()


def source_revision():
    """Describe the current source checkout without modifying Git state."""
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}
