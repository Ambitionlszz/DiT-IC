"""Validate directory evaluation inputs before loading metric networks."""

from dataclasses import dataclass
from pathlib import Path
import re

from PIL import Image


IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"})
_INDEX_PREFIX = re.compile(r"^\d{5,}_(.+)$")


@dataclass(frozen=True)
class ImagePair:
    reconstruction: Path
    ground_truth: Path | None
    size: tuple[int, int]


def _images_by_stem(directory, *, strip_index_prefix=False):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    paths = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported images found in {directory}")
    indexed = {}
    for path in paths:
        key = path.stem
        if strip_index_prefix:
            match = _INDEX_PREFIX.fullmatch(key)
            if match is None:
                raise ValueError(f"Expected an index prefix such as 00000_ in reconstruction name: {path.name}")
            key = match.group(1)
        if key in indexed:
            raise ValueError(f"Duplicate image stem {key!r}: {indexed[key]} and {path}")
        indexed[key] = path
    return indexed


def _image_size(path):
    try:
        with Image.open(path) as image:
            # Decode during preflight so corrupt files fail before model allocation.
            image = image.convert("RGB")
            image.load()
            return image.size
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read image {path}: {error}") from error


def collect_image_pairs(recon_dir, gt_dir=None, ntest=None, *, strip_recon_index_prefix=False):
    """Pair by exact, case-sensitive stem; extensions may differ.

    Directory membership is checked before applying ``ntest``. The explicit
    prefix option accepts the ``00000_source.png`` names from the BT evaluator.
    No filename transformation happens by default.
    """
    if ntest is not None and (not isinstance(ntest, int) or isinstance(ntest, bool) or ntest <= 0):
        raise ValueError("ntest must be a positive integer or None")
    reconstructions = _images_by_stem(recon_dir, strip_index_prefix=strip_recon_index_prefix)
    targets = _images_by_stem(gt_dir) if gt_dir is not None else None
    if targets is not None and reconstructions.keys() != targets.keys():
        missing = sorted(targets.keys() - reconstructions.keys())
        extra = sorted(reconstructions.keys() - targets.keys())
        raise ValueError(
            "Reconstruction and ground-truth stems differ: "
            f"missing reconstructions={missing}; unmatched reconstructions={extra}. "
            "Use --strip-recon-index-prefix explicitly for BT index-prefixed output names."
        )
    keys = sorted(reconstructions)
    if ntest is not None:
        keys = keys[:ntest]
    pairs = []
    for key in keys:
        reconstruction = reconstructions[key]
        target = targets[key] if targets is not None else None
        size = _image_size(reconstruction)
        if target is not None:
            target_size = _image_size(target)
            if size != target_size:
                raise ValueError(f"Image size mismatch for {key!r}: {reconstruction} has {size}, "
                                 f"but {target} has {target_size}")
        pairs.append(ImagePair(reconstruction, target, size))
    return pairs
