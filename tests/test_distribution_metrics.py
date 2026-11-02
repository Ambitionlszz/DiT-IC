"""Distribution metrics use patches as samples and never load test weights."""

from unittest.mock import Mock, patch

import torch
from torch import nn
from torchmetrics.image import KernelInceptionDistance

from eval.distribution_metrics import (
    build_distribution_metrics,
    compute_distribution_metrics,
    distribution_metric_protocol,
)
from eval._update_patch_fid import update_patch_fid


def test_build_uses_explicit_kid_subset_below_small_dataset_patch_count():
    with patch("torchmetrics.image.FrechetInceptionDistance") as fid, \
            patch("torchmetrics.image.KernelInceptionDistance") as kid:
        build_distribution_metrics(torch.device("cpu"))
    fid.assert_called_once_with()
    kid.assert_called_once_with(subset_size=50, subsets=100)


def test_distribution_scores_require_enough_patches():
    for count in (0, 1, 2, 49, 50):
        fid, kid = Mock(), Mock()
        fid.compute.return_value = torch.tensor(17.)
        kid.compute.return_value = (torch.tensor(.2), torch.tensor(.03))
        values, unavailable = compute_distribution_metrics(fid, kid, count)
        assert ("fid" in values) == (count >= 2)
        assert ("kid_mean" in values) == (count >= 50)
        assert fid.compute.call_count == int(count >= 2)
        assert kid.compute.call_count == int(count >= 50)
        assert ("kid_mean" in unavailable) == (count < 50)
        assert ("fid" in unavailable) == (count < 2)


def test_kodak_patches_enable_real_kid_even_with_fewer_than_fifty_images():
    torch.set_num_threads(1)
    collector = Mock()
    count = update_patch_fid(torch.zeros(1, 3, 512, 768), torch.full((1, 3, 512, 768), .552),
                             fid_metric=collector)
    assert count == 8  # six patches at origin plus two with the 128-pixel shift.
    fake_patches = collector.update.call_args_list[1].args[0]
    assert fake_patches.dtype == torch.uint8
    assert torch.all(fake_patches == 141)

    class CheapFeatures(nn.Module):
        def forward(self, images):
            return images.float().mean(dim=(-1, -2)) / 255

    # A 24-image Kodak set provides 192 patches, below torchmetrics' default
    # subset_size=1000 but above the declared and shared subset_size=50.
    patches = count * 24
    kid = KernelInceptionDistance(feature=CheapFeatures(), subset_size=50, subsets=100)
    generator = torch.Generator().manual_seed(981)
    kid.update(torch.randint(0, 256, (patches, 3, 2, 2), dtype=torch.uint8, generator=generator), real=True)
    kid.update(torch.randint(0, 256, (patches, 3, 2, 2), dtype=torch.uint8, generator=generator), real=False)
    values, unavailable = compute_distribution_metrics(None, kid, patches)
    assert not unavailable
    assert torch.isfinite(torch.tensor([values["kid_mean"], values["kid_std"]])).all()
    protocol = distribution_metric_protocol(patches)
    assert protocol["fid"]["patch_count"] == 192
    assert protocol["kid"]["subset_size"] == 50
