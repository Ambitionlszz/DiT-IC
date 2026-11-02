"""Shared sample requirements and metadata for patch FID/KID evaluation."""

FID_PATCH_SIZE = 256
KID_SUBSET_SIZE = 50
KID_SUBSETS = 100


def build_distribution_metrics(device):
    from torchmetrics.image import FrechetInceptionDistance, KernelInceptionDistance

    return (FrechetInceptionDistance().to(device),
            KernelInceptionDistance(subset_size=KID_SUBSET_SIZE, subsets=KID_SUBSETS).to(device))


def compute_distribution_metrics(fid_metric, kid_metric, patch_count):
    """Compute only metrics supported by the actual number of RGB patches."""
    values, unavailable = {}, {}
    if fid_metric is not None:
        if patch_count >= 2:
            values["fid"] = float(fid_metric.compute())
        else:
            unavailable["fid"] = "FID requires at least two real and two reconstructed patches."
    if kid_metric is not None:
        if patch_count >= KID_SUBSET_SIZE:
            mean, std = kid_metric.compute()
            values["kid_mean"], values["kid_std"] = float(mean), float(std)
        else:
            unavailable["kid_mean"] = unavailable["kid_std"] = (
                f"KID with subset_size={KID_SUBSET_SIZE} requires at least {KID_SUBSET_SIZE} patches; found {patch_count}.")
    return values, unavailable


def distribution_metric_protocol(patch_count):
    common = {
        "implementation": "torchmetrics",
        "feature": "inception_v3_2048",
        "pixel_domain": "round(255*x) RGB uint8",
        "patch_size": FID_PATCH_SIZE,
        "patch_count": patch_count,
        "patch_grid": "non-overlapping; also offset by 128 in both axes when both dimensions >=384",
    }
    return {"fid": {**common, "minimum_patches": 2},
            "kid": {**common, "subset_size": KID_SUBSET_SIZE, "subsets": KID_SUBSETS}}
