import bisect
import math

import torch
from torch import nn
from torch.nn import functional as F


def rate_beta(control, quality=None):
    if quality is None:
        return 8 * 32**(-control.value)
    values, weights = quality["values"], quality["rate_coefficients"]
    index = min(max(bisect.bisect_right(values, control.value)-1, 0), len(values)-2)
    fraction = (control.value-values[index])/(values[index+1]-values[index])
    return math.exp((1-fraction)*math.log(weights[index])+fraction*math.log(weights[index+1]))


def estimated_bpp(result, image):
    b, _, h, w = image.shape
    estimated_bits = sum(-value.float().log2().sum() for value in result.likelihoods.values())
    return estimated_bits/(b*h*w) + result.side_bits/(h*w)


class ImageLoss(nn.Module):
    def __init__(self, lpips_weight=1., dists_weight=.5, mse_weight=2.):
        super().__init__()
        self.lpips_weight, self.dists_weight, self.mse_weight = lpips_weight, dists_weight, mse_weight
        self.lpips = self.dists = None
        if lpips_weight:
            import lpips
            self.lpips = lpips.LPIPS(net="alex").requires_grad_(False).eval()
        if dists_weight:
            import pyiqa
            # pyiqa 0.1.16's wrapper keeps a device sentinel outside registered
            # state after its initial CUDA move. Use the differentiable network
            # directly so this loss follows normal nn.Module.to() semantics.
            self.dists = pyiqa.create_metric("dists", as_loss=True, device="cpu").net.requires_grad_(False).eval()

    def forward(self, predicted, target):
        # Perceptual statistics and losses are evaluated in FP32, with image
        # gradients retained through the frozen perceptual networks.
        with torch.autocast(device_type=predicted.device.type, enabled=False):
            predicted, target = predicted.float(), target.float()
            mse = F.mse_loss(predicted, target)
            lpips = self.lpips(predicted, target).mean() if self.lpips is not None else mse*0
            dists = (self.dists((predicted+1).mul(.5).clamp(0, 1), (target+1).mul(.5).clamp(0, 1)).mean()
                     if self.dists is not None else mse*0)
            return self.mse_weight*mse+self.lpips_weight*lpips+self.dists_weight*dists, {
                "mse": mse, "lpips": lpips, "dists": dists,
                "psnr": -10*torch.log10(mse.clamp_min(1e-12)/4),
            }


def reconstruction_objective(result, image, control, s_data, image_loss=None, objective=None, quality=None):
    objective = objective or {}
    base = F.mse_loss(result.coarse.float(), result.clean.float())/s_data.square()
    rate = estimated_bpp(result, image)
    weighted_rate = rate_beta(control, quality)*rate
    weighted_base = objective.get("base_weight", 1. if image_loss is None else .1)*base
    if image_loss is None:
        return weighted_base+weighted_rate, {"base": base, "rate_est": rate,
                                            "weighted_base": weighted_base, "weighted_rate": weighted_rate}
    repaired_weight = objective.get("repaired_weight", .1)
    repaired = result.latent.float() if repaired_weight else result.latent.detach().float()
    latent = F.mse_loss(repaired, result.clean.float())/s_data.square()
    distortion, metrics = image_loss(result.image, image)
    weighted_latent = repaired_weight*latent
    loss = distortion+weighted_latent+weighted_base+weighted_rate
    return loss, dict(metrics, base=base, latent=latent, rate_est=rate, weighted_base=weighted_base,
                      weighted_repaired=weighted_latent, weighted_rate=weighted_rate,
                      weighted_mse=image_loss.mse_weight*metrics["mse"],
                      weighted_lpips=image_loss.lpips_weight*metrics["lpips"],
                      weighted_dists=image_loss.dists_weight*metrics["dists"])
