import torch
from torch import nn
from torch.nn import functional as F


def rate_beta(control):
    return 8 * 32**(-control.value)


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


def reconstruction_objective(result, image, control, s_data, image_loss=None):
    base = F.mse_loss(result.coarse.float(), result.clean.float())/s_data.square()
    rate = estimated_bpp(result, image)
    if image_loss is None:
        return base + rate_beta(control)*rate, {"base": base, "rate_est": rate}
    latent = F.mse_loss(result.latent.float(), result.clean.float())/s_data.square()
    distortion, metrics = image_loss(result.image, image)
    loss = distortion + .1*latent + .1*base + rate_beta(control)*rate
    return loss, dict(metrics, base=base, latent=latent, rate_est=rate)
