"""Paid, measured block distortion; not a Gaussian posterior or RF time."""

import torch
from torch import nn
from torch.nn import functional as F


class DistortionScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("s_data", torch.tensor(1.0))
        self.register_buffer("log_min", torch.tensor(-7.0))
        self.register_buffer("log_max", torch.tensor(2.0))
        self.register_buffer("calibrated", torch.tensor(False))

    @torch.no_grad()
    def measure(self, clean, coarse):
        squared = (clean.float() - coarse.float()).square().mean(1, keepdim=True)
        return F.avg_pool2d(squared, 2, 2).clamp_min((self.s_data * 1e-4).square()).sqrt()

    @torch.no_grad()
    def calibrate(self, clean_second_moment, block_scales):
        self.s_data.copy_(torch.as_tensor(clean_second_moment, device=self.s_data.device).sqrt().clamp_min(1e-8))
        values = torch.cat([v.detach().flatten().to(self.s_data.device) for v in block_scales])
        logs = values.clamp_min(self.s_data * 1e-4).log()
        lo, hi = torch.quantile(logs, torch.tensor([.001, .999], device=logs.device))
        spacing = ((hi-lo) / 13).clamp_min(.05)
        self.log_min.copy_(lo-spacing)
        self.log_max.copy_(hi+spacing)
        self.calibrated.fill_(True)

    def dequantize(self, codes):
        return (self.log_min + codes.float() / 15 * (self.log_max-self.log_min)).exp()

    @torch.no_grad()
    def forward(self, clean, coarse, require_calibration=True):
        if require_calibration and not self.calibrated.item():
            raise ValueError("calibrate distortion scales on the common training prefix first")
        measured = self.measure(clean, coarse)
        logs = measured.log()
        codes = ((logs-self.log_min) / (self.log_max-self.log_min) * 15).round().clamp(0, 15).to(torch.uint8)
        scale = self.dequantize(codes)
        log_rho = (scale / self.s_data).log().repeat_interleave(2, -2).repeat_interleave(2, -1)
        return {"measured": measured, "codes": codes, "scale": scale.repeat_interleave(2, -2).repeat_interleave(2, -1),
                "log_rho": log_rho.squeeze(1),
                "saturation": ((logs < self.log_min) | (logs > self.log_max)).float().mean()}

    def decode(self, codes):
        if not self.calibrated.item():
            raise ValueError("checkpoint has no calibrated distortion interface")
        scale = self.dequantize(codes).repeat_interleave(2, -2).repeat_interleave(2, -1)
        return scale, (scale / self.s_data).log().squeeze(1)
