"""Explicit-step codec with one shared four-group quantization/coding path."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F
from compressai.ans import BufferedRansEncoder, RansDecoder

from models.latent_codec import LatentCodec, update_registered_buffers
from .packet import RateControl, pack_codes
from .router import RateRouter, allocation_from_codes, ste_round


@dataclass
class SourceAnalysis:
    clean: torch.Tensor
    y: torch.Tensor
    hyper_hat: torch.Tensor
    hyper_likelihood: torch.Tensor
    base: torch.Tensor
    allocation: torch.Tensor
    allocation_codes: torch.Tensor


class BTLatentCodec(LatentCodec):
    mechanism = "bt_explicit_local_step_v1"

    def __init__(self, ch_emd=32, channel=320, channel_out=32,
                 delta_coarse=4.0, delta_fine=0.5, kappa=math.log(2), **kwargs):
        kwargs.pop("variable_rate", None)
        super().__init__(ch_emd=ch_emd, channel=channel, channel_out=channel_out,
                         variable_rate=False, **kwargs)
        if not 0 < delta_fine < delta_coarse or kappa < 0:
            raise ValueError("invalid BT step endpoints or Router strength")
        self.register_buffer("log_delta_coarse", torch.tensor(math.log(delta_coarse)))
        self.register_buffer("log_delta_fine", torch.tensor(math.log(delta_fine)))
        self.register_buffer("kappa", torch.tensor(float(kappa)))
        self.register_buffer("router_strength", torch.tensor(1.0))
        self.router = RateRouter(ch_emd, channel)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # CDF lengths change after quantile updates and EMA evaluation.
        for name, buffers in (("entropy_bottleneck", ["_quantized_cdf", "_offset", "_cdf_length"]),
                              ("gaussian_conditional", ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"])):
            update_registered_buffers(getattr(self, name), name, buffers, state_dict, policy="resize")
        return torch.nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)

    def _apply(self, fn, recurse=True):
        self.masks.clear()
        return super()._apply(fn, recurse)

    def _hyper_forward(self, y):
        hyper = self.h_a(y)
        median = self.entropy_bottleneck._get_medians().reshape(1, -1, 1, 1)
        hyper_hat = median + ste_round(hyper - median)
        values = hyper_hat.permute(1, 0, 2, 3).reshape(hyper.shape[1], 1, -1)
        likelihood, _, _ = self.entropy_bottleneck._likelihood(values)
        likelihood = self.entropy_bottleneck.likelihood_lower_bound(likelihood)
        likelihood = likelihood.reshape(hyper.shape[1], hyper.shape[0], *hyper.shape[2:])
        return hyper_hat, likelihood.permute(1, 0, 2, 3).contiguous()

    def _prior(self, base, group):
        mean, raw_scale = self.adapter_out[group](self.g_c(self.adapter_in[group](base))).chunk(2, 1)
        return mean, F.softplus(raw_scale) + 1e-6

    def _mass(self, symbols, normalized_scale):
        gc = self.gaussian_conditional
        return gc.likelihood_lower_bound(gc._likelihood(symbols, normalized_scale))

    def analyze(self, clean):
        # The entire entropy/codec path has FP32 semantics, even inside AMP.
        with torch.autocast(device_type=clean.device.type, enabled=False):
            clean = clean.float()
            y = self.g_a(clean)
            if y.shape[-2] % 4 or y.shape[-1] % 4:
                raise ValueError("BT requires padded image dimensions divisible by 256")
            hyper_hat, hyper_likelihood = self._hyper_forward(y)
            base = self.h_s(hyper_hat)
            mean, scale = self._prior(base, 0)
            reference_step = ((self.log_delta_coarse + self.log_delta_fine) / 2).exp()
            reference_symbols = ((y - mean) / reference_step).round()
            cost = -self._mass(reference_symbols, scale / reference_step).log2().sum(1, keepdim=True)
            allocation, codes = self.router(clean, y, cost)
            return SourceAnalysis(clean, y, hyper_hat, hyper_likelihood, base, allocation, codes)

    def steps(self, control, allocation):
        b, _, h, w = allocation.shape
        quality = control.tensor(b, h, w, allocation.device)
        log_step = ((1-quality) * self.log_delta_coarse + quality * self.log_delta_fine
                    - self.kappa * self.router_strength * allocation)
        return log_step.exp()

    def _cdf(self):
        gc = self.gaussian_conditional
        return (gc.quantized_cdf.tolist(), gc.cdf_length.reshape(-1).int().tolist(),
                gc.offset.reshape(-1).int().tolist())

    def _groups(self, base, delta, y=None, mode="forward", stream=None, trace=False):
        """The same group order, units and LRP context in all three modes."""
        b, c, h, w = base.shape
        masks = self.get_mask_four_parts(b, c, h, w, device=base.device)
        restored = torch.zeros_like(base)
        likelihoods = torch.zeros_like(base)
        all_symbols, all_indexes, audit = [], [], []
        decoder = None
        if mode != "forward":
            cdf, lengths, offsets = self._cdf()
            if mode == "decode":
                decoder = RansDecoder()
                decoder.set_stream(stream)
        for group, mask in enumerate(masks):
            mean, scale = self._prior(base, group)
            mean_s = self.sequeeze_with_mask(mean, mask)
            scale_s = self.sequeeze_with_mask(scale / delta, mask)
            indexes = None
            if mode != "forward" or trace:
                indexes = self.gaussian_conditional.build_indexes(scale_s)
            if mode == "decode":
                flat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, lengths, offsets)
                symbols = torch.tensor(flat, device=base.device, dtype=base.dtype).reshape_as(mean_s)
            else:
                values = (self.sequeeze_with_mask(y, mask) - mean_s) / delta
                symbols = ste_round(values) if mode == "forward" else values.round()
                if mode == "encode":
                    all_symbols.extend(symbols.to(torch.int32).reshape(-1).tolist())
                    all_indexes.extend(indexes.reshape(-1).tolist())
            group_hat = self.unsequeeze_with_mask(mean_s + delta * symbols, mask)
            correction = 0.5 * torch.tanh(self.LRP[group](torch.cat([group_hat, base], 1))) * mask
            group_hat = group_hat + delta * correction
            restored = restored + group_hat
            if mode == "forward":
                mass = self._mass(symbols, scale_s)
                likelihoods = likelihoods + self.unsequeeze_with_mask(mass, mask)
            if trace:
                audit.append({"symbols": symbols.detach().to(torch.int32).cpu(),
                              "indexes": indexes.detach().cpu(),
                              "restored": group_hat.detach().cpu()})
            base = base * (1-mask) + group_hat
        encoded = None
        if mode == "encode":
            encoder = BufferedRansEncoder()
            encoder.encode_with_indexes(all_symbols, all_indexes, cdf, lengths, offsets)
            encoded = encoder.flush()
        return self.g_s(restored), likelihoods, encoded, audit

    def quantize_analysis(self, analysis, control, trace=False):
        with torch.autocast(device_type=analysis.y.device.type, enabled=False):
            delta = self.steps(control, analysis.allocation)
            mean, likelihood, _, audit = self._groups(analysis.base, delta, analysis.y, trace=trace)
            return {"mean": mean, "likelihoods": {"y": likelihood, "h": analysis.hyper_likelihood},
                    "allocation_codes": analysis.allocation_codes, "delta": delta, "trace": audit}

    def forward(self, clean, rate_control=None, trace=False):
        control = rate_control or RateControl()
        return self.quantize_analysis(self.analyze(clean), control, trace=trace)

    @torch.no_grad()
    def compress(self, clean, rate_control=None, trace=False):
        if self.training or clean.shape[0] != 1:
            raise ValueError("real BT encoding requires eval mode and one image")
        if self.router_strength.item() != 1:
            raise ValueError("cannot deploy an unfinished Router warmup")
        control = rate_control or RateControl()
        with torch.autocast(device_type=clean.device.type, enabled=False):
            source = self.analyze(clean)
            # Always reconstruct the actual transmitted hyperprior first.
            hyper_stream = self.entropy_bottleneck.compress(self.h_a(source.y))[0]
            hyper_hat = self.entropy_bottleneck.decompress([hyper_stream], source.hyper_hat.shape[-2:])
            base = self.h_s(hyper_hat)
            # analysis and entropy-bottleneck decoding must agree on the medians.
            if not torch.equal(hyper_hat, source.hyper_hat):
                raise RuntimeError("hard hyperprior differs from actual entropy decode")
            delta = self.steps(control, allocation_from_codes(source.allocation_codes))
            mean, _, main_stream, audit = self._groups(base, delta, source.y, "encode", trace=trace)
            return {"mean": mean, "hyper_stream": hyper_stream, "main_stream": main_stream,
                    "hyper_shape": tuple(hyper_hat.shape[-2:]), "y_shape": tuple(source.y.shape[-2:]),
                    "allocation_codes": source.allocation_codes,
                    "allocation": pack_codes(source.allocation_codes.flatten().tolist(), 3), "trace": audit}

    @torch.no_grad()
    def decompress(self, packet, trace=False):
        from .packet import unpack_codes

        if self.training:
            raise ValueError("real BT decoding requires eval mode")
        if self.router_strength.item() != 1:
            raise ValueError("cannot deploy an unfinished Router warmup")
        device = next(self.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            h, w = packet.y_shape
            hyper_hat = self.entropy_bottleneck.decompress([packet.hyper_stream], packet.hyper_shape)
            codes = torch.tensor(unpack_codes(packet.allocation, 3, h*w), device=device).reshape(1, 1, h, w)
            delta = self.steps(packet.rate_control(device), allocation_from_codes(codes))
            mean, _, _, audit = self._groups(self.h_s(hyper_hat), delta, mode="decode",
                                            stream=packet.main_stream, trace=trace)
            return {"mean": mean, "delta": delta, "trace": audit}
