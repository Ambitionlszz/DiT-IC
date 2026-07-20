import math

import lpips
import pyiqa
import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.losses.vqperceptual import *


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        device,
        disc_start,
        logvar_init=0.0,
        rate_weight=1.0,
        pixelloss_weight=2.0,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        pp_style=False,
        **kwargs,
    ):
        super().__init__()
        if disc_loss not in ("hinge", "vanilla"):
            raise ValueError(f"Unsupported discriminator loss: {disc_loss}")

        self.rate_weight = rate_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        self.perceptual_loss = lpips.LPIPS(net="alex").to(device).eval()
        self.perceptual_loss.requires_grad_(False)
        self.dists = pyiqa.create_metric("dists", as_loss=True).to(device).eval()
        self.dists.requires_grad_(False)

        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.pp_style = pp_style
        if pp_style:
            print("Using pp_style for reconstruction loss")

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is None or not last_layer.requires_grad:
            return torch.as_tensor(
                self.discriminator_weight,
                device=nll_loss.device,
                dtype=nll_loss.dtype,
            )
        nll_grads = torch.autograd.grad(
            nll_loss,
            last_layer,
            retain_graph=True,
            allow_unused=True,
        )[0]
        g_grads = torch.autograd.grad(
            g_loss,
            last_layer,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if nll_grads is None or g_grads is None:
            return torch.as_tensor(
                self.discriminator_weight,
                device=nll_loss.device,
                dtype=nll_loss.dtype,
            )
        weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        return torch.clamp(weight, 0.0, 1e4).detach() * self.discriminator_weight

    @staticmethod
    def _rate_loss(rate, num_pixels):
        return sum(
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in rate.values()
        )

    def forward(
        self,
        inputs,
        reconstructions,
        rate,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
        discriminator=None,
    ):
        discriminator = discriminator or self.discriminator
        batch, _, height, width = inputs.size()
        num_pixels = batch * height * width
        rate_loss = self._rate_loss(rate, num_pixels)

        if split == "eval":
            rec_loss = F.mse_loss(inputs.contiguous(), reconstructions.contiguous())
            psnr = 10 * math.log10(4 / rec_loss.item())
            perceptual_loss = self.perceptual_loss(
                inputs.contiguous(),
                reconstructions.contiguous(),
            ).mean()
            loss = (
                rec_loss
                + self.perceptual_weight * perceptual_loss
                + self.rate_weight * rate_loss
            )
            return loss, rec_loss, psnr, rate_loss, perceptual_loss

        perceptual_loss = torch.zeros((), device=inputs.device)
        dists_loss = torch.zeros((), device=inputs.device)
        if self.pp_style:
            rec_loss = (inputs.contiguous() - reconstructions.contiguous()) ** 2
            if self.perceptual_weight > 0:
                dists_loss = self.dists(
                    inputs.contiguous(),
                    reconstructions.contiguous(),
                )
                perceptual_loss = self.perceptual_loss(
                    inputs.contiguous(),
                    reconstructions.contiguous(),
                )
                rec_loss = (
                    self.pixel_weight * rec_loss
                    + self.perceptual_weight * perceptual_loss
                    + 0.2 * dists_loss
                )
            nll_loss = rec_loss
            weighted_nll_loss = rec_loss if weights is None else weights * rec_loss
            weighted_nll_loss = torch.mean(weighted_nll_loss)
            nll_loss = torch.mean(nll_loss)
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
            if self.perceptual_weight > 0:
                dists_loss = self.dists(
                    inputs.contiguous(),
                    reconstructions.contiguous(),
                )
                perceptual_loss = self.perceptual_loss(
                    inputs.contiguous(),
                    reconstructions.contiguous(),
                )
                rec_loss = (
                    rec_loss
                    + self.perceptual_weight * perceptual_loss
                    + 0.2 * dists_loss
                )
            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss if weights is None else weights * nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / batch
            nll_loss = torch.sum(nll_loss) / batch

        base_loss = weighted_nll_loss + self.rate_weight * rate_loss
        common_log = {
            f"{split}/logvar": self.logvar,
            f"{split}/rate_loss": rate_loss.mean(),
            f"{split}/nll_loss": nll_loss.mean(),
            f"{split}/rec_loss": rec_loss.mean(),
            f"{split}/lpips": perceptual_loss.mean(),
            f"{split}/dists": dists_loss.mean(),
            f"{split}/rate_weight": torch.as_tensor(self.rate_weight),
        }

        if optimizer_idx == -1:
            common_log[f"{split}/total_loss"] = base_loss.mean()
            return base_loss, common_log

        if optimizer_idx == 0:
            if cond is None:
                if self.disc_conditional:
                    raise ValueError("Conditional discriminator requires cond")
                logits_fake = discriminator(reconstructions.contiguous())
            else:
                if not self.disc_conditional:
                    raise ValueError("Unconditional discriminator does not accept cond")
                logits_fake = discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    discriminator_weight = self.calculate_adaptive_weight(
                        nll_loss,
                        g_loss,
                        last_layer=last_layer,
                    )
                except RuntimeError:
                    if self.training:
                        raise
                    discriminator_weight = torch.tensor(0.0, device=inputs.device)
            else:
                discriminator_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
            )
            loss = (
                base_loss
                + discriminator_weight * disc_factor * g_loss
            )
            common_log.update(
                {
                    f"{split}/total_loss": loss.mean(),
                    f"{split}/d_weight": discriminator_weight,
                    f"{split}/disc_factor": torch.tensor(disc_factor),
                    f"{split}/g_loss": g_loss.mean(),
                }
            )
            return loss, common_log

        if optimizer_idx == 1:
            if cond is None:
                discriminator_input = torch.cat(
                    (
                        inputs.contiguous().detach(),
                        reconstructions.contiguous().detach(),
                    ),
                    dim=0,
                )
            else:
                discriminator_input = torch.cat(
                    (
                        torch.cat((inputs.contiguous().detach(), cond), dim=1),
                        torch.cat(
                            (reconstructions.contiguous().detach(), cond),
                            dim=1,
                        ),
                    ),
                    dim=0,
                )
            logits_real, logits_fake = discriminator(discriminator_input).chunk(2)

            disc_factor = adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start,
            )
            loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            return loss, {
                f"{split}/disc_loss": loss.clone().mean(),
                f"{split}/logits_real": logits_real.mean(),
                f"{split}/logits_fake": logits_fake.mean(),
            }

        raise ValueError(f"Unsupported optimizer_idx: {optimizer_idx}")
