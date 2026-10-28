"""Self-Flow supervises the image context consumed by the deployed C decoder.

Only the ordinary RD objective trains the transmitted representation. SF uses
an explicit parameter whitelist, including when its native view shares RD's
graph. No alternate image decoder, auxiliary packet, or learned target head.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.sana_self_flow import unwrap_sana_transformer
from .data import counter_seed
from .model import BTCodec, BTForwardResult
from .packet import BTImagePacket, RateControl, fixed_side_bits
from .quantization_residual import CResidualTeacher

METHOD = "context_sf_v1"
MIGRATION = "pure_c10k_context_sf_v1"


def pool_cells(tokens, shape):
    height, width = shape
    if height % 2 or width % 2 or tokens.shape[1] != height*width:
        raise ValueError("context requires an even AE token grid")
    return F.avg_pool2d(tokens.transpose(1, 2).reshape(tokens.shape[0], -1, height, width).float(), 2).flatten(2).transpose(1, 2)


class ContextGenerator(nn.Module):
    def __init__(self, hidden, bottleneck=256):
        super().__init__()
        self.precision = nn.Linear(1, hidden)
        nn.init.zeros_(self.precision.weight)
        nn.init.zeros_(self.precision.bias)
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, bottleneck), nn.SiLU(), nn.Linear(bottleneck, hidden))

    def forward(self, feature, log_step, shape):
        with torch.autocast(device_type=feature.device.type, enabled=False):
            pooled = pool_cells(feature, shape)
            if log_step.shape[-2:] != (shape[0]//2, shape[1]//2):
                raise ValueError("packet precision must match the coding grid")
            value = pooled+self.precision(log_step.float().flatten(2).transpose(1, 2))
            return value+self.mlp(self.norm(value))


class ContextReadout(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(layers)])


def context_block(block, hidden, condition, modulation, context, gate, height, width):
    """Native SANA block, plus parallel image cross-attention at stage 3.

    Reuse its FP32-safe attention processor and all existing Q/K/V/out weights.
    The text branch uses exactly the original query and residual ordering.
    """
    batch = hidden.shape[0]
    shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = (
        block.scale_shift_table[None]+modulation.reshape(batch, 6, -1)).chunk(6, 1)
    norm = (block.norm1(hidden)*(1+scale_sa)+shift_sa).to(hidden.dtype)
    hidden = hidden+gate_sa*block.attn1(norm)
    if block.attn2 is None:
        raise ValueError("context C requires the existing text cross-attention")
    text = block.attn2(hidden, encoder_hidden_states=condition, attention_mask=None)
    image = block.attn2(hidden, encoder_hidden_states=context, attention_mask=None)
    hidden = text+hidden+gate*image
    norm = block.norm2(hidden)*(1+scale_ff)+shift_ff
    ff = block.ff(norm.unflatten(1, (height, width)).permute(0, 3, 1, 2))
    return hidden+gate_ff*ff.flatten(2, 3).permute(0, 2, 1)


def context_forward(sana, generator, readout, latent, condition, timestep, log_step,
                    *, prefix=7, stop=None, checkpoint_blocks=False, context_mode="native"):
    if context_mode not in ("native", "off", "wrong_image"):
        raise ValueError("unknown offline context intervention")
    batch, _, height, width = latent.shape
    patch = int(sana.config.patch_size)
    height, width = height//patch, width//patch
    hidden = sana.patch_embed(latent)
    modulation, embedded = sana.time_embed(timestep, batch_size=batch, hidden_dtype=hidden.dtype)
    condition = sana.caption_norm(sana.caption_projection(condition).view(batch, -1, hidden.shape[-1]))
    context = None
    for index, block in enumerate(sana.transformer_blocks):
        if index <= prefix:
            args = (hidden, None, condition, None, modulation, height, width)
            hidden = checkpoint(block, *args, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else block(*args)
        else:
            operation = partial(context_block, block, height=height, width=width)
            # RMS normalization only for attention. The SF target uses the
            # original content-bearing G output, not the normalized condition.
            normalized = F.rms_norm(context.float(), (context.shape[-1],), eps=1e-6)
            if context_mode == "wrong_image":
                normalized = normalized.roll(1, 0)
            gate = readout.gates[index-prefix-1]
            if context_mode == "off":
                gate = gate*0
            args = (hidden, condition, modulation, normalized, gate)
            hidden = checkpoint(operation, *args, use_reentrant=False) if checkpoint_blocks and torch.is_grad_enabled() else operation(*args)
        if index == prefix:
            context = generator(hidden, log_step, (height, width))
        if index == stop:
            return None, context, hidden
    hidden = sana.norm_out(hidden, embedded, sana.scale_shift_table)
    hidden = sana.proj_out(hidden).reshape(batch, height, width, patch, patch, -1)
    output = hidden.permute(0, 5, 1, 3, 2, 4).reshape(batch, -1, height*patch, width*patch)
    return output, context, None


def context_distance(student, teacher):
    if student.shape != teacher.shape:
        raise ValueError("context and teacher coding grids must agree")
    with torch.autocast(device_type=student.device.type, enabled=False):
        s = student.float()-student.float().mean(1, keepdim=True)
        t = teacher.detach().float()-teacher.detach().float().mean(1, keepdim=True)
        return (1-F.cosine_similarity(s, t, dim=-1, eps=1e-6)).mean(1)


def accumulate_sf(loss, parameters, *, retain_graph=False):
    """Never call loss.backward for SF: only these named parameters may change.

    Loss is already scaled by GradScaler and gradient accumulation. Native RD
    remains connected to the codec and is subsequently differentiated normally.
    """
    parameters = tuple(parameters)
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=False)
    for parameter, gradient in zip(parameters, gradients):
        if parameter.grad is None:
            parameter.grad = gradient.detach()
        else:
            parameter.grad.add_(gradient.detach())


class ContextTeacher(CResidualTeacher):
    def __init__(self, student):
        super().__init__(student.DiT, student.text_proj, student.teacher_layer)
        self.prefix = student.student_layer
        self.context_generator = deepcopy(student.context_generator)
        self.context_readout = ContextReadout(student.teacher_layer-student.student_layer)
        self.context_readout.gates.load_state_dict(dict(list(student.context_readout.gates.state_dict().items())[:student.teacher_layer-student.student_layer]))
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, latent, text, timestep, log_step):
        condition = self.text_projection(text).expand(latent.shape[0], -1, -1)
        _, _, hidden = context_forward(self, self.context_generator, self.context_readout,
            latent, condition, timestep, log_step, prefix=self.prefix, stop=self.target_layer)
        return pool_cells(hidden, latent.shape[-2:])

    def _adapted_state(self):
        return {key: value for key, value in self.state_dict().items()
                if "lora_" in key or key.startswith(("text_projection.", "context_generator.", "context_readout."))}

    @torch.no_grad()
    def update_context_student(self, student, decay):
        source = {key: value for key, value in unwrap_sana_transformer(student.DiT).state_dict().items() if "lora_" in key}
        for label, module in (("text_projection", student.text_proj), ("context_generator", student.context_generator),
                              ("context_readout", student.context_readout)):
            source.update({label+"."+key: value for key, value in module.state_dict().items()})
        for key, target in self._adapted_state().items():
            target.lerp_(source[key].detach().to(target), 1-float(decay))


@dataclass
class ContextViews:
    target: torch.Tensor
    indices: torch.Tensor
    mixed: torch.Tensor | None
    mixed_log_step: torch.Tensor | None
    quality_b: float

    def native_loss(self, context):
        weights = context.new_ones(context.shape[0])
        if self.mixed is not None:
            weights[self.indices] = .5
        return (context_distance(context, self.target)*weights).mean()


class CContextCodec(BTCodec):
    def __init__(self, *args, **kwargs):
        kwargs.update(backend="C", representation=False)
        super().__init__(*args, **kwargs)
        if not self.codec.precision_conditioning or self.codec.local_precision_protocol:
            raise ValueError("context SF uses original precision-v2 C packets")
        sana = unwrap_sana_transformer(self.DiT)
        hidden = int(sana.config.num_attention_heads*sana.config.attention_head_dim)
        with torch.random.fork_rng(devices=[]):
            self.context_generator = ContextGenerator(hidden)
            self.context_readout = ContextReadout(len(sana.transformer_blocks)-self.student_layer-1)
        self.context_mode = "native"  # offline diagnostics only; not checkpoint state

    def sf_parameters(self):
        blocks = unwrap_sana_transformer(self.DiT).transformer_blocks[:self.student_layer+1]
        return tuple(p for n, p in blocks.named_parameters() if p.requires_grad and "lora_" in n)+tuple(self.context_generator.parameters())

    def predict(self, coarse, log_rho=None, transition=1., capture=False):
        if log_rho is None:
            raise ValueError("context decoder needs packet-recoverable precision")
        self.prediction_calls += 1
        output, context, _ = context_forward(unwrap_sana_transformer(self.DiT),
            self.context_generator, self.context_readout, coarse,
            self._fixed_text_condition(coarse.shape[0], coarse.device),
            self.self_flow_timestep(coarse.shape[0], coarse.device), log_rho,
            prefix=self.student_layer, checkpoint_blocks=self.checkpoint_blocks, context_mode=self.context_mode)
        return output, context if capture else None

    def repair(self, coarse, scale_info=None, transition=1., capture=False):
        prediction, context = self.predict(coarse, scale_info["log_step"], transition, capture)
        latent = self.sched.step(prediction, coarse)
        from .decoder import decode_latent
        image = decode_latent(self.vae, latent/self.vae.config.scaling_factor, self.checkpoint_blocks).clamp(-1, 1)
        return latent, image, context

    def forward_analysis(self, source, rate_control=None, transition=1., capture=False):
        control = rate_control or RateControl()
        view = self.codec.quantize_analysis(source, control)
        info = {"log_step": view["delta"].log(), "delta": view["delta"]}
        latent, image, context = self.repair(view["mean"], info, transition, capture)
        h, w = view["allocation_codes"].shape[-2:]
        return BTForwardResult(source.clean, view["mean"], latent, image, view["likelihoods"],
            fixed_side_bits(h, w, "C", False), view["allocation_codes"], info, context, source)

    @torch.no_grad()
    def context_views(self, result, control, teacher, config, step, micro, rank, *, quality_b=None):
        qa = control.value
        qb = random.Random(counter_seed(config["train"]["seed"], step, 4700)).choice(config["quality"]["values"]) if quality_b is None else quality_b
        if control.mixed or qb not in config["quality"]["values"]:
            raise ValueError("context SF samples two native uniform qualities")
        a, da = result.coarse.detach(), result.scale_info["log_step"].detach()
        indices = torch.empty(0, device=a.device, dtype=torch.long)
        mixed = mixed_delta = None
        high, high_delta = a, da
        if abs(qb-qa) >= 2/65535:
            devices = [a.device.index] if a.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                source = self.codec.analyze(result.clean.detach(), RateControl(qb))
                b = self.codec.quantize_analysis(source, RateControl(qb), compute_likelihood=False)
            batch, _, height, width = da.shape
            if batch % 2:
                raise ValueError("context SF requires an even microbatch")
            gen = torch.Generator(device=a.device).manual_seed(counter_seed(config["train"]["seed"], step, 4800+micro*31+rank))
            indices = torch.randperm(batch, generator=gen, device=a.device)[:batch//2]
            count = max(1, round(height*width*config["self_flow"]["cell_fraction"]))
            order = torch.rand(len(indices), height*width, generator=gen, device=a.device).argsort(-1)[:, :count]
            mask = torch.zeros(len(indices), height*width, device=a.device, dtype=torch.bool)
            mask.scatter_(1, order, True)
            mask = mask.reshape(len(indices), 1, height, width)
            token_mask = mask.repeat_interleave(2, -2).repeat_interleave(2, -1)
            mixed = torch.where(token_mask, b["mean"][indices], a[indices])
            mixed_delta = torch.where(mask, b["delta"][indices].log(), da[indices])
            if qb > qa:
                high, high_delta = b["mean"], b["delta"].log()
        chunk = config["self_flow"]["teacher_batch_size"]
        target = torch.cat([teacher(c, self.fixed_text_embed, self.self_flow_timestep(len(c), c.device), d)
            for c, d in zip(high.split(chunk), high_delta.split(chunk))])
        return ContextViews(target, indices, mixed, mixed_delta, qb)

    def mixed_context_loss(self, views, batch):
        _, context, _ = context_forward(unwrap_sana_transformer(self.DiT), self.context_generator,
            self.context_readout, views.mixed.detach(),
            self._fixed_text_condition(len(views.indices), views.mixed.device),
            self.self_flow_timestep(len(views.indices), views.mixed.device), views.mixed_log_step.detach(),
            prefix=self.student_layer, stop=self.student_layer, checkpoint_blocks=self.checkpoint_blocks)
        return .5*context_distance(context, views.target[views.indices]).sum()/batch

    @torch.no_grad()
    def decompress(self, packet):
        if self.training:
            raise ValueError("decompress requires eval mode")
        if isinstance(packet, bytes):
            packet = BTImagePacket.from_bytes(packet)
        if packet.backend != "C":
            raise ValueError("context C decoder needs a C packet")
        view = self.codec.decompress(packet)
        latent, image, _ = self.repair(view["mean"], {"log_step": view["delta"].log()})
        self.last_decode_stats = {"repair_latent_rms": (latent.float()-view["mean"]).square().mean().sqrt().item()}
        h, w = packet.original_shape
        return image[..., :h, :w]

    def export_state(self):
        return dict(super().export_state(), context_generator=self.context_generator.state_dict(),
                    context_readout=self.context_readout.state_dict())

    def load_bt_state(self, state, **kwargs):
        super().load_bt_state(state, **kwargs)
        if not kwargs.get("codec_only", False):
            self.context_generator.load_state_dict(state["context_generator"], strict=True)
            self.context_readout.load_state_dict(state["context_readout"], strict=True)
