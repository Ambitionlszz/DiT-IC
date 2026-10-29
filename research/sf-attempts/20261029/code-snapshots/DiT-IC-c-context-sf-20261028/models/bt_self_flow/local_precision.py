"""One paid local-quality stream, one full student, and an EMA H20 target."""
from dataclasses import fields, replace
from copy import deepcopy
import random

import torch
from torch import nn
from torch.nn import functional as F

from models.sana_self_flow import unwrap_sana_transformer
from .data import counter_seed
from .model import BTCodec, BTForwardResult
from .packet import BTImagePacket, LocalPrecisionControl, RateControl, fixed_side_bits, quality_code
from .quantization_residual import CResidualTeacher, c_sana_forward

METHOD = "local_precision_v1"
MIGRATION = "pure_c10k_local_precision_v1"


def sample_local_control(config, qa, step, micro, rank, shape, device, quality_b=None):
    """Independent qB; exactly half the local batch mixes 25% coding cells."""
    if qa.mixed:
        raise ValueError("the RD budget must be a whole-image qA")
    qb = (random.Random(counter_seed(config['train']['seed'], step, 3700)).choice(
        config['quality']['values']) if quality_b is None else quality_b)
    if qb not in config['quality']['values']:
        raise ValueError("qB must be a native quality")
    if quality_code(qb) == qa.low_code:
        return qa, qb
    batch, height, width = shape
    if batch % 2:
        raise ValueError("local-precision training requires an even local batch")
    generator = torch.Generator(device=device).manual_seed(counter_seed(
        config['train']['seed'], step, 3800+31*micro+rank))
    active = torch.randperm(batch, generator=generator, device=device)[:batch//2]
    order = torch.rand(batch, height*width, generator=generator, device=device).argsort(-1)
    count = max(1, round(height*width*config['self_flow']['cell_fraction']))
    alternate = torch.zeros(batch, height*width, dtype=torch.bool, device=device)
    alternate[active[:, None], order[active, :count]] = True
    alternate = alternate.reshape(batch, 1, height, width)
    high_mask = alternate if qb > qa.value else ~alternate
    return LocalPrecisionControl(min(qa.value, qb), max(qa.value, qb), high_mask,
                                 analysis_quality=qa.value), qb


def centered_tokens(feature):
    with torch.autocast(device_type=feature.device.type, enabled=False):
        value = feature.float()
        return F.normalize(value-value.mean(1, keepdim=True), dim=-1, eps=1e-6)


def centered_token_loss(student, target, coarse_mask, token_shape):
    if student.shape != target.shape or student.shape[1] != token_shape[0]*token_shape[1]:
        raise ValueError("student/teacher feature grids must agree")
    with torch.autocast(device_type=student.device.type, enabled=False):
        distance = 1-(centered_tokens(student)*centered_tokens(target.detach())).sum(-1)
        valid = F.interpolate(coarse_mask.float(), size=token_shape, mode='nearest').flatten(1)
        per_image = (distance*valid).sum(1)/valid.sum(1).clamp_min(1)
        return per_image.mean(), valid.mean()


def precision_tokens(adapter, log_step, shape):
    with torch.autocast(device_type=log_step.device.type, enabled=False):
        local = F.interpolate(log_step.float(), size=shape, mode='nearest').flatten(2).transpose(1, 2)
        return adapter(local)


class LocalPrecisionTeacher(CResidualTeacher):
    def __init__(self, student):
        super().__init__(student.DiT, student.text_proj, student.teacher_layer)
        self.precision_adapter = deepcopy(student.precision_adapter)
        self.requires_grad_(False).eval()

    @torch.no_grad()
    def forward(self, latent, text, timestep, log_step):
        condition = self.text_projection(text).expand(latent.shape[0], -1, -1)
        return c_sana_forward(self, latent, condition, timestep,
            precision_tokens=precision_tokens(self.precision_adapter, log_step, latent.shape[-2:]),
            capture_layer=self.target_layer, stop_layer=self.target_layer)[1]

    def _adapted_state(self):
        return {k: v for k, v in self.state_dict().items()
                if 'lora_' in k or k.startswith(('text_projection.', 'precision_adapter.'))}

    @torch.no_grad()
    def update_local_student(self, student, decay):
        source = {k: v for k, v in unwrap_sana_transformer(student.DiT).state_dict().items() if 'lora_' in k}
        source.update({'text_projection.'+k: v for k, v in student.text_proj.state_dict().items()})
        source.update({'precision_adapter.'+k: v for k, v in student.precision_adapter.state_dict().items()})
        for key, target in self._adapted_state().items():
            target.lerp_(source[key].detach().to(target), 1-float(decay))


class CLocalPrecisionCodec(BTCodec):
    def __init__(self, *args, **kwargs):
        kwargs.update(backend='C', representation=False)
        super().__init__(*args, **kwargs)
        if not self.codec.local_precision_protocol:
            raise ValueError("local SF requires an explicit precision-v3 codec")
        sana = unwrap_sana_transformer(self.DiT)
        hidden = int(sana.config.num_attention_heads*sana.config.attention_head_dim)
        with torch.random.fork_rng(devices=[]):
            self.precision_adapter = nn.Linear(1, hidden)
            nn.init.zeros_(self.precision_adapter.weight)
            nn.init.zeros_(self.precision_adapter.bias)

    def predict(self, coarse, log_rho=None, transition=1., capture=False):
        if log_rho is None:
            raise ValueError("the deployed C decoder requires packet-recoverable log delta")
        self.prediction_calls += 1
        prediction, feature, _, _ = c_sana_forward(unwrap_sana_transformer(self.DiT), coarse,
            self._fixed_text_condition(coarse.shape[0], coarse.device),
            self.self_flow_timestep(coarse.shape[0], coarse.device),
            precision_tokens=precision_tokens(self.precision_adapter, log_rho, coarse.shape[-2:]),
            capture_layer=self.student_layer if capture else None, checkpoint_blocks=self.checkpoint_blocks)
        return prediction, feature

    def repair(self, coarse, scale_info=None, transition=1., capture=False):
        prediction, feature = self.predict(coarse, scale_info['log_step'], transition, capture)
        latent = self.sched.step(prediction, coarse)
        from .decoder import decode_latent
        image = decode_latent(self.vae, latent/self.vae.config.scaling_factor, self.checkpoint_blocks).clamp(-1, 1)
        return latent, image, feature

    def forward(self, image, rate_control=None, transition=1., capture=False, *, local_control=None, **kwargs):
        anchor = rate_control or RateControl()
        if anchor.mixed:
            if not isinstance(anchor, LocalPrecisionControl):
                raise ValueError("mixed encoding requires an explicit analysis anchor")
            control, anchor = anchor, RateControl(anchor.analysis_quality)
        else:
            control = local_control or anchor
        source = self.codec.analyze(self.encode_clean_latent(image), anchor)
        return self.forward_analysis(source, control, transition, capture)

    def forward_analysis(self, source, rate_control=None, transition=1., capture=False):
        control = rate_control or RateControl()
        view = self.codec.quantize_analysis(source, control)
        info = {'log_step': view['delta'].log(), 'delta': view['delta']}
        latent, image, feature = self.repair(view['mean'], info, transition, capture)
        h, w = view['allocation_codes'].shape[-2:]
        side_bits = fixed_side_bits(h, w, 'C', False)
        if control.mixed:
            # Uniform samples are serialized as a one-q packet, without a mask.
            mask = control.high_mask.flatten(1)
            is_mixed = mask.any(1) & ~mask.all(1)
            side_bits += float(is_mixed.float().mean())*(fixed_side_bits(h, w, 'C', True)-side_bits)
        return BTForwardResult(source.clean, view['mean'], latent, image, view['likelihoods'],
                               side_bits, view['allocation_codes'], info, feature, source)

    def local_sf(self, result, anchor, qb, teacher, config):
        qh = max(anchor.value, qb)
        # Detach every analysis field before using the shared codec. The only
        # SF-to-codec graph is the student's actually billed forward.
        source = replace(result.source, **{f.name: getattr(result.source, f.name).detach()
                         for f in fields(result.source) if torch.is_tensor(getattr(result.source, f.name))})
        with torch.no_grad():
            reference = LocalPrecisionControl(qh, analysis_quality=anchor.value)
            fine = self.codec.quantize_analysis(source, reference, compute_likelihood=False)
            coarse = result.scale_info['delta'].detach() > fine['delta']*(1+1e-6)
            chunk = config['self_flow'].get('teacher_batch_size', 4)
            target = torch.cat([teacher(c, self.fixed_text_embed,
                self.self_flow_timestep(c.shape[0], c.device), d.log())
                for c, d in zip(fine['mean'].split(chunk), fine['delta'].split(chunk))])
        loss, fraction = centered_token_loss(result.feature, target, coarse, result.coarse.shape[-2:])
        return loss, dict(sf_loss=loss.detach(), sf_valid_fraction=fraction,
            sf_q_b=loss.new_tensor(qb), sf_teacher_spatial_std=target.float().std(dim=1).mean(),
            sf_native_spatial_std=result.feature.detach().float().std(dim=1).mean())

    @torch.no_grad()
    def decompress(self, packet):
        if self.training:
            raise ValueError("decompress requires eval mode")
        if isinstance(packet, bytes):
            packet = BTImagePacket.from_bytes(packet)
        if packet.backend != 'C':
            raise ValueError("local precision C cannot decode another backend")
        view = self.codec.decompress(packet)
        latent, image, _ = self.repair(view['mean'], {'log_step': view['delta'].log()})
        self.last_decode_stats = {'repair_latent_rms': (latent.float()-view['mean']).square().mean().sqrt().item()}
        h, w = packet.original_shape
        return image[..., :h, :w]

    def export_state(self):
        return dict(super().export_state(), precision_adapter=self.precision_adapter.state_dict())

    def load_bt_state(self, state, **kwargs):
        super().load_bt_state(state, **kwargs)
        if not kwargs.get('codec_only', False):
            self.precision_adapter.load_state_dict(state['precision_adapter'], strict=True)
