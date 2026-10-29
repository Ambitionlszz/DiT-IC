"""Full-feature 8->20 Self-Flow with a detached, task-supervised C readout.

The auxiliary latent is a training view, never a packet or a billed stream.
Every native image retains the complete C RD objective.
"""
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from models.sana_self_flow import (SelfFlowTeacher, TokenQualityConditioner,
    quality_conditioned_sana_forward, _quality_conditioned_block_forward,
    mix_dual_rate_tokens, unwrap_sana_transformer)
from .decoder import decode_latent
from .model import BTCodec, BTForwardResult
from .packet import BTImagePacket, RateControl, fixed_side_bits
from .data import counter_seed


METHOD = 'dual_rate_task_v1'


def frozen_call(module, *args, **kwargs):
    """Freeze weights, not the derivative w.r.t. the input.

    Call INSIDE checkpoint's callable so recomputation also binds detached
    parameters. Detaching a wrapper around an internally checkpointed decoder
    would restore trainable weights before its backward recomputation.
    """
    state = {k: v.detach() for k, v in module.named_parameters()}
    state.update({k: v.detach() for k, v in module.named_buffers()})
    return functional_call(module, state, args, kwargs, strict=False)


class QualityBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden, condition, timestep, modulation, height, width):
        return _quality_conditioned_block_forward(
            self.block, hidden, condition, timestep, modulation, height, width)


def frozen_decode(vae, latent, checkpoint_blocks):
    base = vae.get_base_model() if hasattr(vae, 'get_base_model') else vae
    if base.use_slicing or base.use_tiling:
        raise ValueError('task SF requires the full-frame C decoder')
    decoder = base.decoder
    def run(module, value):
        return (checkpoint(partial(frozen_call, module), value, use_reentrant=False)
                if checkpoint_blocks and torch.is_grad_enabled() else frozen_call(module, value))
    hidden = run(decoder.conv_in, latent)
    if decoder.in_shortcut:
        hidden = hidden + latent.repeat_interleave(decoder.in_shortcut_repeats, dim=1)
    for block in reversed(decoder.up_blocks):
        hidden = run(block, hidden)
    hidden = frozen_call(decoder.norm_out, hidden.movedim(1, -1)).movedim(-1, 1)
    return run(decoder.conv_out, frozen_call(decoder.conv_act, hidden))


class FullFeaturePredictor(nn.Module):
    """Small raw-feature residual MLP; cosine alone does not fix magnitude."""
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, 2*hidden)
        self.fc2 = nn.Linear(2*hidden, hidden)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, hidden):
        return hidden + self.fc2(F.silu(self.fc1(self.norm(hidden))))


class DualRateTeacher(SelfFlowTeacher):
    @torch.no_grad()
    def forward(self, latent, text, timestep, quality):
        condition = self.text_projection(text).expand(latent.shape[0], -1, -1)
        return quality_conditioned_sana_forward(
            self, latent, condition, timestep, quality, self.quality_conditioner,
            capture_layer=self.target_layer, stop_layer=self.target_layer)[1]


def synchronize_gradients(parameters, world, bucket_bytes=25*1024*1024):
    """One mean reduction after sequential native/aux backward and accumulation.

    Include zero gradients for unused parameters on every rank, in fixed order.
    No DDP reducer may run concurrently with this explicit synchronization.
    """
    if world == 1:
        return
    import torch.distributed as dist
    bucket, size = [], 0
    def flush():
        nonlocal bucket, size
        if not bucket:
            return
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for p in bucket:
            value = flat[offset:offset+p.numel()].view_as(p)
            if p.grad is None:
                p.grad = value.clone()
            else:
                p.grad.copy_(value)
            offset += p.numel()
        bucket, size = [], 0
    for p in parameters:
        if not p.requires_grad:
            continue
        if bucket and (p.dtype != bucket[0].dtype or size+p.numel()*p.element_size() > bucket_bytes):
            flush()
        bucket.append(p)
        size += p.numel()*p.element_size()
    flush()


class CDualRateTaskCodec(BTCodec):
    def __init__(self, *args, **kwargs):
        kwargs.update(backend='C', representation=False)
        super().__init__(*args, **kwargs)
        if not self.codec.precision_conditioning:
            raise ValueError('dual-rate task SF requires precision-v2 C')
        sana = unwrap_sana_transformer(self.DiT)
        hidden = int(sana.config.num_attention_heads*sana.config.attention_head_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4903)
            self.quality_conditioner = TokenQualityConditioner(6, hidden, len(sana.transformer_blocks))
            self.representation_projector = FullFeaturePredictor(hidden)

    def quality_map(self, coarse, quality):
        # Interpolated q is not used by this six-point experiment.
        index = round(float(quality)*5)
        if abs(float(quality)-index/5) > 2/65535:
            raise ValueError('task SF conditioner requires one of the six native quality points')
        return torch.full((coarse.shape[0], *coarse.shape[-2:]), index, dtype=torch.long, device=coarse.device)

    def repair_quality(self, coarse, quality):
        self.prediction_calls += 1
        prediction, _ = quality_conditioned_sana_forward(
            unwrap_sana_transformer(self.DiT), coarse,
            self._fixed_text_condition(coarse.shape[0], coarse.device),
            self.self_flow_timestep(coarse.shape[0], coarse.device), self.quality_map(coarse, quality),
            self.quality_conditioner, checkpoint_blocks=self.checkpoint_blocks)
        latent = self.sched.step(prediction, coarse)
        return latent, decode_latent(self.vae, latent/self.vae.config.scaling_factor,
                                     self.checkpoint_blocks).clamp(-1, 1)

    def forward_analysis(self, source, rate_control=None, transition=1., capture=False):
        control = rate_control or RateControl()
        if control.mixed:
            raise ValueError('native C uses a uniform, independently decodable quality')
        view = self.codec.quantize_analysis(source, control)
        latent, image = self.repair_quality(view['mean'], control.value)
        h, w = view['allocation_codes'].shape[-2:]
        return BTForwardResult(source.clean, view['mean'], latent, image, view['likelihoods'],
                               fixed_side_bits(h, w, 'C', False), view['allocation_codes'], source=source)

    @torch.no_grad()
    def decompress(self, packet):
        if self.training:
            raise ValueError('decompress requires eval mode')
        packet = BTImagePacket.from_bytes(packet) if isinstance(packet, bytes) else packet
        if packet.backend != 'C' or bool(packet.mask):
            raise ValueError('expected a native C packet')
        view = self.codec.decompress(packet)
        latent, image = self.repair_quality(view['mean'], packet.low_code/65535)
        self.last_decode_stats = {'repair_latent_rms': (latent.float()-view['mean']).square().mean().sqrt().item()}
        h, w = packet.original_shape
        return image[..., :h, :w]

    def readout(self, feature, mixed, quality_map):
        """Current shared layers 21--28/VAE, frozen for this call only."""
        sana = unwrap_sana_transformer(self.DiT)
        batch, _, height, width = mixed.shape
        time = self.self_flow_timestep(batch, mixed.device)
        modulation, embedded = frozen_call(sana.time_embed, time, batch_size=batch, hidden_dtype=feature.dtype)
        condition = frozen_call(self.text_proj, self.fixed_text_embed).expand(batch, -1, -1)
        condition = frozen_call(sana.caption_projection, condition).view(batch, -1, feature.shape[-1])
        condition = frozen_call(sana.caption_norm, condition)
        block_q, output_q = frozen_call(self.quality_conditioner, quality_map.flatten(1))
        hidden = feature
        for index in range(self.teacher_layer+1, len(sana.transformer_blocks)):
            q = block_q.to(feature.dtype)*self.quality_conditioner.layer_gain[index].detach().view(1, 1, 6, 1)
            args = (hidden, condition, modulation, q, height, width)
            operation = partial(frozen_call, QualityBlock(sana.transformer_blocks[index]))
            hidden = (checkpoint(operation, *args, use_reentrant=False) if self.checkpoint_blocks
                      and torch.is_grad_enabled() else operation(*args))
        output_base = sana.scale_shift_table.detach()[None, None] + embedded[:, None, None]
        shift, scale = (output_base+output_q.to(feature.dtype)).chunk(2, dim=2)
        hidden = frozen_call(sana.norm_out.norm, hidden)*(1+scale.squeeze(2))+shift.squeeze(2)
        prediction = frozen_call(sana.proj_out, hidden).transpose(1, 2).reshape(batch, -1, height, width)
        latent = self.sched.step(prediction, mixed.detach())
        return frozen_decode(self.vae, latent/self.vae.config.scaling_factor,
                             self.checkpoint_blocks).clamp(-1, 1)

    @torch.no_grad()
    def auxiliary_views(self, result, control, config, step, micro, rank):
        device = result.coarse.device
        generator = torch.Generator(device=device).manual_seed(counter_seed(
            config['train']['seed'], step, 700+micro*13+rank))
        count = max(1, result.coarse.shape[0]//2)
        selection = torch.randperm(result.coarse.shape[0], device=device, generator=generator)[:count]
        a = result.coarse.detach()[selection]
        qa = torch.full((count,), round(control.value*5), dtype=torch.long, device=device)
        qb = torch.randint(6, (count,), device=device, generator=generator)
        # q conditions analysis itself. Never re-quantize A's analysis at q_b.
        b = torch.empty_like(a)
        for index in qb.unique().tolist():
            chosen = (qb == index).nonzero().flatten()
            source_b = self.codec.analyze(result.clean.detach()[selection][chosen], RateControl(index/5))
            b[chosen] = self.codec.quantize_analysis(source_b, RateControl(index/5), compute_likelihood=False)['mean']
        mask = torch.zeros(count, 1, height:=a.shape[-2], width:=a.shape[-1], dtype=torch.bool, device=device)
        priorities = torch.rand(count, height*width, device=device, generator=generator)
        mask.flatten(2).scatter_(2, priorities.argsort(-1)[:, None, :max(1, round(height*width*config['self_flow']['cell_fraction']))], True)
        mixed, qmap = mix_dual_rate_tokens(a, b, qa, qb, mask)
        high = torch.where((qb > qa)[:, None, None, None], b, a)
        highq = torch.maximum(qa, qb)[:, None, None].expand_as(qmap)
        return selection, mixed, qmap, high, highq, qa, qb, mask

    def auxiliary_task(self, result, image, control, teacher, image_loss, config, step, micro, rank,
                       *, prepare=False, intervention=None):
        selection, mixed, qmap, high, highq, qa, qb, mask = self.auxiliary_views(result, control, config, step, micro, rank)
        time = self.self_flow_timestep(mixed.shape[0], mixed.device)
        with torch.no_grad():
            target = teacher(high, self.fixed_text_embed, time, highq)
        from contextlib import nullcontext
        prefix_context = torch.no_grad() if prepare else nullcontext()
        with prefix_context:
            _, feature = quality_conditioned_sana_forward(
                unwrap_sana_transformer(self.DiT), mixed,
                self._fixed_text_condition(mixed.shape[0], mixed.device), time, qmap,
                self.quality_conditioner, capture_layer=self.student_layer,
                stop_layer=self.student_layer, checkpoint_blocks=self.checkpoint_blocks and not prepare)
        predicted = self.representation_projector(feature)
        cosine = (1-F.cosine_similarity(predicted.float(), target.float(), dim=-1)).mean()
        if intervention == 'oracle':
            predicted = target
        elif intervention == 'wrong_image':
            predicted = target.flip(0)
        elif intervention == 'shuffle':
            predicted = target.roll(target.shape[1]//2, dims=1)
        reconstructed = self.readout(predicted, mixed, qmap)
        distortion, pixels = image_loss(reconstructed, image[selection])
        loss = cosine+distortion
        # Only head preparation uses feature magnitude regression. Main SF is
        # exactly cosine + D411, with no hidden per-image multiplier.
        if prepare:
            loss = loss + F.mse_loss(predicted.float(), target.float())/target.float().square().mean().clamp_min(1e-6)
        return loss, dict(sf_cosine=cosine, sf_aux_distortion=distortion,
            sf_aux_mse=pixels['mse'], sf_aux_lpips=pixels['lpips'], sf_aux_dists=pixels['dists'],
            sf_pred_rms=predicted.float().square().mean().sqrt(), sf_target_rms=target.float().square().mean().sqrt(),
            sf_mixed_fraction=mask.float().mean(), sf_equal_quality=(qa == qb).float().mean(),
            sf_b_better=(qb > qa).float().mean())


def prepare_predictor(model, teacher, dataset, image_loss, config, rank, world, amp_dtype):
    """Short, separately recorded head-only initialization; no native C steps."""
    from types import SimpleNamespace
    from .checkpoint import capture_rng, restore_rng
    device = next(model.parameters()).device
    rng = capture_rng(device)
    parameters = list(model.representation_projector.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config['self_flow'].get('prepare_lr', 1e-3), weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and amp_dtype == torch.float16, init_scale=16.)
    records = []
    try:
        for index in range(config['self_flow'].get('prepare_steps', 64)):
            # Same native batch layout; these draws do not advance CounterSampler.
            batch = config['train']['batch_size']
            image = torch.stack([dataset[(rank*batch+j+index*world*batch, counter_seed(
                config['train']['seed'], index, 1700+rank*batch+j))] for j in range(batch)]).to(device)
            control = RateControl(config['quality']['values'][index % 6])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == 'cuda'):
                with torch.no_grad():
                    clean = model.encode_clean_latent(image)
                    source = model.codec.analyze(clean, control)
                    coarse = model.codec.quantize_analysis(source, control, compute_likelihood=False)['mean']
                loss, metrics = model.auxiliary_task(SimpleNamespace(clean=clean, coarse=coarse),
                    image, control, teacher, image_loss, config, index, 0, rank, prepare=True)
            finite = torch.isfinite(loss).int()
            if world > 1:
                import torch.distributed as dist
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError('nonfinite predictor preparation; C has not been updated')
            scaler.scale(loss).backward()
            synchronize_gradients(parameters, world)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 4.)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < before:
                raise FloatingPointError('predictor preparation overflow; reduce preparation scale')
            if index == 0 or (index+1) % 16 == 0:
                record = dict(prepare_update=index+1, loss=loss.detach().item(), grad_norm=norm.item(),
                              **{k: v.detach().item() for k, v in metrics.items()})
                records.append(record)
                if rank == 0:
                    import json
                    print(json.dumps(dict(task_sf_head_preparation=record)), flush=True)
    finally:
        optimizer.zero_grad(set_to_none=True)
        restore_rng(rng)
    return dict(steps=config['self_flow'].get('prepare_steps', 64), native_steps_advanced=0,
                frozen_C=True, records=records)
