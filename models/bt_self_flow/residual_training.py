"""Transition checks and one-time SF calibration on independent train crops."""

from contextlib import nullcontext
import math

import torch
import torch.distributed as dist

from .checkpoint import capture_rng, restore_rng
from .data import counter_seed
from .packet import RateControl
from .quantization_residual import erasure_mask, normalized_residual


def balanced_quality(config, step):
    import random
    values = list(config["quality"]["values"])
    random.Random(counter_seed(config["train"]["seed"], step//len(values), 813)).shuffle(values)
    return values[step % len(values)]


def training_mask(config, step, micro, rank, batch, device):
    sf = config["self_flow"]
    ramp = min(1., max(0., (step-sf["warmup"]+1)/max(1, sf["ramp"])))
    grid = config["data"]["crop"]//64
    codes = torch.zeros(batch, 1, grid, grid, dtype=torch.uint8, device=device)
    generator = torch.Generator(device=device).manual_seed(counter_seed(
        config["train"]["seed"], step*config["train"]["accumulation_steps"]+micro, 900+rank))
    return erasure_mask(codes, .5*ramp, sf["cell_fraction"], generator), ramp


def calibrate_residual(model, teacher, dataset, image_loss, config, rank, world, amp_dtype):
    """Measure at μ, preserving all main-loop RNG/data/optimizer state.

    All ranks participate on disjoint crops; four valid samples per q in
    total. The privileged residual is also tested through the actual decoder.
    """
    device = next(model.parameters()).device
    saved_rng = capture_rng(device)
    was_training = model.training
    statistics = torch.zeros(6, 7, dtype=torch.float64, device=device)
    records = []
    try:
        model.eval()
        for qi, q in enumerate(config["quality"]["values"]):
            accepted = 0
            required = config["self_flow"].get("calibration_samples_per_q", 4)
            for index in range(rank, required*16, world):
                if accepted >= math.ceil(required/world):
                    break
                seed = counter_seed(config["train"]["seed"], qi*1000+index, 915)
                image = dataset[(seed % len(dataset), counter_seed(seed, 0, 916))].unsqueeze(0).to(device)
                control = RateControl(q)
                with torch.no_grad():
                    source = model.codec.analyze(model.encode_clean_latent(image), control)
                    generator = torch.Generator(device=device).manual_seed(seed)
                    mask = erasure_mask(source.allocation_codes, 1., config["self_flow"]["cell_fraction"], generator)
                    received = model.codec.quantize_analysis(source, control, erase_mask=mask)
                    native = model.codec.steps(control, source.allocation)
                    fine_delta = torch.minimum(native, received["delta"])
                    if not (fine_delta < received["delta"]-1e-7).any().item():
                        continue
                    fine = model.codec.quantize_analysis(source, control, delta_override=fine_delta, compute_likelihood=False)
                amp = torch.autocast(device.type, dtype=amp_dtype) if device.type == "cuda" else nullcontext()
                with amp:
                    with torch.no_grad():
                        time = model.self_flow_timestep(1, device)
                        target = normalized_residual(teacher(fine["mean"], model.fixed_text_embed, time),
                                                     teacher(received["mean"], model.fixed_text_embed, time))
                    captured = []
                    hook = model.residual_head.register_forward_hook(lambda module, inputs, output: captured.append(output))
                    try:
                        _, prediction, _ = model.repair_received(received["mean"].detach(), received["delta"].detach())
                    finally:
                        hook.remove()
                    mean, variance = captured.pop()
                    distortion, _ = image_loss(prediction, image)
                    nll = .5*((target-mean.float()).square()/variance.float()+variance.float().log()).mean()
                    rd_gradient = torch.autograd.grad(distortion, mean, retain_graph=True)[0].float().norm()
                    sf_gradient = torch.autograd.grad(nll, mean)[0].float().norm()
                    base_d = float(distortion.detach())
                    del prediction, distortion, nll, mean, variance, captured
                    with torch.no_grad():
                        _, oracle_image, _ = model.repair_received(received["mean"], received["delta"], residual_override=target)
                        oracle_d = float(image_loss(oracle_image, image)[0])
                values = [float(rd_gradient.square()), float(sf_gradient.square()), 1., base_d,
                          oracle_d, float(oracle_d < base_d), float(target.square().mean().sqrt())]
                if not all(math.isfinite(v) for v in values) or min(values[:2]) <= 1e-24:
                    raise FloatingPointError("degenerate/nonfinite residual calibration gradient")
                statistics[qi] += torch.tensor(values, dtype=torch.float64, device=device)
                accepted += 1
                records.append(dict(q=q, sample=index, base_d411=base_d, privileged_d411=oracle_d))
            if accepted < math.ceil(required/world):
                raise RuntimeError(f"insufficient nondegenerate calibration samples for q={q}")
        if world > 1:
            dist.all_reduce(statistics)
        ratios = (statistics[:, 0]/statistics[:, 1]).sqrt()
        alpha = min(10., config["self_flow"]["gradient_target"]*float(ratios.quantile(.5)))
        counts = statistics[:, 2]
        base = statistics[:, 3]/counts
        oracle = statistics[:, 4]/counts
        wins = statistics[:, 5]/counts
        systematically_hurts = all(float(oracle[i]) > float(base[i]) and float(wins[i]) <= .25 for i in (0, 5))
        return dict(alpha=alpha, gradient_ratios=ratios.tolist(), samples=counts.tolist(),
                    baseline_d411=base.tolist(), privileged_d411=oracle.tolist(),
                    privileged_win_fraction=wins.tolist(), privileged_gate_passed=not systematically_hurts,
                    local_samples=records, protocol="independent_training_crops_mu_gradient_v1")
    finally:
        model.train(was_training)
        restore_rng(saved_rng)
