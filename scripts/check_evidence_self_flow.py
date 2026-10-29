"""Bounded real-model two-GPU preflight; its updates are NOT research training."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist

from models.bt_self_flow.checkpoint import DeploymentEMA, evaluation_state, read_checkpoint
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.evidence_training import evidence_backward, average_gradients, synchronize_initial_state
from models.bt_self_flow.experiment import asset_identity, check_stage_model, load_config, make_model, optimizer_groups
from models.bt_self_flow.losses import ImageLoss
from models.bt_self_flow.packet import RateControl
from train_bt_self_flow import auxiliary_update, distributed_setup, gradient_diagnostics, initialize_from, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--init-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rank, world, device = distributed_setup('cuda')
    torch.set_num_threads(4)
    config = load_config(args.config)
    seed_all(config['train']['seed'])
    model = make_model(config, device)
    source = read_checkpoint(args.init_checkpoint)
    assert source['assets'] == asset_identity(config)
    check_stage_model(config, source['config'], source['assets'])
    initialize_from(model, source, 'joint')
    synchronize_initial_state(model, world)
    groups, auxiliary = optimizer_groups(model, config)
    main_params = [p for g in groups for p in g['params']]
    codec = [p for g in groups if g['name']=='codec' for p in g['params']]
    backend = [p for g in groups if g['name']!='codec' for p in g['params']]
    optimizer = torch.optim.AdamW(groups, betas=(.9,.999), weight_decay=0)
    aux_optimizer = torch.optim.AdamW(auxiliary, lr=config['optimizer']['aux_lr'], weight_decay=0)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    loss_fn = ImageLoss(**config['image_loss'], checkpoint_features=True).to(device).eval()
    ema = DeploymentEMA(model)
    dataset = ImageDataset(config['data']['valid'], crop=512, training=False)
    batch = config['train']['batch_size']
    image = torch.stack([dataset[(rank*batch+i)%len(dataset)] for i in range(batch)]).to(device)
    records = []
    model.train()
    # Exercise BOTH active SF at its full coefficient and the max-quality
    # no-teacher/no-SF case with the exact requested per-rank batch and losses.
    for index, q in enumerate((.4, 1.)):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        result, loss, metrics = evidence_backward(model, image, RateControl(q), loss_fn, config,
            3000+index, rank, world, scaler, codec, backend, torch.bfloat16)
        average_gradients(main_params, world)
        norm = torch.nn.utils.clip_grad_norm_(main_params, config['optimizer']['max_grad_norm'])
        assert torch.isfinite(loss) and torch.isfinite(norm), 'nonfinite preflight update'
        gradients = gradient_diagnostics(model)
        optimizer.step()
        auxiliary_update(model.codec, aux_optimizer, world)
        ema.update(model)
        torch.cuda.synchronize(device)
        row = dict(rank=rank, update=index+1, q=q, grad_norm=float(norm), loss=float(loss),
            seconds=time.monotonic()-started, peak_memory_bytes=torch.cuda.max_memory_allocated(device),
            sf=float(metrics['sf']), sf_weight=float(metrics['sf_weight']),
            gradient_norms=gradients)
        records.append(row)
        print(json.dumps(row), flush=True)
        del result, loss, metrics
    # Check replica identity without relying on the reducer that this method
    # deliberately does not use. Include entropy auxiliary and new context.
    max_difference = torch.zeros((), device=device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            reference = parameter.detach().clone()
            dist.broadcast(reference, 0) if world > 1 else None
            max_difference.copy_(torch.maximum(max_difference, (reference-parameter.detach()).abs().max()))
    assert max_difference == 0, 'replicas diverged'
    optimizer.zero_grad(set_to_none=True)
    # Check the deployment EMA, not only raw training parameters. Encoding and
    # decoding in FP32 is the inherited actual-packet validation protocol.
    with torch.no_grad(), evaluation_state(model, ema.state):
        one = image[:1]
        expected = model(one, rate_control=RateControl(.4)).image
        packet = model.compress(one, RateControl(.4))
        original_encode, original_router = model.encode_clean_latent, model.codec.router.forward
        def forbidden(*args, **kwargs):
            raise AssertionError('receiver used source/planner')
        model.encode_clean_latent, model.codec.router.forward = forbidden, forbidden
        before = model.prediction_calls
        try:
            actual = model.decompress(packet.to_bytes())
        finally:
            model.encode_clean_latent, model.codec.router.forward = original_encode, original_router
        parity = float((actual-expected).abs().max())
        nfe = model.prediction_calls-before
        assert parity <= 2e-4 and nfe == 1, f'packet parity/NFE: {parity}/{nfe}'
    report = dict(passed=True, rank=rank, world_size=world, per_rank_batch=batch,
        accumulation=config['train']['accumulation_steps'], effective_batch=world*batch,
        steps=records, replica_max_difference=float(max_difference), packet_max_abs_error=parity,
        packet_bytes=packet.byte_breakdown(), decoder_predictions=nfe,
        ema_contains_context='evidence_projection' in ema.state)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.with_name(target.stem+f'.rank{rank}'+target.suffix).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
