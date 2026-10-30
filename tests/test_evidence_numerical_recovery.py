"""Rank-local bad gradients must stop both evidence ranks before any update."""
from contextlib import redirect_stdout
from datetime import timedelta
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import METHOD, SCHEMA, clone_state, read_checkpoint, save_atomic
from test_evidence_self_flow import factory, tiny_config
from test_bt_precision_recovery import assert_same_weights
from test_c_precision_v3 import full_factory, write_image
from test_evidence_self_flow import LoRADecoder


def failure_worker(rank, directory):
    torch.set_num_threads(1)
    root = Path(directory)
    dist.init_process_group('gloo', rank=rank, world_size=2,
        init_method=f"file://{root/'rendezvous'}", timeout=timedelta(seconds=90))
    config = tiny_config(root)
    args = SimpleNamespace(init_checkpoint=str(root/'init.pt'), resume=None, max_updates=None)

    def bad_factory(cfg, device):
        model = factory(cfg, device)
        # Corrupt only one rank. Explicit averaging must expose it to both.
        if rank == 1:
            parameter = next(p for p in model.codec.parameters() if p.requires_grad)
            parameter.register_hook(lambda gradient: gradient * float('nan'))
        return model

    try:
        with patch.object(trainer, 'make_model', side_effect=factory), \
             patch.object(trainer, 'asset_identity', return_value={}), redirect_stdout(io.StringIO()):
            trainer.train(config, args, rank, 2, torch.device('cpu'))
            config['train']['output'] = str(root/'failed')
            with patch.object(trainer, 'make_model', side_effect=bad_factory):
                try:
                    trainer.train(config, args, rank, 2, torch.device('cpu'))
                except FloatingPointError as error:
                    assert 'last completed update' in str(error)
                else:
                    raise AssertionError('bad rank-local gradient was used by an optimizer')
            paused = read_checkpoint(root/'failed/paused.pt')
            assert paused['step'] == 0 and paused['micro_seen'] == 0
            assert paused['attempts'] == 1
            assert paused['scheduler']['last_epoch'] == 0
            assert not paused['optimizer']['state']
            assert not paused['aux_optimizer']['state']
            assert len(paused['rng_by_rank']) == 2
            diagnostic = json.loads((root/'failed/nonfinite_0000000_0000001.json').read_text())
            assert diagnostic['kind'] == 'gradient'
            assert all(item['nonfinite_gradient_count'] > 0 for item in diagnostic['ranks'])
            args.init_checkpoint = None
            args.resume = str(root/'failed/paused.pt')
            trainer.train(config, args, rank, 2, torch.device('cpu'))
        assert_same_weights(read_checkpoint(root/'joint/complete.pt'), read_checkpoint(root/'failed/complete.pt'))
    finally:
        dist.destroy_process_group()


def test_bad_gradient_on_one_evidence_rank_stops_both_and_resumes_exactly(tmp_path):
    write_image(tmp_path)
    config = tiny_config(tmp_path)
    model = full_factory(config)
    model.vae.decoder = LoRADecoder(model.vae.decoder)
    save_atomic(tmp_path/'init.pt', dict(method=METHOD, schema=SCHEMA, phase='backend_init', backend='shared',
        step=0, raw=clone_state(model.export_state(), cpu=True), config=config, assets={}, calibration={}))
    mp.spawn(failure_worker, args=(str(tmp_path),), nprocs=2)
