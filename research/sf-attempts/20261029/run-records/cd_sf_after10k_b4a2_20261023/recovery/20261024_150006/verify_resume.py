"""Verify the saved experiment state before resuming it after a host reboot."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys

import torch
import yaml


root = Path('/data2/ls/checkpoints/cd_sf_after10k_b4a2_20261023')
audit = Path(__file__).resolve().parent
checkpoint_path = root / 'joint/0024000.pt'
crc = json.loads((audit / 'checkpoint_crc.json').read_text())
assert crc['all_zip_crc_passed'] and crc['checkpoint'] == str(checkpoint_path)
state = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
config = yaml.safe_load((root / 'configs/joint.yaml').read_text())
manifest = json.loads((root / 'manifest.json').read_text())
assert state['method'] == 'bt_self_flow_v1' and state['schema'] == 1
assert state['phase'] == 'joint' and state['backend'] == 'D'
assert state['step'] == 24000
assert state['config'] == config == manifest['config']
assert state['assets'] == manifest['initialization']['assets']
assert state['contract']['world_size'] == 2
assert len(state['rng_by_rank']) == 2
assert state['teacher_initialized_step'] == 10000 and state['teacher']
assert state['raw'] and state['ema']
assert state['optimizer']['state'] and state['aux_optimizer']['state']
assert state['scheduler']['last_epoch'] == state['step']
assert config['train']['batch_size'] == 4
assert config['train']['accumulation_steps'] == 2
assert config['train']['effective_batch'] == 16
assert config['train']['steps'] == 200000
assert config['self_flow']['start_step'] == 10000
report = {
    'verified_at': datetime.now(timezone.utc).isoformat(),
    'checkpoint': str(checkpoint_path),
    'step': state['step'],
    'attempts': state['attempts'],
    'micro_seen': state['micro_seen'],
    'teacher_initialized_step': state['teacher_initialized_step'],
    'teacher_tensors': len(state['teacher']),
    'rep_weight': state['rep_weight'],
    'scheduler_last_epoch': state['scheduler']['last_epoch'],
    'scaler': state['scaler'],
    'rng_ranks': len(state['rng_by_rank']),
    'optimizer_parameter_states': len(state['optimizer']['state']),
    'aux_optimizer_parameter_states': len(state['aux_optimizer']['state']),
    'raw_components': list(state['raw']),
    'ema_components': list(state['ema']),
    'full_precision_retry': state.get('full_precision_retry', False),
    'configuration_matches_frozen_manifest': True,
    'pretrained_assets_match': True,
    'python': sys.executable,
    'torch': torch.__version__,
    'passed': True,
}
with (audit / 'resume_state_verified.json').open('w') as target:
    json.dump(report, target, indent=2)
    target.write('\n')
    target.flush()
    os.fsync(target.fileno())
print(json.dumps(report), flush=True)
