import sys
from pathlib import Path
from types import SimpleNamespace
import json
import torch

sys.path.insert(0, '/data2/ls/code/DiT-IC-bt-self-flow')
import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import read_checkpoint, save_atomic
from models.bt_self_flow.experiment import load_config

torch.set_num_threads(1)
backend = sys.argv[1]
root = Path('/tmp/bt-self-flow-runtime-20261016/corrected_smoke') / backend
config = load_config(f'/data2/ls/checkpoints/bt_self_flow_v1/configs/{backend.lower()}_joint.yaml')
config['train'].update(output=str(root), log_every=1)
trainer.train(config, SimpleNamespace(resume=None, init_checkpoint='/data2/ls/checkpoints/bt_self_flow_v1/backend_init.pt',
                                     max_updates=2), 0, 1, torch.device('cuda:0'))
state = read_checkpoint(root/'paused.pt')
assert state['step'] == 2 and state['micro_seen'] == 8
rows = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
assert all(torch.isfinite(torch.tensor(r['grad_norm'])) for r in rows)
report = {'backend': backend, 'status': 'passed', 'updates': state['step'], 'micro_seen': state['micro_seen'],
          'peak_memory_bytes': max(r['max_memory_bytes'] for r in rows), 'updates_log': rows,
          'scope': 'engineering_smoke_full_public_512_from_formal_shared_init'}
out = Path('/data2/ls/checkpoints/bt_self_flow_v1_launch_20261015/nonfinite_joint_20261017')
save_atomic(out/f'{backend}_disk_roundtrip.pt', {'method': state['method'], 'schema': state['schema'],
                                              'step': state['step'], 'weight': torch.arange(128.)})
assert read_checkpoint(out/f'{backend}_disk_roundtrip.pt')['step'] == 2
(out/f'{backend}_smoke.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report), flush=True)
