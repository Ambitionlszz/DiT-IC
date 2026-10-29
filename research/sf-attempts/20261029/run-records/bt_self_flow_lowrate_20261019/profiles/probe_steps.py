"""Read-only rate probe of an existing codec under alternative step endpoints."""
import json
import math
from pathlib import Path
import torch
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.experiment import load_config, make_model
from models.bt_self_flow.validation import validate

root = Path('/data2/ls/checkpoints/bt_self_flow_lowrate_20261019/profiles')
config = load_config('configs/bt_self_flow/common.yaml')
model = make_model(config, torch.device('cuda:0'))
checkpoint = read_checkpoint('/data2/ls/checkpoints/bt_self_flow_v1/common/complete.pt')
model.load_bt_state(checkpoint['raw'])
model.eval()
dataset = ImageDataset(config['data']['valid'], 0, False, 6)
report = {'source_step': checkpoint['step'], 'images': len(dataset),
          'meaning': 'Fixed old codec; sensitivity probe, not a forecast of a retrained codec.',
          'candidates': {}}
for coarse, fine in ((4., .5), (8., 1.), (12., 1.), (16., 1.), (12., .8)):
    model.codec.log_delta_coarse.fill_(math.log(coarse))
    model.codec.log_delta_fine.fill_(math.log(fine))
    metrics = validate(model, dataset, actual_rate=True)
    report['candidates'][f'{coarse:g}/{fine:g}'] = metrics
    (root/'step_probe.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'steps': [coarse, fine], 'actual_bpp': {
        q: values['rate_actual'] for q, values in metrics.items()}}), flush=True)
print(json.dumps({'max_memory_bytes': torch.cuda.max_memory_allocated()}), flush=True)
