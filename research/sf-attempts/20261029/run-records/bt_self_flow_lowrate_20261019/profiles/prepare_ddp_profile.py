import json
from pathlib import Path
from scripts.launch_bt_c_ablation import resolved_configs
root=Path('/data2/ls/checkpoints/bt_self_flow_lowrate_20261019/profiles')
c=resolved_configs(root)['common']
c['train'].update(output=str(root/'ddp16'),log_every=1)
(root/'ddp16.json').write_text(json.dumps(c,indent=2)+'\n')
