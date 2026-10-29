import argparse,json,time,torch
from pathlib import Path
from models.bt_self_flow.experiment import load_config,make_model
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.data import ImageDataset
from models.bt_self_flow.packet import RateControl
from models.bt_self_flow.validation import validate

root=Path("/data2/ls/checkpoints/bt_self_flow_lowrate_20261019/profiles")
c=load_config("configs/bt_self_flow/lowrate_common.yaml")
c["train"].update(output=str(root/"batch16"),effective_batch=16,log_every=1)
(root/"batch16.json").write_text(json.dumps(c,indent=2))
