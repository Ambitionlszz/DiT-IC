"""Discarded real-backbone fault injection: paired FP32 restart then AMP."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse,json,sys,os,math
repo=Path('/data2/ls/code/DiT-IC-c-sf-residual-20261025');sys.path.insert(0,str(repo))
import torch
import torch.distributed as dist
import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.experiment import load_config
from scripts.launch_quantization_residual import ResidualRun
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--worker',action='store_true');parser.add_argument('--resume');parser.add_argument('--target',type=int,required=True);args=parser.parse_args();root=args.root.resolve()
if args.worker:
 rank,world,device=trainer.distributed_setup('cuda');config=load_config(root/'config.json');original_objective=trainer.reconstruction_objective;original_mask=trainer.training_mask;calls=[0];injected=(root/'fault_injected.json').exists()
 def objective(*a,**kw):
  loss,metrics=original_objective(*a,**kw);calls[0]+=1
  if not injected and rank==1 and calls[0]==2:
   (root/'fault_injected.json').write_text(json.dumps(dict(rank=rank,micro=1,reason='deliberate fault injection; discarded updates')))
   loss=loss*float('nan')
  return loss,metrics
 def steady_mask(cfg,step,micro,rank,batch,device):
  # Exercise the maximum planned teacher participation during the probe.
  mask,_=original_mask(cfg,max(step,cfg['self_flow']['warmup']+cfg['self_flow']['ramp']),micro,rank,batch,device)
  return mask,1.
 try:
  with patch.object(trainer,'reconstruction_objective',side_effect=objective),patch.object(trainer,'training_mask',side_effect=steady_mask):
   trainer.train(config,SimpleNamespace(resume=args.resume,init_checkpoint=None,max_updates=args.target),rank,world,device)
 finally:
  dist.destroy_process_group()
else:
 run=ResidualRun(root,['0','1'],sys.executable)
 def command(config_path,initialization,previous):
  return [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__).resolve()),'--worker','--root',str(root),'--target',str(args.target),'--resume',str(previous)]
 with patch.object(run,'joint_command',side_effect=command):run.train_joint(root/'config.json','unused',max_numerical_restarts=1)
 run.update('probe_completed')
 checkpoint=read_checkpoint(root/'joint/paused.pt');records=[json.loads(x) for x in (root/'joint/train.jsonl').read_text().splitlines()];log=(root/'logs/joint.log').read_text();history=json.loads((root/'numerical_restarts.json').read_text())
 assert len(history)==1 and checkpoint['step']==args.target and not checkpoint['full_precision_retry']
 assert 'fp32_replay_completed' in log
 assert all(math.isfinite(row[key]) for row in records for key in ['grad_norm','sf_loss','mse','lpips','dists'])
 assert all(row['sf_valid']>.0 for row in records)
 report=dict(passed=True,capacity_and_recovery_only=True,updates_discarded=True,target_step=args.target,automatic_restart_count=len(history),full_teacher_participation=True,records=records,optimizer_states=len(checkpoint['optimizer']['state']),teacher_state_tensors=len(checkpoint['teacher']))
 (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
