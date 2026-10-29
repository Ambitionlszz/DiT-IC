import sys
from pathlib import Path
sys.path.insert(0,'/data2/ls/code/DiT-IC-bt-self-flow')
import hashlib,json
from types import SimpleNamespace
import torch
import train_bt_self_flow as trainer
from models.bt_self_flow.checkpoint import read_checkpoint
from models.bt_self_flow.experiment import load_config
from models.sana_self_flow import unwrap_sana_transformer
backend=sys.argv[1]
root=Path('/tmp/bt-self-flow-runtime-20261016/recovery_gpu')
root.mkdir(parents=True,exist_ok=True)
config=load_config(f'/data2/ls/checkpoints/bt_self_flow_v1/configs/{backend.lower()}_joint.yaml')
config['train'].update(output=str(root/backend),log_every=1)
config['data']['workers']=0
original_objective=trainer.reconstruction_objective
original_make=trainer.make_model
original_save=trainer.save_atomic
calls=[]
saves=[]
models=[]
def make(*args,**kwargs):
 model=original_make(*args,**kwargs)
 models.append(model)
 return model
def objective(*args,**kwargs):
 image=args[1]
 fingerprint=hashlib.sha256(image.detach().cpu().numpy().tobytes()).hexdigest()
 calls.append(fingerprint)
 loss,metrics=original_objective(*args,**kwargs)
 # Fail after one microbatch has already accumulated gradients.
 if len(calls)==2:
  assert torch.is_autocast_enabled('cuda')
  loss=loss*float('nan')
 return loss,metrics
def save(path,state):
 if Path(path).name=='paused.pt':
  saves.append({'step':state['step'],'micro_seen':state['micro_seen'],'full_precision_retry':state['full_precision_retry']})
 return original_save(path,state)
trainer.make_model=make
trainer.reconstruction_objective=objective
trainer.save_atomic=save
trainer.train(config,SimpleNamespace(resume=None,init_checkpoint='/data2/ls/checkpoints/bt_self_flow_v1/backend_init.pt',max_updates=2),0,1,torch.device('cuda:0'))
state=read_checkpoint(root/backend/'paused.pt')
assert saves[0]=={'step':0,'micro_seen':0,'full_precision_retry':True}
assert saves[-1]=={'step':2,'micro_seen':8,'full_precision_retry':False}
assert calls[:2]==calls[2:4]
assert state['step']==2 and state['micro_seen']==8
assert not models[0].checkpoint_blocks
assert not unwrap_sana_transformer(models[0].DiT).gradient_checkpointing
rows=[json.loads(line) for line in (root/backend/'train.jsonl').read_text().splitlines()]
assert all(torch.isfinite(torch.tensor(row['grad_norm'])) for row in rows)
report={'status':'passed','scope':'injected_nonfinite_forward_full_public_model_not_research_result','backend':backend,'updates':state['step'],'micro_seen':state['micro_seen'],'same_batches_replayed':True,'saves':saves,'max_memory_bytes':max(row['max_memory_bytes'] for row in rows),'flags_restored':True,'source':'/data2/ls/checkpoints/bt_self_flow_v1/backend_init.pt','precision':'normal_fp16_with_one_fp32_update','calls':len(calls)}
(root/f'{backend}_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
