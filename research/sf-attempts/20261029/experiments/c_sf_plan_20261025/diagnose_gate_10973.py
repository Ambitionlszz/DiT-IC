from pathlib import Path
import sys,json,itertools,gc
sys.path.insert(0,'/data2/ls/code/DiT-IC-c-sf-residual-20261025')
import torch
from models.bt_self_flow.checkpoint import read_checkpoint,restore_rng
from models.bt_self_flow.data import ImageDataset,CounterSampler
from models.bt_self_flow.experiment import load_config,make_model
from models.bt_self_flow.residual_training import training_mask
from models.sana_self_flow import unwrap_sana_transformer
from train_bt_self_flow import sample_control
root=Path('/data2/ls/checkpoints/c_quant_residual_sf_continue_20261025')
config=load_config(root/'configs/joint.json');ck=read_checkpoint(root/'joint/paused.pt');device=torch.device('cuda:0');torch.cuda.set_device(device)
model=make_model(config,device);model.load_bt_state(ck['raw']);model.train()
dataset=ImageDataset(config['data']['train'],512)
keys=list(itertools.islice(iter(CounterSampler(len(dataset),config['train']['seed'],1,2,4,ck['micro_seen'],23)),4))
image=torch.stack([dataset[key] for key in keys]).to(device)
report=dict(step=ck['step'],rank=1,keys=keys,model_state_finite=all(torch.isfinite(t).all().item() for d in ck['raw'].values() for t in d.values() if t.is_floating_point()),trials=[])
current=[]
def observe(name):
 def hook(module,inputs,output):
  tensors=[output] if torch.is_tensor(output) else [v for v in output if torch.is_tensor(v)] if isinstance(output,(tuple,list)) else []
  for value in tensors:
   if value.is_floating_point() and not torch.isfinite(value).all().item():
    if not current:current.append(dict(module=name,type=type(module).__name__,shape=list(value.shape),dtype=str(value.dtype),finite_fraction=torch.isfinite(value).float().mean().item(),input_max=[t.detach().abs().max().item() for t in inputs if torch.is_tensor(t)]));print('FIRST_NONFINITE',current[0],flush=True)
 return hook
sana=unwrap_sana_transformer(model.DiT)
def gate_pre(module, inputs):
 modulation=inputs[4]; gates=(module.scale_shift_table[None]+modulation.reshape(modulation.shape[0],6,-1)).chunk(6,dim=1); g=gates[-1];print('GATE26',str(module.scale_shift_table.dtype),str(modulation.dtype),str(g.dtype),g.abs().max().item(),flush=True)
 module._diagnostic_gate=g
def ff_hook(module,inputs,output):
 g=sana.transformer_blocks[26]._diagnostic_gate;f=output.flatten(2,3).permute(0,2,1);product=g*f;print('FF26',str(f.dtype),f.abs().max().item(),'product',str(product.dtype),torch.isfinite(product).all().item(),(g.float()*f.float()).abs().max().item(),flush=True)
sana.transformer_blocks[26].register_forward_pre_hook(gate_pre)
sana.transformer_blocks[26].ff.register_forward_hook(ff_hook)
handles=[m.register_forward_hook(observe(name)) for name,m in sana.named_modules() if name and (not list(m.children()) or name.startswith('transformer_blocks.') and name.count('.')==1)]
for amp_enabled in [True,False]:
 current.clear();restore_rng(ck['rng_by_rank'][1]);control=sample_control(config,ck['step'],1,(4,8,8),device);mask,ramp=training_mask(config,ck['step'],0,1,4,device)
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp_enabled): result=model(image,rate_control=control,sf_mask=mask)
 trial=dict(amp=amp_enabled,quality=control.value,mask_cells=int(mask.sum()),first_nonfinite=list(current),tensors={k:dict(finite=torch.isfinite(getattr(result,k)).all().item(),max_abs=getattr(result,k).abs().max().item()) for k in ['clean','coarse','latent','image']},peak_allocated=torch.cuda.max_memory_allocated())
 report['trials'].append(trial);print('TRIAL',json.dumps(trial),flush=True);del result;gc.collect();torch.cuda.empty_cache()
for h in handles:h.remove()
(root/'nonfinite_gate_diagnosis_10973.json').write_text(json.dumps(report,indent=2)+'\n')
