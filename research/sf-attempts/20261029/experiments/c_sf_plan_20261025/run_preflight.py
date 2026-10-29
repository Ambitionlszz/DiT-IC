"""Wait for the authorized safe handover, then probe 8/1 and OOM-only 4/2."""
from pathlib import Path
import datetime, json, os, subprocess, sys, time
repo=Path('/data2/ls/code/DiT-IC-c-sf-residual-20261025')
root=Path('/data2/ls/checkpoints/c_quant_residual_sf_20261025')
old=Path('/data2/ls/checkpoints/cd_sf_after10k_b4a2_20261023')
python='/tmp/bt-self-flow-runtime-20261016/linux/venvs/ditic/bin/python'
status=root/'preflight/status.json'
def publish(**kwargs):
    value=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**kwargs)
    temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(status)
    print(json.dumps(value),flush=True)
limit=time.monotonic()+900
publish(stage='waiting_for_old_checkpoint')
while True:
    alive=[]
    for pid in (5818,5853,5859,5860):
        p=Path(f'/proc/{pid}/status')
        if p.exists() and '\nState:\tZ' not in p.read_text(): alive.append(pid)
    if not alive and not (old/'joint/paused.pt.tmp').exists(): break
    if time.monotonic()>limit: raise TimeoutError('old job did not finish safe checkpoint')
    time.sleep(2)
import torch
checkpoint=torch.load(old/'joint/paused.pt',map_location='cpu',weights_only=True,mmap=True)
assert checkpoint['step']>=33000 and len(checkpoint['rng_by_rank'])==2
record=dict(old_run=str(old),saved_step=checkpoint['step'],checkpoint=str(old/'joint/paused.pt'),safe_stop=True)
(root/'old_run_saved.json').write_text(json.dumps(record,indent=2)+'\n')
del checkpoint
memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
assert len(memory)>=2 and all(int(x)<2048 for x in memory[:2]),memory
publish(stage='old_job_saved_and_gpus_released',**record)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',GLOO_SOCKET_IFNAME='lo',PYTHONUNBUFFERED='1',TORCH_HOME='/data2/ls/checkpoints/torch-hub',HF_HOME='/data2/ls/checkpoints/hf-cache')
for name in ('c_sf','c_sf_b4a2'):
    cfg=repo/f'configs/bt_self_flow/quantization_residual/{name}.json'
    output=root/f'preflight/{name}.json';log=root/f'preflight/{name}.log'
    publish(stage='probing',configuration=str(cfg),log=str(log))
    command=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/check_quantization_residual_gpu.py','--config',str(cfg),'--init-checkpoint','/data2/ls/checkpoints/ditic-50k-warmups/c_precision_v2_lambda16_50k_20261022/backend_init.pt','--output',str(output)]
    with log.open('w') as f:
        code=subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if code==0:
        report=json.loads(output.read_text());assert report['passed']
        (root/'preflight/selected.json').write_text(json.dumps(dict(config=str(cfg),report=str(output)),indent=2)+'\n')
        publish(stage='passed',configuration=str(cfg),report=str(output))
        break
    text=log.read_text(errors='replace')
    oom=any(s in text.lower() for s in ('out of memory','outofmemoryerror','cuda_error_out_of_memory'))
    publish(stage='probe_failed',configuration=str(cfg),returncode=code,oom=oom,log=str(log))
    if not oom or name=='c_sf_b4a2':
        sys.exit(code or 1)
