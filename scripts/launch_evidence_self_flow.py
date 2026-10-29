"""Launch the authorized E3 experiment detached; never commandeer other GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--gpus', default='1,2')
    parser.add_argument('--config', default=str(REPO/'configs/bt_self_flow/evidence_v1/e3_self_flow.yaml'))
    parser.add_argument('--init-checkpoint', default='/data1/ls/code/runs/newc_pit_gpu0_20261029/backend_init.pt')
    parser.add_argument('--resume')
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = root/'launch.json'
    if identity.exists():
        prior = json.loads(identity.read_text())
        try:
            os.kill(prior['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('recorded launcher is still alive; refusing a duplicate experiment')
        if not args.resume:
            raise RuntimeError('run root already has a launch record; use explicit resume')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    devices = {line.split(',')[0].strip(): line.split(',')[1].strip() for line in rows.splitlines()}
    selected = args.gpus.split(',')
    if len(set(selected)) != 2:
        raise ValueError('this recipe requires exactly two distinct physical GPUs')
    uuids = [devices[index] for index in selected]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids), SF_RUN_ROOT=str(root),
        TORCH_HOME='/data1/ls/pretrained/pretrained/torch', HF_HOME='/data1/ls/cache/huggingface',
        XDG_CACHE_HOME='/data1/ls/cache', OMP_NUM_THREADS='4', GLOO_SOCKET_IFNAME='lo',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
        '--nproc-per-node=2', '--max-restarts=0', 'train_bt_self_flow.py', '--config', str(Path(args.config).resolve()),
        '--checkpoint-steps', '10']
    command += ['--resume', args.resume] if args.resume else ['--init-checkpoint', args.init_checkpoint]
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), repository=str(REPO),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO,text=True).strip(),
        physical_gpus=selected, gpu_uuids=uuids, command=command,
        config_sha256=sha256(args.config), initialization=args.init_checkpoint,
        initialization_sha256=sha256(args.init_checkpoint),
        checkpoint=str(root/'joint'), log=str(root/'launcher.log'))
    # Preserve the previous launch when explicitly resuming a stopped run.
    if identity.exists():
        identity.rename(root/f"launch_before_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with (root/'launcher.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record['pid'] = process.pid
    identity.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
