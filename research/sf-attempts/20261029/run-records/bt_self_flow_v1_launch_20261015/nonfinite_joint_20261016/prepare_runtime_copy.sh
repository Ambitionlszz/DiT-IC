#!/usr/bin/env bash
set -euo pipefail
bt_runtime_root=/tmp/bt-self-flow-runtime-20261016
bt_runtime_image="$bt_runtime_root/ditic-code.ext4"
bt_runtime_mount="$bt_runtime_root/linux"
bt_source_image=/data2/ls/code/volumes/ditic-code.ext4
bt_fuse_tools=/data2/ls/code/deployment-bootstrap/fuse-tools
[[ "$bt_runtime_image" == /tmp/bt-self-flow-runtime-20261016/ditic-code.ext4 ]]
[[ $(stat -c %s "$bt_runtime_image") == $(stat -c %s "$bt_source_image") ]]
# Repair/check only the disposable local copy, never the live source image.
set +e
/usr/sbin/e2fsck -f -p "$bt_runtime_image" > "$bt_runtime_root/filesystem_check.log" 2>&1
bt_fs_status=$?
set -e
if (( bt_fs_status > 1 )); then
  cat "$bt_runtime_root/filesystem_check.log"
  exit "$bt_fs_status"
fi
mkdir -p "$bt_runtime_mount"
LD_LIBRARY_PATH="$bt_fuse_tools/lib/x86_64-linux-gnu" \
  "$bt_fuse_tools/usr/bin/fuse2fs" "$bt_runtime_image" "$bt_runtime_mount" \
  -o "rw,nosuid,nodev,default_permissions,fakeroot,no_default_opts,kernel_cache,use_ino,subtype=ext4,fsname=$bt_runtime_image,big_writes,max_read=131072,attr_timeout=600,entry_timeout=600,negative_timeout=30"
/usr/bin/python3 - <<'PY'
import hashlib, json
from pathlib import Path
source=Path('/data2/ls/code/linux')
target=Path('/tmp/bt-self-flow-runtime-20261016/linux')
venv=target/'venvs/ditic'
old=str(source)
new=str(target)
checks={}
for relative in ('python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12',
                 'venvs/ditic/lib/python3.12/site-packages/torch/__init__.py',
                 'venvs/ditic/lib/python3.12/site-packages/torch/version.py',
                 'venvs/ditic/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12'):
    source_hash=hashlib.sha256((source/relative).read_bytes()).hexdigest()
    target_hash=hashlib.sha256((target/relative).read_bytes()).hexdigest()
    assert source_hash==target_hash,relative
    checks[relative]=source_hash
cfg=venv/'pyvenv.cfg'
cfg.write_text(cfg.read_text().replace(old,new))
for path in (venv/'bin').iterdir():
    if path.is_symlink():
        link=str(path.readlink())
        if link.startswith(old+'/'):
            path.unlink()
            path.symlink_to(new+link[len(old):])
report={'source':str(source),'runtime':str(target),'hash_checks':checks,'source_unchanged':True}
Path('/tmp/bt-self-flow-runtime-20261016/runtime_copy.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
PY
LD_LIBRARY_PATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$bt_runtime_mount/venvs/ditic/bin/python" - <<'PY'
import sys,torch
print({'python':sys.executable,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0)},flush=True)
PY
