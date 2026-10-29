#!/usr/bin/env bash
set -euo pipefail
sf_runtime_root=/tmp/bt-self-flow-runtime-20261016
sf_runtime_image="$sf_runtime_root/ditic-code.ext4"
sf_runtime_mount="$sf_runtime_root/linux"
sf_source_image=/data2/ls/code/volumes/ditic-code.ext4
sf_fuse_tools=/data2/ls/code/deployment-bootstrap/fuse-tools
[[ $(stat -c %s "$sf_runtime_image") == $(stat -c %s "$sf_source_image") ]]
set +e
/usr/sbin/e2fsck -f -p "$sf_runtime_image" > "$sf_runtime_root/filesystem_check.log" 2>&1
sf_fs_status=$?
set -e
if (( sf_fs_status > 1 )); then
  cat "$sf_runtime_root/filesystem_check.log"
  exit "$sf_fs_status"
fi
mkdir -p "$sf_runtime_mount"
LD_LIBRARY_PATH="$sf_fuse_tools/lib/x86_64-linux-gnu" \
 "$sf_fuse_tools/usr/bin/fuse2fs" "$sf_runtime_image" "$sf_runtime_mount" \
 -o "rw,nosuid,nodev,default_permissions,fakeroot,no_default_opts,kernel_cache,use_ino,subtype=ext4,fsname=$sf_runtime_image,big_writes,max_read=131072,attr_timeout=600,entry_timeout=600,negative_timeout=30"
/usr/bin/python3 - <<'PY'
from pathlib import Path
import hashlib,json
target=Path('/tmp/bt-self-flow-runtime-20261016/linux')
venv=target/'venvs/ditic'
old='/data2/ls/code/linux'
new=str(target)
cfg=venv/'pyvenv.cfg'
cfg.write_text(cfg.read_text().replace(old,new))
for p in (venv/'bin').iterdir():
 if p.is_symlink():
  link=str(p.readlink())
  if link.startswith(old+'/'):
   p.unlink();p.symlink_to(new+link[len(old):])
reference=Path('/data2/ls/checkpoints/bt_self_flow_v1_launch_20261015/nonfinite_joint_20261017/runtime_copy.json')
checks={}
for name,expected in json.loads(reference.read_text())['hash_checks'].items():
 actual=hashlib.sha256((target/name).read_bytes()).hexdigest()
 assert actual==expected,(name,actual,expected)
 checks[name]=actual
(target.parent/'runtime_copy.json').write_text(json.dumps({'runtime':str(target),'hash_checks':checks,'source_image_unchanged':True},indent=2)+'\n')
print(json.dumps({'runtime_checks_passed':len(checks)}),flush=True)
PY
LD_LIBRARY_PATH= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 "$sf_runtime_mount/venvs/ditic/bin/python" - <<'PY'
import sys,torch
print({'python':sys.executable,'torch':torch.__version__,'cuda':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0)},flush=True)
PY
