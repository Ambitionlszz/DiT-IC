import hashlib
import json
from pathlib import Path
import time

from huggingface_hub import HfApi, hf_hub_download

repo = "Ambitionls/ditic-50k-warmups"
revision = "3e96cf9492dd5ce86a3fc72f6aa4c9269209f531"
filename = "c_precision_v2_lambda16_C_joint10k_20261023/0010000.pt"
token = Path("/home/ls/.cache/huggingface/token").read_text().strip()
api = HfApi(token=token)
info = api.model_info(repo, revision=revision, files_metadata=True)
if info.sha != revision:
    raise RuntimeError("Hub returned a different repository revision")
entry = next(item for item in info.siblings if item.rfilename == filename)
expected = entry.lfs.sha256
record = dict(repo=repo, revision=revision, filename=filename, bytes=entry.size, expected_sha256=expected,
              status="downloading", started_at=time.time())
manifest = Path(__file__).with_name("download.json")
manifest.write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record), flush=True)
path = hf_hub_download(repo, filename, revision=revision, token=token,
                       local_dir="/data2/ls/checkpoints/ditic-50k-warmups")
digest = hashlib.sha256()
with open(path, "rb") as source:
    for block in iter(lambda: source.read(8*1024*1024), b""):
        digest.update(block)
actual = digest.hexdigest()
if actual != expected or Path(path).stat().st_size != entry.size:
    raise RuntimeError("checkpoint size/hash differs from pinned Hub metadata")
record.update(status="verified", path=path, actual_sha256=actual, completed_at=time.time())
manifest.write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record), flush=True)
