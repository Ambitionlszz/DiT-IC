"""Inspect metadata with PyTorch's restricted unpickler; tensor bytes stay unloaded."""

import json
from pathlib import Path
import zipfile

import torch
from torch._weights_only_unpickler import Unpickler

root = Path(__file__).resolve().parent
download = json.loads((root/'download.json').read_text())
size = download['bytes']
chunk = 16*1024*1024
count = (size+chunk-1)//chunk
sparse = Path('/tmp/c_sf_b8_metadata_sparse.pt')
with sparse.open('wb') as target:
    target.truncate(size)
    for index in (0, count-1):
        target.seek(index*chunk)
        target.write((root/'download.parts'/f'{index:05d}.part').read_bytes())


class MetadataUnpickler(Unpickler):
    def persistent_load(self, identity):
        kind, storage_type, key, location, elements = identity
        if kind not in ('storage', b'storage'):
            raise RuntimeError('unexpected persistent object')
        if not hasattr(self, 'storages'):
            self.storages = {}
        if key not in self.storages:
            dtype = storage_type.dtype
            self.storages[key] = torch.empty(elements, dtype=dtype, device='meta')._typed_storage()
        return self.storages[key]


with zipfile.ZipFile(sparse) as archive:
    name = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
    with archive.open(name) as source:
        checkpoint = MetadataUnpickler(source, encoding='utf-8').load()

record = {key: checkpoint.get(key) for key in ('method', 'schema', 'phase', 'backend', 'step', 'micro_seen',
           'attempts', 'config', 'contract', 'assets', 'source_sha256', 'scheduler', 'scaler')}
record['optimizer_groups'] = [{key: value for key, value in group.items() if key != 'params'} |
                              {'parameter_count': len(group['params'])}
                              for group in checkpoint['optimizer']['param_groups']]
record['optimizer_state_count'] = len(checkpoint['optimizer']['state'])
record['rng_ranks'] = len(checkpoint['rng_by_rank'])
record['metadata_only'] = True
(root/'source_metadata.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(dict(step=record['step'], micro_seen=record['micro_seen'], world=record['contract']['world_size'],
                      train=record['config']['train'], optimizer_groups=record['optimizer_groups'], scaler=record['scaler'])), flush=True)
