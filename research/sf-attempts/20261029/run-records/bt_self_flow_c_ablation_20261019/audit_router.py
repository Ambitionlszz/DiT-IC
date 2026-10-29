"""Read actual transmitted allocation codes without loading a GPU model."""
import collections
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

repo = Path('/data2/ls/code/DiT-IC-bt-self-flow')
spec = importlib.util.spec_from_file_location('bt_packet_audit', repo/'models/bt_self_flow/packet.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
root = Path('/data2/ls/checkpoints/bt_self_flow_metrics_20261019')
result = {}
for backend, folder in [('C', 'C_36000'), ('D', 'D_38000_gpu0')]:
    rows = []
    for path in sorted((root/folder/'q00000').glob('*.btic')):
        packet = module.BTImagePacket.from_bytes(path.read_bytes())
        yh, yw = packet.y_shape
        assert (yh, yw) in ((8, 12), (12, 8))
        codes = module.unpack_codes(packet.allocation, 3, yh*yw)
        histogram = collections.Counter(codes)
        mean = statistics.mean(codes)
        factors = [math.exp(-math.log(2)*(code-mean)*2/7) for code in codes]
        identical_qualities = []
        for directory in sorted((root/folder).glob('q*')):
            other = module.BTImagePacket.from_bytes((directory/path.name).read_bytes())
            identical_qualities.append(other.allocation == packet.allocation)
        assert len(identical_qualities) == 6 and all(identical_qualities)
        rows.append(dict(image=path.stem, y_shape=[yh, yw], unique=len(histogram),
                         histogram=dict(histogram), uniform=len(histogram) == 1,
                         std_code=statistics.pstdev(codes), min_factor=min(factors),
                         max_factor=max(factors), max_step_ratio=max(factors)/min(factors),
                         codes=codes, allocation_identical_across_six_qualities=True))
    assert len(rows) == 24
    histogram = collections.Counter(code for row in rows for code in row['codes'])
    result[backend] = dict(images=24, tokens_per_image=96,
                           uniform_images=sum(row['uniform'] for row in rows),
                           nonuniform_images=sum(not row['uniform'] for row in rows),
                           mean_unique=statistics.mean(row['unique'] for row in rows),
                           mean_spatial_std_code=statistics.mean(row['std_code'] for row in rows),
                           mean_max_step_ratio=statistics.mean(row['max_step_ratio'] for row in rows),
                           max_step_ratio=max(row['max_step_ratio'] for row in rows),
                           histogram=dict(histogram), images_detail=rows)
    print(backend, {key: value for key, value in result[backend].items() if key != 'images_detail'}, flush=True)
    print('per_image', [(r['image'], r['unique'], round(r['max_step_ratio'], 3)) for r in rows], flush=True)
Path(__file__).with_name('router_packet_audit.json').write_text(json.dumps(result, indent=2)+'\n')
