"""Authorize the reviewed code migration only after its concrete checks pass."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

root = Path('/data2/ls/checkpoints/bt_self_flow_v1')
audit = root/'runtime_speedup_20261020'
results = Path('/data2/ls/results/cd_runtime_speedup_20261020')
repo = Path('/data2/ls/code/DiT-IC-cd-speedup-20261020')
commit = 'e86a8fa2dababd11a74b51aced51c2bf30c6b1de'
tests = json.loads((results/'test_acceptance.json').read_text())
assert tests['passed'] == 30 and tests['failed'] == 0
comparison = json.loads((results/'continuation_comparison.json').read_text())
assert all(value['exact_state_equal'] for value in comparison.values())
assert all(json.loads((results/'scope_check.json').read_text())['unchanged_files'].values())
packets = {}
for arm in ('C', 'D'):
    directory = results/f'full_packet_{arm}'
    summary = json.loads((directory/'summary.json').read_text())
    rows = [json.loads(line) for line in (directory/'images.jsonl').read_text().splitlines()]
    assert summary['backend'] == arm and summary['step'] == 46016
    assert len(rows) == 6 and all(row['nfe'] == 1 and row['forward_max_error'] == 0 for row in rows)
    packets[arm] = {'packets': len(rows), 'forward_max_error': 0, 'nfe': 1}
paused = json.loads((root/'pipeline.json').read_text())
assert paused['status'] == 'paused'
assert all(job['returncode'] == 0 for job in paused['jobs'].values())
for pid in [paused['supervisor_pid'], *[job['pid'] for job in paused['jobs'].values()]]:
    assert not Path(f'/proc/{pid}').exists(), pid
git = ['git', '-c', f'safe.directory={repo}', '-c', 'core.filemode=false']
assert subprocess.check_output(git+['rev-parse', 'HEAD'], cwd=repo, text=True).strip() == commit
assert not subprocess.check_output(git+['status', '--porcelain'], cwd=repo, text=True).strip()
before = json.loads((audit/'before_manifest.json').read_text())
assert json.loads((root/'manifest.json').read_text()) == before
after = dict(before, repository=str(repo), commit=commit)

def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as target:
        target.write(json.dumps(value, indent=2)+'\n')
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

record = dict(migrated_at=datetime.now(timezone.utc).isoformat(), before=before, after=after,
              configurations_unchanged=before['configs'] == after['configs'],
              source_commits=['8644d932e8159c63627740b56da4120920403e02',
                              '5f4df50a64677d300bdc0ec75ff81623cdf2853c'],
              passed_tests=30, exact_continuation_state=True, packet_checks=packets,
              resume_points=json.loads((audit/'resume_points.json').read_text()),
              parallel_validation=False,
              reason='User requested testing and replacing old code with applicable speed optimizations.')
atomic_json(audit/'migration.json', record)
atomic_json(root/'manifest.json', after)
print('Migrated manifest; original configurations are identical. Ready to resume C/D.')
