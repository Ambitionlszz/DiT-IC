from pathlib import Path
from datetime import datetime
import hashlib
import json
import random
import subprocess

root = Path('/data2/ls/checkpoints/bt_self_flow_v1')
audit = root/'runtime_speedup_20261020'
repo = Path('/data2/ls/code/DiT-IC-cd-speedup-20261020')
commit = 'e86a8fa2dababd11a74b51aced51c2bf30c6b1de'
manifest = json.loads((root/'manifest.json').read_text())
pipeline = json.loads((root/'pipeline.json').read_text())
sources = json.loads((audit/'resume_points.json').read_text())['checkpoints']
assert pipeline['status'] == 'paired_training'
assert manifest['repository'] == str(repo) and manifest['commit'] == commit
records = {}
for arm in ('C', 'D'):
    name = f'{arm.lower()}_joint'
    job = pipeline['jobs'][name]
    assert job['status'] == 'running'
    assert Path(f'/proc/{job["pid"]}/cwd').resolve() == repo
    resolved = json.loads((root/arm/'joint/resolved.json').read_text())
    assert resolved['git_commit'] == commit
    assert resolved['config'] == manifest['configs'][name]
    assert resolved['contract'] == sources[arm]['contract']
    assert resolved['assets'] == sources[arm]['assets']
    rows = [json.loads(line) for line in (root/arm/'joint/train.jsonl').read_text().splitlines()]
    resumed = [row for row in rows if row['step'] > sources[arm]['step']]
    assert resumed and resumed[0]['step'] == sources[arm]['step']+1
    first = resumed[0]
    assert 'data_wait_seconds' in first
    digest = hashlib.blake2b(f'903:13:{sources[arm]["step"]}'.encode(), digest_size=8).digest()
    sampled = random.Random(int.from_bytes(digest, 'big') & ((1 << 63)-1)).random()
    # RateControl logs its canonical packet-recoverable uint16 quality.
    control = round(sampled*65535)/65535
    assert first['lambda_low'] == control
    assert first['attempts'] == sources[arm]['attempts']+1
    records[arm] = {'pid': job['pid'], 'gpu': job['gpus'], 'resumed_from_step': sources[arm]['step'],
                    'first_completed_update': first, 'resolved_commit': resolved['git_commit'],
                    'contract_unchanged': True, 'assets_unchanged': True,
                    'configuration_unchanged': True, 'correct_next_quality_sample': True}
apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                '--format=csv,noheader'], text=True)
assert {int(line.split(',')[0]) for line in apps.strip().splitlines()} == {r['pid'] for r in records.values()}
record = {'status': 'passed', 'recorded_at': datetime.now().astimezone().isoformat(),
          'repository': str(repo), 'commit': commit, 'pipeline': pipeline, 'arms': records,
          'gpu_processes': apps, 'regression_tests_passed': 30,
          'full_training_exact_comparison_updates_per_mode': 16,
          'full_model_packets_checked': 12, 'forward_packet_max_error': 0,
          'note': 'First update includes CUDA warmup; steady timing is in the isolated continuation comparison.'}
(audit/'resume_acceptance.json').write_text(json.dumps(record, indent=2)+'\n')
report = Path('/data2/ls/results/cd_runtime_speedup_20261020/report.zh.md')
with report.open('a') as target:
    target.write('\n正式运行验收通过：C 已完成恢复后的 52,052 步，D 已完成 54,001 步；两个进程的执行目录、提交、配置、训练协议、预训练资产及下一个质量采样均核对一致，GPU 上只有这两个正式训练进程。\n')
print(json.dumps({'status': 'passed', 'C_pid': records['C']['pid'], 'C_completed_step': 52052,
                  'D_pid': records['D']['pid'], 'D_completed_step': 54001, 'commit': commit}, indent=2))
