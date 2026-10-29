"""Pinned CPU analysis of saved Kodak24 actual total-rate validation curves."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

OUT = Path(__file__).resolve().parent
RUN = Path('/data2/ls/checkpoints/c_quant_residual_sf_stable_20261026')
HISTORY = Path('/data2/ls/checkpoints/bt_self_flow_stopped_eval_20261022/comparison.json')
METRICS = ('psnr', 'lpips', 'dists')
RATES = (.04, .06, .08)
LABELS = {'baseline': '512 baseline / 200K', 'old_c': 'Previous C / 95.5K',
          'old_cd': 'Previous C+D / 96.8K', 'prefix': 'Current pure C / 10K',
          'current': 'Current C+residual SF / 18K', 'old_sf18': 'Previous C+D+SF / 18K',
          'old_sf32': 'Previous C+D+SF / 32K'}
SOURCES = [HISTORY]
IDENTITY = None

def validation(path):
    global IDENTITY
    report = json.loads(path.read_text())
    identity = report['_validation_identity']
    assert identity['images'] == 24 and identity['crop'] == 0 and identity['actual_rate'] is True
    if IDENTITY is None:
        IDENTITY = identity
    assert IDENTITY == identity
    SOURCES.append(path)
    points = [dict(avg_bpp=r['rate_actual'], **{m:r[m] for m in METRICS})
              for q,r in report.items() if not q.startswith('_')]
    assert len(points) == 6 and all(r.get('packet_forward_max_error', 0) == 0
                                   for q,r in report.items() if not q.startswith('_'))
    return points, float(np.mean([r['loss'] for q,r in report.items() if not q.startswith('_')]))

def interpolate(points, metric, rate):
    points = sorted(points, key=lambda p:p['avg_bpp'])
    rates = np.array([p['avg_bpp'] for p in points])
    assert np.all(np.diff(rates) > 0) and rates[0] <= rate <= rates[-1]
    return float(PchipInterpolator(np.log(rates), [p[metric] for p in points], extrapolate=False)(np.log(rate)))

history = json.loads(HISTORY.read_text())
curves = {key:history['curves'][old] for key,old in
          (('baseline','Baseline 512 / 200K'), ('old_c','C'), ('old_cd','D'))}
trend = {}
for step in (10000,12000,14000,16000,18000):
    points, loss = validation(RUN/f'joint/validation_{step:07d}.json')
    trend[str(step)] = dict(mean_validation_rd=loss, matched_rate={str(rate):{m:interpolate(points,m,rate)
                             for m in METRICS} for rate in RATES})
    if step == 10000: curves['prefix'] = points
    if step == 18000: curves['current'] = points
for step,key in ((18000,'old_sf18'),(32000,'old_sf32')):
    curves[key] = validation(Path('/data2/ls/checkpoints/cd_sf_after10k_b4a2_20261023')/f'joint/validation_{step:07d}.json')[0]
matched = {str(rate):{key:{m:interpolate(points,m,rate) for m in METRICS}
                     for key,points in curves.items()} for rate in RATES}
logs=[];events=[]
for line in (RUN/'logs/joint.log').read_text().splitlines():
    try: row=json.loads(line)
    except ValueError: continue
    if not isinstance(row,dict): continue
    if 'step' in row and 'grad_norm' in row:logs.append(row)
    if any(k in row for k in ('nonfinite_forward','amp_skipped_update','fp32_replay_completed')):events.append(row)
limits = [
    'Matched-rate points use PCHIP in log actual total packet bpp, with no extrapolation; these are interpolated, not additional model evaluations.',
    'Historical baseline/C/C+D have different training budgets, warmups, losses, and evaluator implementations. These are reference checkpoint comparisons, not controlled ablations.',
    'Historical C/C+D losses use 2/1/0.5 and both base/repaired terms; current uses 4/1/1 and only base.',
    'Current validation uses FP32 deployment EMA, Kodak24 full resolution, six qualities, actual packet accounting on all images; packet decode versus hard forward is checked on two images per quality.',
    'Historical metrics use pyiqa PSNR/torchmetrics Alex LPIPS; current uses MSE-derived RGB PSNR/upstream Alex LPIPS. Both use pyiqa DISTS.',
    '10K pure C to 18K SF includes 8000 additional joint optimization steps. No matched-budget SF-off control has been trained; improvement is not proof of causal SF gain.',
    'Most of the current advantage versus historical methods is already present at 10K before SF, consistent with the new warmup/frontend/system accounting for much of the gain.'
]
report=dict(created_at=datetime.now(timezone.utc).isoformat(),run=str(RUN),latest_logged_step=logs[-1]['step'],validation_step=18000,
            sf_first_update=10001,labels=LABELS,validation_identity=IDENTITY,curves=curves,matched_rate=matched,
            validation_trend=trend,latest_update=logs[-1],numerical_events=events,
            pipeline=json.loads((RUN/'pipeline.json').read_text()),
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES},limits=limits)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'matched_rate.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=('bpp','model',*METRICS));writer.writeheader()
    for rate,models in matched.items():
        for key,row in models.items():writer.writerow(dict(bpp=rate,model=LABELS[key],**row))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13.5,4.4))
colors=dict(baseline='#222222',old_c='#377eb8',old_cd='#dc9933',prefix='#969696',current='#c52b39')
keys=('baseline','old_c','old_cd','prefix','current')
low=max(min(p['avg_bpp'] for p in curves[k]) for k in keys)
high=min(max(p['avg_bpp'] for p in curves[k]) for k in keys)
grid=np.linspace(low,high,200)
for ax,metric in zip(axes,METRICS):
    for key in keys:
        ax.plot(grid,[interpolate(curves[key],metric,r) for r in grid],color=colors[key],label=LABELS[key],
                linewidth=2.3 if key=='current' else 1.6,linestyle='--' if key=='prefix' else '-')
    ax.set(xlabel='Actual total packet bpp',ylabel={'psnr':'PSNR (dB), higher better','lpips':'LPIPS, lower better','dists':'DISTS, lower better'}[metric],xlim=(low,high))
    ax.grid(alpha=.2)
fig.suptitle('Kodak24 full resolution | Current C+residual SF at 18K',fontsize=13)
fig.text(.5,.895,'Reference checkpoints have different training budgets and objectives; no controlled SF-off ablation',ha='center',fontsize=9,color='#555555')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=9)
fig.subplots_adjust(top=.8,bottom=.25,left=.06,right=.99,wspace=.3)
fig.savefig(OUT/'rd_comparison.png',dpi=160);fig.savefig(OUT/'rd_comparison.svg');plt.close(fig)
lines=['# C+residual Self-Flow progress / Kodak24','',f"Train step: {logs[-1]['step']}; latest completed validation: 18000.",'',
       '| total bpp | model | PSNR ↑ | LPIPS ↓ | DISTS ↓ |','|---:|---|---:|---:|---:|']
for rate,models in matched.items():
    for key in (*keys,'old_sf18','old_sf32'):
        r=models[key];lines.append(f"| {rate} | {LABELS[key]} | {r['psnr']:.4f} | {r['lpips']:.5f} | {r['dists']:.5f} |")
lines+=['','## Interpretation','',
        'At 0.04 bpp the current model still trails the 512 baseline on all three metrics. At 0.06 bpp PSNR and DISTS are better, while LPIPS is close but slightly worse. At 0.08 bpp all three metrics are better.',
        'All three metrics improve from the current pure-C 10K prefix to 18K at the three matched rates, but gains are modest and cannot isolate Self-Flow from continued optimization.',
        'No new forward NaNs or numerical restarts have occurred in this continuation. Two AMP gradient overflow retries lowered the scale and successfully replayed the same data.',
        '', '## Limits','', *['- '+s for s in limits]]
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(output=str(OUT),latest_step=logs[-1]['step'],validation_step=18000,matched_rate=matched,validation_trend=trend,numerical_events=events),indent=2))
