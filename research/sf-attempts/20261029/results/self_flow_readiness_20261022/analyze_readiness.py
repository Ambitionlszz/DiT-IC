"""Analyze saved validation reports and the archived Self-Flow trajectories."""
from pathlib import Path
import hashlib
import json
import math
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/data2/ls/results/cd_tensorboard_20261021')
import dashboard

OUT = Path(__file__).resolve().parent
ROOT = Path('/data2/ls/checkpoints/bt_self_flow_v1')
runs = dashboard.read_logs()
history_path = Path('/data2/ls/results/recent_experiment_summary_20261019/summary.json')
history = json.loads(history_path.read_text())
shutil.copyfile(history_path, OUT/'sources/old_self_flow_summary.json')
summary = {'current': {}, 'historical_self_flow': {}, 'protocol': {
    'current': 'Saved Kodak24 full-image deployment-EMA validation; six qualities; all actual packet bytes counted; two decoded checks per quality.',
    'historical': 'Archived A/B Self-Flow BD-rate relative to their own original baseline, not relative to RD-only continuation.',
    'source_history_sha256': hashlib.sha256(history_path.read_bytes()).hexdigest(),
    'limitations': 'No new GPU inference, training, independent test set, or statistical significance estimate.'}}

fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
for arm, color in [('C', '#2563eb'), ('D', '#ea580c')]:
    vals = runs[arm]['validation']
    latest = max(vals)
    mean = lambda step: dashboard.mean_report(vals[step])['loss']
    tail = sorted(vals)[-6:]
    recent_gain = (mean(latest)/mean(80000)-1)*100
    best_step = min(vals, key=mean)
    same_rate = {}
    for rate in (.04, .06, .08, .10):
        before = dashboard.at_rate(vals[80000], rate)
        after = dashboard.at_rate(vals[latest], rate)
        same_rate[str(rate)] = {
            'before': before, 'latest': after,
            'psnr_gain_dB': after['psnr']-before['psnr'],
            'lpips_change_percent': (after['lpips']/before['lpips']-1)*100,
            'dists_change_percent': (after['dists']/before['dists']-1)*100}
    all_recent = sorted(s for s in vals if s >= 60000)
    for step in all_recent:
        shutil.copyfile(ROOT/arm/'joint'/f'validation_{step:07d}.json',
                        OUT/'sources'/f'{arm}_validation_{step:07d}.json')
        assert vals[step]['_validation_identity'] == dict(
            dataset_fingerprint=dashboard.FINGERPRINT, images=24, crop=0, actual_rate=True)
        assert all(vals[step][str(q)]['packet_forward_max_error'] == 0 for q in dashboard.QUALITIES)
    summary['current'][arm] = {
        'latest_step': latest, 'best_validation_step': best_step,
        'loss_60k': mean(60000), 'loss_80k': mean(80000), 'loss_latest': mean(latest),
        'change_from_80k_percent': recent_gain,
        'last_six_loss': {str(s): mean(s) for s in tail},
        'last_six_fitted_loss_change_per_10k': float(np.polyfit(tail, [mean(s) for s in tail], 1)[0]*10000),
        'fixed_bpp_change_from_80k': same_rate}
    axes[0, 0].plot([s/1000 for s in all_recent], [mean(s) for s in all_recent],
                    marker='o', markersize=3, color=color, label=f'{arm}: {latest//1000}k')
axes[0, 0].set(title='Current C/D: slow improvement continues',
               xlabel='Joint optimizer updates (thousands)', ylabel='Mean six-quality validation RD loss')
axes[0, 0].legend(frameon=False)

for ax, metric in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ['psnr', 'lpips', 'dists']):
    for name, label, color in [('fixed', 'A: normal reconstruction / fixed SF weight', '#2563eb'),
                                ('mixed', 'B: mixed reconstruction', '#ea580c')]:
        rows = history['histories'][name]
        x = [r['step']/1000 for r in rows]
        y = [r['metrics'][metric]['bd_rate_percent'] for r in rows]
        ax.plot(x, y, marker='o', markersize=3, color=color, label=label)
        worst = max(rows, key=lambda r: r['metrics'][metric]['bd_rate_percent'])
        summary['historical_self_flow'].setdefault(name, {})[metric] = {
            'worst_step': worst['step'], 'worst_bd_rate_percent': worst['metrics'][metric]['bd_rate_percent'],
            'last_step': rows[-1]['step'], 'last_bd_rate_percent': y[-1]}
    ax.axhline(0, color='#64748b', linewidth=1, linestyle='--')
    ax.axvspan(4, 10, color='#fbbf24', alpha=.12)
    ax.set(title=f'Archived Self-Flow: {metric.upper()}',
           xlabel='Additional optimizer updates (thousands)', ylabel='BD-rate vs starting baseline (%)')
axes[1, 0].legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Checkpoint readiness and the measured early Self-Flow regression', x=.07, ha='left', fontsize=14)
fig.text(.07, .015, 'Historical SF panels: lower is better; zero = starting baseline. No matched RD-only continuation. C/D: saved validation, not new inference.', fontsize=8)
fig.tight_layout(rect=(.01, .045, 1, .95))
fig.savefig(OUT/'readiness_and_old_self_flow.png', dpi=180, facecolor='white')
fig.savefig(OUT/'readiness_and_old_self_flow.pdf', facecolor='white')
plt.close(fig)
progress = min(1., (90000-500)/(200000-500))
factor = .1+.9*.5*(1+math.cos(math.pi*progress))
summary['proposed_common_main_learning_rates'] = {
    'rule': '0.2 times source D learning rate at 90k; proposed, not trained',
    'codec_router': .2*5e-5*factor, 'backend': .2*1e-4*factor, 'vae_decoder': .2*1e-5*factor}
(OUT/'readiness.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
