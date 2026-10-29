#!/bin/bash
#
# Script: Run q1 ~ q4 merge_ema checkpoint evaluation on GPU 3, then generate comparison charts.
# Usage:  bash scripts/run_q14_eval.sh
#

set -euo pipefail

# ============== Config ==============
PYTHON=/root/miniconda3/envs/ditic/bin/python
GPU=6
CONFIG_PATH="configs/inference_merge.yaml"
CKPT_DIR="/data2/luosheng/hf_models/hub/DiT-IC"
IMG_PATH="/data2/luosheng/data/Datasets/Kodak"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="results/merge_compare_q14_${TIMESTAMP}"
LOG_DIR="results/logs"

export CUDA_VISIBLE_DEVICES=${GPU}

mkdir -p "${LOG_DIR}" "${RESULTS_BASE}"

echo "============================================================"
echo "  Q1-Q4 Merge EMA Evaluation"
echo "  Python:  ${PYTHON}"
echo "  GPU:     ${GPU}"
echo "  Time:    ${TIMESTAMP}"
echo "============================================================"
echo ""

# ============== Run four experiments ==============
for q in 1 2 3 4; do
    CKPT="${CKPT_DIR}/q${q}_merge_ema.pt"
    LOG="${LOG_DIR}/eval_q${q}_merge_${TIMESTAMP}.log"

    echo "------------------------------------------------------------"
    echo "[$(date '+%H:%M:%S')] Running Q${q} ..."
    echo "  Checkpoint: ${CKPT}"
    echo "  Log:        ${LOG}"
    echo "------------------------------------------------------------"

    ${PYTHON} -u compress.py \
        --config_path="${CONFIG_PATH}" \
        --codec_path="${CKPT}" \
        --img_path="${IMG_PATH}" \
        --rec_path="${RESULTS_BASE}/q${q}/rec/" \
        --bin_path="${RESULTS_BASE}/q${q}/bin/" \
        --use_merge \
        --save_img \
        --verbose_perf \
        2>&1 | tee "${LOG}"

    echo ""
    echo "[$(date '+%H:%M:%S')] Q${q} finished."
    echo ""
done

echo "============================================================"
echo "  All four experiments done. Generating comparison charts ..."
echo "============================================================"

# ============== Collect summary metrics & generate charts ==============
${PYTHON} -c "
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results_dir = 'results'
base = '${RESULTS_BASE}'
os.makedirs(base, exist_ok=True)

# ---- Find all summary files ----
summary_files = {}
for q in range(1, 5):
    pattern = os.path.join(results_dir, f'*q{q}_merge_ema*', 'summary_metrics.json')
    matches = sorted(glob.glob(pattern))
    if matches:
        latest = max(matches, key=os.path.getmtime)
        summary_files[f'Q{q}'] = latest
        print(f'Found Q{q}: {latest}')
    else:
        print(f'WARNING: No summary found for Q{q}')

if not summary_files:
    print('ERROR: No summary files found! Did the experiments complete?')
    exit(1)

# ---- Load data ----
data = {}
for label, path in summary_files.items():
    with open(path) as f:
        data[label] = json.load(f)

qs = sorted(data.keys(), key=lambda x: int(x[1]))

# ---- Extract metrics ----
def get_metric(q, key):
    return data[q].get(key, np.nan)

bpp      = [get_metric(q, 'avg_bpp') for q in qs]
psnr     = [get_metric(q, 'psnr') for q in qs]
ms_ssim  = [get_metric(q, 'ms_ssim') for q in qs]
lpips    = [get_metric(q, 'lpips') for q in qs]
dists    = [get_metric(q, 'dists') for q in qs]
clipiqa  = [get_metric(q, 'clipiqa') for q in qs]
musiq    = [get_metric(q, 'musiq') for q in qs]
niqe     = [get_metric(q, 'niqe') for q in qs]
fid      = [get_metric(q, 'fid') for q in qs]
kid_mean = [get_metric(q, 'kid_mean') for q in qs]

# ---- Plot style ----
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'legend.fontsize': 11, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})
C = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444']  # Q1-Q4 colors

def label_bars(ax, bars, vals, fmt='{:.4f}'):
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2., b.get_height(),
                    fmt.format(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

def label_scatter(ax, xs, ys, labels):
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(0, 12),
                    ha='center', fontsize=10, fontweight='bold', color='#3b82f6')

# ===== Chart 1: Rate-Distortion (BPP vs PSNR, BPP vs MS-SSIM) =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
ax.plot(bpp, psnr, 'o-', color='#3b82f6', linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
label_scatter(ax, bpp, psnr, qs)
ax.set_xlabel('BPP (bits per pixel)'); ax.set_ylabel('PSNR (dB)')
ax.set_title('Rate-Distortion: BPP vs PSNR')
ax.grid(True, alpha=0.3, linestyle='--')

ax = axes[1]
ax.plot(bpp, ms_ssim, 's-', color='#22c55e', linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
label_scatter(ax, bpp, ms_ssim, qs)
ax.set_xlabel('BPP (bits per pixel)'); ax.set_ylabel('MS-SSIM')
ax.set_title('Rate-Distortion: BPP vs MS-SSIM')
ax.grid(True, alpha=0.3, linestyle='--')

fig.suptitle('Rate-Distortion Comparison — Q1 to Q4 Merge EMA', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
p = os.path.join(base, 'chart_rd_psnr_msssim.png'); fig.savefig(p); plt.close()
print(f'Saved: {p}')

# ===== Chart 2: Perceptual Quality (LPIPS, DISTS, FID, KID) =====
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for title, vals, color, ax in [
    ('LPIPS ↓', lpips, '#ef4444', axes[0]),
    ('DISTS ↓', dists, '#f59e0b', axes[1]),
    ('FID ↓',   fid,   '#8b5cf6', axes[2]),
    ('KID ↓',   kid_mean, '#ec4899', axes[3]),
]:
    bars = ax.bar(qs, vals, color=color, alpha=0.8, edgecolor='white', linewidth=1.2)
    label_bars(ax, bars, vals, '{:.5f}')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
fig.suptitle('Perceptual Quality Metrics — Q1 to Q4 Merge EMA', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
p = os.path.join(base, 'chart_perceptual_quality.png'); fig.savefig(p); plt.close()
print(f'Saved: {p}')

# ===== Chart 3: No-Reference Quality (CLIPIQA, MUSIQ, NIQE) =====
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for title, vals, color, ax in [
    ('CLIPIQA ↑', clipiqa, '#06b6d4', axes[0]),
    ('MUSIQ ↑',   musiq,   '#84cc16', axes[1]),
    ('NIQE ↓',    niqe,    '#f97316', axes[2]),
]:
    bars = ax.bar(qs, vals, color=color, alpha=0.8, edgecolor='white', linewidth=1.2)
    label_bars(ax, bars, vals)
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
fig.suptitle('No-Reference Quality Metrics — Q1 to Q4 Merge EMA', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
p = os.path.join(base, 'chart_nr_quality.png'); fig.savefig(p); plt.close()
print(f'Saved: {p}')

# ===== Chart 4: Combined Overview (2x4 grid) =====
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
for ax, (title, vals, color) in zip(axes, [
    ('BPP',       bpp,      '#3b82f6'),
    ('PSNR ↑',    psnr,     '#22c55e'),
    ('MS-SSIM ↑', ms_ssim,  '#a855f7'),
    ('LPIPS ↓',   lpips,    '#ef4444'),
    ('DISTS ↓',   dists,    '#f59e0b'),
    ('FID ↓',     fid,      '#8b5cf6'),
    ('CLIPIQA ↑', clipiqa,  '#06b6d4'),
    ('MUSIQ ↑',   musiq,    '#84cc16'),
]):
    bars = ax.bar(qs, vals, color=color, alpha=0.85, edgecolor='white', linewidth=1)
    label_bars(ax, bars, vals, '{:.4f}')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')
fig.suptitle('Full Metric Comparison — Q1 to Q4 Merge EMA', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
p = os.path.join(base, 'chart_overview.png'); fig.savefig(p); plt.close()
print(f'Saved: {p}')

# ===== Chart 5: BPP bar chart =====
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(qs, bpp, color=C, alpha=0.85, edgecolor='white', linewidth=1.5)
label_bars(ax, bars, bpp, '{:.4f}')
ax.set_ylabel('BPP (bits per pixel)')
ax.set_title('Bitrate Comparison — Q1 to Q4 Merge EMA', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.2, linestyle='--', axis='y')
p = os.path.join(base, 'chart_bpp.png'); fig.savefig(p); plt.close()
print(f'Saved: {p}')

# ===== Save all_metrics.json =====
all_metrics = {
    'checkpoints': qs,
    'bpp':      dict(zip(qs, [float(x) if not np.isnan(x) else None for x in bpp])),
    'psnr':     dict(zip(qs, [float(x) if not np.isnan(x) else None for x in psnr])),
    'ms_ssim':  dict(zip(qs, [float(x) if not np.isnan(x) else None for x in ms_ssim])),
    'lpips':    dict(zip(qs, [float(x) if not np.isnan(x) else None for x in lpips])),
    'dists':    dict(zip(qs, [float(x) if not np.isnan(x) else None for x in dists])),
    'clipiqa':  dict(zip(qs, [float(x) if not np.isnan(x) else None for x in clipiqa])),
    'musiq':    dict(zip(qs, [float(x) if not np.isnan(x) else None for x in musiq])),
    'niqe':     dict(zip(qs, [float(x) if not np.isnan(x) else None for x in niqe])),
    'fid':      dict(zip(qs, [float(x) if not np.isnan(x) else None for x in fid])),
    'kid_mean': dict(zip(qs, [float(x) if not np.isnan(x) else None for x in kid_mean])),
}
json_path = os.path.join(base, 'all_metrics.json')
with open(json_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f'Saved: {json_path}')

# ===== Save CSV =====
csv_path = os.path.join(base, f'summary_comparison.csv')
with open(csv_path, 'w') as f:
    f.write('metric,' + ','.join(qs) + '\n')
    rows = [
        ('BPP',       bpp,      '{:.4f}'),
        ('PSNR',      psnr,     '{:.4f}'),
        ('MS-SSIM',   ms_ssim,  '{:.6f}'),
        ('LPIPS',     lpips,    '{:.6f}'),
        ('DISTS',     dists,    '{:.6f}'),
        ('CLIPIQA',   clipiqa,  '{:.4f}'),
        ('MUSIQ',     musiq,    '{:.4f}'),
        ('NIQE',      niqe,     '{:.4f}'),
        ('FID',       fid,      '{:.2f}'),
        ('KID',       kid_mean, '{:.6f}'),
    ]
    for name, vals, fmt in rows:
        line = name
        for v in vals:
            line += ',' + (fmt.format(v) if not (isinstance(v, float) and np.isnan(v)) else 'N/A')
        f.write(line + '\n')
print(f'Saved: {csv_path}')

# ===== Print summary table =====
print()
print('=' * 110)
print(f'{\"Metric\":<14}', end='')
for q in qs:
    print(f'{q:>18}', end='')
print()
print('-' * 110)
for name, vals, fmt in rows:
    print(f'{name:<14}', end='')
    for v in vals:
        s = fmt.format(v) if not (isinstance(v, float) and np.isnan(v)) else 'N/A'
        print(f'{s:>18}', end='')
    print()
print('=' * 110)
print()
print('All charts and summaries saved to: ' + base)
print('Done!')
"

echo ""
echo "============================================================"
echo "  All done!"
echo "  Charts & summary: ${RESULTS_BASE}/"
echo "  Logs:             ${LOG_DIR}/"
echo "============================================================"
