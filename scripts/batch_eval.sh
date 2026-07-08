#!/bin/bash
# -------------------------------------------------------------------
# Batch evaluate DiT-IC checkpoints (q1 ~ q4) on Kodak
# Usage: bash scripts/batch_eval.sh
# -------------------------------------------------------------------

set -e

# ---------- Config ----------
MODEL_DIR="/data2/luosheng/hf_models/hub/DiT-IC"
CONFIG_PATH="configs/inference_merge.yaml"
IMG_PATH="/data2/luosheng/data/Datasets/Kodak"
GT_DIR="/data2/luosheng/data/Datasets/Kodak"   # same as IMG_PATH; used by eval.evaluate
GPU_ID=6

RESULT_BASE="results/Kodak"
LOG_BASE="results/logs"
mkdir -p "$LOG_BASE"

CKPTS=("q1_merge_ema" "q2_merge_ema" "q3_merge_ema" "q4_merge_ema")

RUN_TS=$(date +%Y%m%d_%H%M%S)
export SUMMARY_FILE="$LOG_BASE/summary_${RUN_TS}.txt"
export SUMMARY_JSON="$LOG_BASE/summary_${RUN_TS}.json"
RUN_INFO="$LOG_BASE/run_info_${RUN_TS}.txt"

# ---------- Save run config ----------
{
    echo "Batch Evaluation Run Info"
    echo "========================="
    echo "Date:        $(date)"
    echo "GPU:         ${GPU_ID}"
    echo "Config:      ${CONFIG_PATH}"
    echo "Image dir:   ${IMG_PATH}"
    echo "GT dir:      ${GT_DIR}"
    echo "Model dir:   ${MODEL_DIR}"
    echo "Checkpoints: ${CKPTS[*]}"
} > "$RUN_INFO"
echo "Run info saved to: $RUN_INFO"

# ---------- Print header ----------
cat << 'EOF'
==============================================
  Batch Evaluation of DiT-IC Checkpoints
==============================================
EOF
echo "  Started at: $(date)"
echo "  Checkpoints: ${CKPTS[*]}"
echo "  Summary:     $SUMMARY_FILE"

# ---------- Main loop ----------
for ckpt_name in "${CKPTS[@]}"; do
    echo ""
    echo "=============================================="
    echo "  >>> Processing: $ckpt_name"
    echo "  Started at: $(date)"
    echo "=============================================="

    CODEC_PATH="$MODEL_DIR/${ckpt_name}.pt"
    REC_PATH="$RESULT_BASE/rec_${ckpt_name}/"
    BIN_PATH="$RESULT_BASE/bin_${ckpt_name}/"

    mkdir -p "$REC_PATH" "$BIN_PATH"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    COMPRESS_LOG="$LOG_BASE/compress_${ckpt_name}_${TIMESTAMP}.log"
    EVAL_LOG="$LOG_BASE/eval_${ckpt_name}_${TIMESTAMP}.log"

    # ------ Step 1: Compression + Built-in Evaluation ------
    CMD_COMPRESS="CUDA_VISIBLE_DEVICES=${GPU_ID} python -u compress.py \
        --config_path ${CONFIG_PATH} \
        --codec_path ${CODEC_PATH} \
        --img_path ${IMG_PATH} \
        --rec_path ${REC_PATH} \
        --bin_path ${BIN_PATH} \
        --use_ema \
        --use_merge \
        --save_img"
    echo "[$(date)] [Step 1/2] Compress + built-in eval"
    echo "  CMD: ${CMD_COMPRESS}"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -u compress.py \
        --config_path="$CONFIG_PATH" \
        --codec_path="$CODEC_PATH" \
        --img_path="$IMG_PATH" \
        --rec_path="$REC_PATH" \
        --bin_path="$BIN_PATH" \
        --use_ema \
        --use_merge \
        --save_img \
        2>&1 | tee "$COMPRESS_LOG"
    echo "[$(date)] [Step 1/2] Done."

    # ------ Step 2: External Evaluation (on saved reconstructions) ------
    CMD_EVAL="CUDA_VISIBLE_DEVICES=${GPU_ID} python -m eval.evaluate \
        --recon_dir ${REC_PATH} \
        --gt_dir ${GT_DIR}"
    echo "[$(date)] [Step 2/2] External eval"
    echo "  CMD: ${CMD_EVAL}"
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval.evaluate \
        --recon_dir "$REC_PATH" \
        --gt_dir "$GT_DIR" \
        2>&1 | tee "$EVAL_LOG" || echo "[$(date)] [Step 2/2] eval.evaluate exited with code $? (non-fatal)"
    echo "[$(date)] [Step 2/2] Done."

done

echo ""
echo "=============================================="
echo "  All checkpoints processed!"
echo "  Finished at: $(date)"
echo "=============================================="

# ---------- Summarize (using python, no jq dependency) ----------
echo ""
echo "=============================================="
echo "  Building summary ..."
echo "=============================================="

# Use quoted heredoc ('PYEOF') to prevent shell expansion of $ and {}
python << 'PYEOF'
import json, glob, os

results_dir = 'results'
ckpts = ['q1_merge_ema', 'q2_merge_ema', 'q3_merge_ema', 'q4_merge_ema']

records = []
txt_lines = []
txt_lines.append('=' * 60)
txt_lines.append('  Batch Evaluation Summary')
txt_lines.append('=' * 60)
txt_lines.append('')

for ckpt in ckpts:
    # Find the latest results dir for this checkpoint
    pattern = f'{results_dir}/inference_merge_{ckpt}_*'
    dirs = sorted(glob.glob(pattern), reverse=True)

    txt_lines.append(f'--- {ckpt} ---')

    if dirs:
        json_path = os.path.join(dirs[0], 'summary_metrics.json')
        if os.path.exists(json_path):
            txt_lines.append(f'  Source: {json_path}')
            with open(json_path) as f:
                data = json.load(f)
            data['checkpoint'] = ckpt
            records.append(data)

            keys = ['avg_bpp', 'psnr', 'lpips', 'dists', 'ms_ssim',
                    'fid', 'kid_mean', 'kid_std',
                    'clipiqa', 'musiq', 'niqe']
            for k in keys:
                v = data.get(k, 'N/A')
                if isinstance(v, float):
                    txt_lines.append(f'  {k:<16} {v:.5f}')
                else:
                    txt_lines.append(f'  {k:<16} {v}')
        else:
            txt_lines.append(f'  WARNING: summary_metrics.json not found in {dirs[0]}')
    else:
        txt_lines.append(f'  WARNING: no results dir found for {ckpt}')

    txt_lines.append('')

# Write text summary (paths passed via env from shell)
txt_path = os.environ.get('SUMMARY_FILE', 'results/logs/summary.txt')
with open(txt_path, 'w') as f:
    f.write('\n'.join(txt_lines))
print(f'Summary (text): {txt_path}')

json_path = os.environ.get('SUMMARY_JSON', 'results/logs/summary.json')
with open(json_path, 'w') as f:
    json.dump(records, f, indent=2)
print(f'Summary (json): {json_path}')

print()
print('\n'.join(txt_lines))
PYEOF
