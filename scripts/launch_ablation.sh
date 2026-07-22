#!/bin/bash
# Launch two ablation experiments in tmux:
# Window 0 (GPU4): Pure SANA baseline lrd=0.4
# Window 1 (GPU6): g_s residual ablation lrd=0.4

SESSION="ablation_gs_residual"
REPO="/data2/luosheng/code/DiT-IC"
CONDA_ENV="/root/miniconda3/envs/ditic"

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/common_pure_checkpoint.pt" >&2
  exit 2
fi

INIT_CHECKPOINT="$1"
if [ ! -f "$INIT_CHECKPOINT" ]; then
  echo "Common Pure SANA checkpoint not found: $INIT_CHECKPOINT" >&2
  exit 2
fi

cd "$REPO"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null

# Create new detached session with first window (baseline on GPU4)
tmux new-session -d -s "$SESSION" -n "baseline_gpu4"
tmux send-keys -t "$SESSION:0" \
  "cd $REPO && export PATH=$CONDA_ENV/bin:\$PATH && CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. $CONDA_ENV/bin/torchrun --standalone --nproc_per_node=1 train_nogan_ddp.py --config configs/baseline_lrd0p4.yaml --frozen_decode_ablation sana --init_checkpoint \"$INIT_CHECKPOINT\" 2>&1 | tee logs/baseline_lrd0p4_\$(date +%Y%m%d_%H%M%S).log" C-m

# Create second window (ablation on GPU6)
tmux new-window -t "$SESSION" -n "gs_residual_gpu6"
tmux send-keys -t "$SESSION:1" \
  "cd $REPO && export PATH=$CONDA_ENV/bin:\$PATH && CUDA_VISIBLE_DEVICES=6 PYTHONPATH=. $CONDA_ENV/bin/torchrun --standalone --nproc_per_node=1 train_nogan_ddp.py --config configs/pure_sana_ablation_gs_residual.yaml --frozen_decode_ablation gs_residual --init_checkpoint \"$INIT_CHECKPOINT\" 2>&1 | tee logs/gs_residual_lrd0p4_\$(date +%Y%m%d_%H%M%S).log" C-m

echo "Tmux session '$SESSION' created:"
echo "  Window 0 (baseline_gpu4):  Pure SANA lrd=0.4 on GPU 4"
echo "  Window 1 (gs_residual_gpu6): g_s residual lrd=0.4 on GPU 6"
echo ""
echo "Attach:  tmux attach -t $SESSION"
echo "List:    tmux list-windows -t $SESSION"
