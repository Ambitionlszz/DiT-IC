# Superseded after update 13271

The active run now uses fixed SF weight 1 at /data2/ls/checkpoints/c_sf_weight1_20261026. See /data2/ls/experiments/c_sf_weight1_20261026/CURRENT_RUN.md.

# Current experiment: C+SF only

The user clarified on 2026-10-26 that pure C is already trained on another server. The earlier local alternating C/SF schedule is cancelled and must not be restarted. Pure C was never launched locally.

C+SF continues from step 10442 with the full optimizer, scheduler, scaler, two-rank RNG, calibrated SF weight and teacher EMA restored. Both GPUs run C+SF together: per-GPU batch 4, accumulation 1, effective batch 8. The cumulative training target remains 200000; there is no 12000-step segment ceiling. Model, loss, quality sampling and learning-rate schedule are unchanged.

The active supervisor is scripts/launch_quantization_residual.py, not scripts/launch_c_sf_ablation.py.

Active pipeline: /data2/ls/checkpoints/c_sf_ablation_b8_20261026/c_sf/pipeline.json

Resume audit: /data2/ls/checkpoints/c_sf_ablation_b8_20261026/c_sf_only_resume.json

Training log: /data2/ls/checkpoints/c_sf_ablation_b8_20261026/c_sf/logs/joint.log
