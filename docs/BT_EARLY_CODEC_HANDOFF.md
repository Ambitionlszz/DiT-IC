# Selected Codec Checkpoint Handoff

`scripts/launch_bt_lowrate_c.py --warmup-checkpoint /absolute/path/0030000.pt`
starts the existing C joint experiment from an explicitly selected codec update.
Use a new run root to preserve the original warmup manifest and logs.

The launcher pins the selected file path and SHA-256 in its manifest, skips
further codec training, and runs `--prepare-backend --allow-incomplete-codec`.
Preparation loads the selected raw codec weights, performs the normal endpoint
calibration using the source dataset, sample count, and seed, then checks rate
and quality span before publishing `backend_init.pt`. It does not modify the
source checkpoint or pretend that the original warmup schedule completed.

The initialization records the source hash, selected update, original planned
updates, raw-weight selection, and early-handoff status. C starts at update zero
with fresh optimizers. Its original 200,000-update schedule, low-rate quantizers,
zero auxiliary losses, per-GPU batch size 2, accumulation 2, and effective batch
8 remain unchanged. Subsequent restarts restore C's latest checkpoint.

Without the explicit selected-checkpoint option, the original complete-warmup
requirement remains in force. Failed calibration prevents C from starting.
