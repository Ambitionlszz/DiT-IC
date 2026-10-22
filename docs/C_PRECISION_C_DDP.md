# Precision C on Two GPUs

This launcher trains only C. The inherited precision v2 objectives, quality
points, learning rate schedule and update counts are retained. Warmup uses
batch 8 per rank and effective batch 16 for 50K updates. Joint C uses batch 2
per rank, accumulation 2 and effective batch 8 for 200K updates.

`scripts/launch_c_precision.py` reserves 18 GiB on each GPU during warmup
with idle memory allocations. It checks the completed warmup's rate and
quality spans, calibrated EMA, assets and exact shared codec initialization
before releasing both reservations and starting C with two ranks. A failed
check stops the pipeline without starting joint C. Logs and pipeline.json
record the stage, process IDs and errors. Repeating the same command resumes
the latest published checkpoint and checks the frozen run manifest.

Set DITIC_DATA_ROOT, DITIC_SANA_PATH, DITIC_TEXT_PATH and TORCH_HOME to the
host's existing data, pretrained assets and perceptual weight cache. Run in
tmux with a new root outside the source worktree:

```bash
python scripts/launch_c_precision.py --root /path/to/new/run --gpus 0,6
```

The same command with `--smoke` uses 14 updates per stage, pauses and resumes
both stages, four calibration samples and full Kodak validation. Smoke
disables only the quality-span gate because it is not a convergence test.
Production keeps the gate enabled and uses 512 calibration samples.

Send SIGTERM to the supervisor PID in pipeline.json to pause at a completed
update boundary. It signals the two trainers and lets their data workers and
torchrun launcher remain alive until checkpoint saving completes. Reservation
processes exit on supervisor shutdown and also check their parent's lifetime.

This recipe uses rate coefficients [16, 8, 4, 2, 1, 0.5], from low to high q,
in both warmup and joint C. A fresh output root starts both phases from zero.

To replace a running precision warmup, use --replace-run /path/to/previous/run.
The new reservation processes initialize CUDA before the old holders stop.
Each GPU's holder is replaced while the previous trainer still owns its CUDA
memory. Once both new 18 GiB reservations are ready, the previous supervisor
is asked to save and stop. The new training starts only after the previous
trainers exit. handover.json records the checked process identities and status.
Repeating the same new-run command after a completed handover resumes normally.
