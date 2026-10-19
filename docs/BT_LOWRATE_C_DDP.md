# Low-rate C without auxiliary losses, on two GPUs

This is one experiment. Both phases use the same two physical GPUs.

| Phase | Updates | Per-rank batch | Accumulation | Effective batch |
| --- | ---: | ---: | ---: | ---: |
| Latent codec warmup | 50,000 | 32 | 1 | 64 |
| C joint training | 200,000 | 2 | 2 | 8 |

The global quantization endpoints are `delta_coarse=8`, `delta_fine=1`,
with `rate_weight=3`. The C phase sets both `base_weight` and
`latent_weight` to zero. The image objective remains MSE/LPIPS/DISTS with
weights 2/1/0.5. The latent warmup retains its latent reconstruction objective.

`lowrate_common_ddp.yaml` and `lowrate_c_no_aux_ddp.yaml` select these recipes.
The launcher freezes resolved paths, configurations and the reviewed commit
in the run manifest. Restart the same launcher to resume the latest saved
update. Both training stages use `torchrun` with two workers; preparation of
the C initialization uses one worker after the codec calibration passes.

## Validation and performance

Training progressively enables the Router after 2,000 updates and finishes
the ramp after another 3,000. Validation temporarily enables the deployed
Router for both hard reconstruction and actual rANS coding, then restores
the raw training state, even on failure. Compression still rejects a partially
enabled Router outside that explicit validation context. Reports record the
deployment Router policy, so selection cannot mix old validation scores.

The new recipes split validation images between ranks, retaining all six
quality points, actual packet bytes, and forward/decode checks. Aggregation
weights normal metrics by image count, packet metrics by checked packet count,
and takes the maximum forward/decode error. A validation exception is shared
with all ranks. CUDA EMA updates batch small tensors into foreach kernels;
the CUDA regression checks exact agreement with the original EMA update.
Training logs include data-wait, update, validation and checkpoint durations.
Pause and joint-completion aliases reuse the last unchanged checkpoint with
an atomic hardlink (a durable copy on filesystems without hardlinks). Codec
endpoint calibration always writes a fresh checkpoint with calibrated buffers.

## Verification and launch

Use the path overrides in `BT_GPU_PORTABILITY.md`. Also point `TMPDIR` at
a writable data volume and use the existing perceptual-weight cache.

```bash
python scripts/verify_bt_lowrate_c.py \
  --root /path/to/engineering_checks --production-root /path/to/research_run \
  --gpus GPU-uuid-a,GPU-uuid-b
```

This runs real 512px public backbones with the production batch sizes,
shortened schedules, two-rank pause/resume in both stages, codec calibration,
stage handoff, full-image validation and real packets. Engineering checkpoints
are isolated from research outputs. A per-commit `passed.json` records the
scope and measured throughput; short runs do not certify final research
quality or claim that 250,000 production updates have already completed.

For protected execution, run `guard_bt_training.py` with the verification
command after `--`. Supply the existing `train_depart.py` as `--holder-script`.
`--defer` waits for SIGUSR2, allowing a controlled handover from old holders.
The optional verification `--fault-drill` deliberately fails real CUDA ranks
once per commit. Verify both replacement holders before resuming verification.

The guardian owns an independent pipeline process group. A failed rank,
failed stage, killed pipeline supervisor, or normal pipeline exit causes
cleanup of that group and reservations on the exact selected GPU UUIDs.
Each holder reserves 27 GiB; failed holders are retried. A GPU occupied by an
unrelated CUDA allocation is left alone and retried when idle. Holder creation
falls back to discarding output if the experiment disk cannot create logs.

`guard.json` records the guardian, pipeline and holder PIDs. SIGTERM requests
a boundary-safe training pause and retains the GPUs; SIGUSR2 releases the
holders and resumes the pipeline; SIGUSR1 explicitly releases and exits.
The guardian itself must remain alive. Machine/container shutdown or killing
the guardian cannot be repaired by that same process.

After reviewing successful engineering checks and the reservation drill,
create `production.start` in the research run directory and signal SIGUSR2.
The verified wrapper then starts `launch_bt_lowrate_c.py`. It automatically
advances from the 50k codec endpoint to the C initialization and 200k joint run.
