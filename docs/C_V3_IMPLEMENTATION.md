# New C: Precision-Consistent Generative Compression

Implemented on 2026-10-27 in branch `codex/c-v3-precision-consistent-20261027`,
based on pure C commit `073f54eced8bff354f49f3233da9dca9aee8eccf`.
This branch defines the new C. Existing C v2 experiments keep their original
worktree, configuration and packet format.

## Architecture

1. Frozen DCAE encoder produces the source latent `z`.
2. `SourcePrecisionPlanner(z, phi(global_log_step))` predicts hard eight-level
   spatial codes before `g_a`. It has no coding-feature or reference-cost input.
3. The existing q16 and 3-bit codes recover the centered local log-step field.
4. `g_a` receives this field at the source and coding scales, H/32 and H/64.
5. Quantization, hyperprior, four-group contextual entropy coding and LRP retain
   their existing shared encode/decode implementation.
6. `g_s` receives the same field at H/64 and H/32.
7. One-step SANA receives spatial modulation projected from the same codec
   precision embedding. Native time conditioning and scheduler remain in use.
8. DCAE decoder produces the image.

The precision embedding is registered only under `codec.precision_embedding`.
Analysis, synthesis, the planner and SANA call those same weights; the generator
conditioner has its own output projections and per-layer gains. There is no
second registered embedding or duplicate optimizer parameter.

New affine/modulation output projections start at zero, with generator layer
gains starting at one. The generator's initial zero condition reproduces native
SANA behavior. Encoder and decoder use the hard field in their forward passes;
STE provides the planner's training gradient.

The full generator uses the actual local step as a condition, not as diffusion
time or a multiplier on its predicted correction. No RMS map or new side map is
sent. Hyperprior and main-stream sizes may change because the representation
has changed.

## Entry Points

- Warmup: `configs/bt_self_flow/precision_v3/common.yaml`.
- Joint training: `configs/bt_self_flow/precision_v3/c_joint.yaml`.
- Evaluation: existing `eval_bt_self_flow.py`, using a v3 checkpoint/config.
- Full pretrained engineering check: existing `scripts/check_c_precision_full.py`
  with the v3 joint config. It selects only C for this architecture.

Both configs enable `precision_conditioning: true` and
`precision_consistent: true`. Omitting the latter retains legacy behavior.
The new architecture accepts only pure C, whole-image quality and the codec/joint
stages. Six quality values, rate coefficients, objectives and trainable LoRA
scope follow the approved design.

The approved single-process warmup and joint both use batch 8, accumulation 1,
effective batch 8. Warmup encodes the frozen DC-AE in batches of two, then runs
the trainable codec and its loss on the full batch of eight. Joint retains the
normal full-batch execution. A distributed launch
requires an explicit batch/accumulation configuration that preserves the
effective batch; these defaults are not a two-GPU recipe.

`C_V3_RUN_ROOT` selects storage independently of legacy `BT_RUN_ROOT`.
`DITIC_SANA_PATH`, `DITIC_TEXT_PATH` and `DITIC_DATA_ROOT` retain the existing
portable asset/data conventions.

Manual two-stage commands are:

```bash
python train_bt_self_flow.py --config configs/bt_self_flow/precision_v3/common.yaml
python train_bt_self_flow.py --config configs/bt_self_flow/precision_v3/c_joint.yaml \
  --prepare-backend checkpoints/c_precision_v3/common/complete.pt \
  --output checkpoints/c_precision_v3/backend_init.pt
python train_bt_self_flow.py --config configs/bt_self_flow/precision_v3/c_joint.yaml \
  --init-checkpoint checkpoints/c_precision_v3/backend_init.pt
```

## Protected Single-GPU Launch

`scripts/launch_c_v3_single.py` runs the approved GPU5 experiment. It requires a
passed real-backbone batch-eight CPU preflight and a clean reviewed commit.
`--replace-watcher-pid` and `--replace-holder-pid` must identify the current
single-slot holder/controller on the exact GPU UUID. Other training processes
and multi-slot controllers are rejected.

The supervisor preloads the replacement model on CPU and initializes its CUDA
context before stopping the old holder. A gated memory-only reservation claims
20 GiB immediately after the old holder exits. A real CUDA batch-eight full C
check is initialized while that reservation stays alive, then runs after its
reserved blocks are released. The guard process retains its CUDA context.
The check's model stays resident until the 20 GiB protection
has been restored. Only a passed CUDA check permits formal warmup updates.

Warmup has a 9.5 GiB allocator budget alongside the 20 GiB reservation. The
reservation performs no repeated compute. It remains alive during warmup,
calibration and CPU preparation of the EMA backend initialization. Joint loads
its model, loss, optimizers and data workers before the reservation is released
and its update gate is opened. The two training stages retain 50K/200K updates,
six qualities and the normal calibration/quality-span checks.

The run root contains `manifest.json`, `status.json`, `handover.json`, all stage
logs and preflight reports. Failed GPU checks restore protection before the
remaining CUDA worker is released. `initial_codec.pt` preserves the untrained codec
parameters for the planned matched global-condition ablation. This launcher
starts only the full C experiment. It refuses an already launched run root;
an interrupted run must be inspected before resuming.

The warmup must finish and pass the original rate/quality-span checks before
backend preparation. Old C v2 weights are not a compatible complete v3 state;
no silent checkpoint conversion is performed.

## State And Packets

New packets use BT version 3, with the existing 55-byte header layout and q16/a3
payloads. Legacy packets remain version 1. Both codec paths reject packets from
the other architecture before entropy decoding. New packets do not support D
scale maps or mixed-quality masks.

All new codec parameters are covered by the existing optimizer policy, checkpoint
state and deployment EMA. The generator conditioner is exported as `conditioner`
and must be present when restoring a complete v3 model. Stage preparation
initializes this conditioner and transfers the calibrated EMA codec. The training
contract records `bt_packet_v3_q16_a3_precision_consistent`.

## Validation

The [2026-10-30 numerical repair report](C_V3_NUMERICS_20261030.md) records the
reproduced CUDA cross-attention backward failure, its repair and live joint
continuation from the effective-batch-16 warmup experiment.

Executed on 2026-10-27: **102 distinct tests passed**:

- 12 new C v3 CPU contracts;
- 1 opt-in C v3 CUDA BF16 contract on physical GPU6;
- 26 legacy codec/backend/precision-v2/GAN regression tests;
- 63 training, evaluation, loss, Self-Flow and variable-rate contracts.

The selected legacy regression commands excluded CUDA cases and the v2-specific
two-rank case; the broader training suite exercised its existing CPU distributed
contracts. The full pretrained-backbone 512px GPU check has not been executed.

`tests/test_c_precision_v3.py` covers:

- source-only planning before analysis, without evaluating a reference prior;
- identical hard precision conditions at both analysis/synthesis scales;
- six-quality real rANS, independent decoding in a fresh process, exact integer
  symbols/CDF indices, complete byte accounting and cross-version rejection;
- production codec channels and 512x768 image geometry;
- zero generator modulation and preservation of the native scheduler;
- image gradients through the planner, shared embedding, four affine projections
  and generator conditioner;
- real small diffusers SANA/DC-AE checkpointed execution;
- optimizer coverage, conditioner EMA and strict restore;
- warmup and joint exact continuation, including EMA codec handoff;
- an opt-in CUDA BF16 update and real packet test, limited to 1 GiB of PyTorch
  allocations. `C_V3_CUDA_CHECK=1` and an explicitly selected CUDA device enable it.

The real pretrained 512px batch-eight CPU check has also passed: main and auxiliary
updates, finite gradients, actual v3 packet decoding and zero forward/packet
reconstruction difference. Publication checks cover the holder identity and
gated transfer sequence without requiring a GPU. The CUDA full-backbone check
is a mandatory first stage of the protected launch; its report is retained in
the run root.

The small real-library tests and production-channel codec test are implementation
checks. They do not measure converged RD or replace a full pretrained-backbone
512px GPU check. Engineering checks do not establish converged quality gains;
the run manifest and status file record formal execution separately.
