# New C + PiT experiment

This combines precision-consistent C commit
`9d8bab744cc5f12b4e4d9610067fdb34f476b752` with the two-block PiT decoder from
`codex/c-pit-pixel-adapter-20261028`, commit
`dd3853e9e395b3715d6717ca7812b44a885b3073`.
The repository name does not name the research project.

## Architecture

The previous PiT experiment used precision-v2 C. New C is a different frontend:

| Component | Previous C + PiT | New C + PiT |
| --- | --- | --- |
| Precision planner | Analysis-feature/rate-cost router | Source-only planner before analysis |
| Analysis and synthesis | Previous transforms | Shared precision embedding, conditioning at two scales in each transform |
| SANA | Native, without precision conditioning | Precision-conditioned blocks and output head |
| PiT content | Native SANA final tokens | Final tokens from the same precision-conditioned SANA invocation |
| Packet | Previous C format | New C v3, with no additional PiT fields |

`PrecisionCPixelAdapterCodec` reuses the original PiT math, insertion point and
decoder path. It overrides the source of SANA prediction and final tokens.
`quality_conditioned_sana_forward` runs once, captures the last block before
output normalization, and retains new C's shared precision embedding and
generator conditioner. The native one-step scheduler produces the AE latent.
PiT then updates the full-resolution AE features before norm/activation/RGB.

Both the SANA conditioner and PiT consume the actual decoded local quantization
step. PiT receives the conditioned tokens, projects their content, and combines
it with the embedding of `log2(delta / delta_ref)`. There is no receiver access
to source planning, no additional side information, and one SANA call per
packet. The internal backend identifier remains `C`; the requested new D is
the PiT image decoder, selected by
`decoder.method: precision_consistent_pixel_adapter_v1`.

The original method identifier remains restricted to precision-v2. The new
identifier requires precision-v3. The six quality/rate pairs, two-scale
transforms, v3 entropy coding and native scheduler are preserved.

## Initialization and optimization

Use `configs/bt_self_flow/precision_v3/pixel_joint.yaml`. The completed new-C
50K warmup EMA initializes the codec and calibrated interface through
`--prepare-backend`. Public SANA/DC-AE, text projection, LoRA and the new
generator conditioner use the shared backend initialization. PiT uses its
isolated seed 903 and zero output projection, so its initial output matches
new C exactly. No PiT warmup is necessary because warmup optimizes latent
reconstruction and constructs no image decoder.

The source warmup used two GPUs with effective batch 16. This run starts a
new joint stage on GPU0 with batch 8, accumulation 1, effective batch 8;
it does not restore the warmup optimizer or claim exact warmup continuation.
Relocated public asset paths are accepted only when all recorded asset
hashes match. Codec architecture, LoRA ranks and quality/rate settings must
still match exactly.

Joint training uses 200K updates, the inherited six qualities and rate
coefficients, image loss weights 4 MSE / 1 LPIPS / 1 DISTS, base latent weight
0.1, and the inherited milestone learning-rate schedule. PiT has its own
optimizer group at the backend learning rate. The shared precision embedding,
generator conditioner and both PiT blocks participate in gradients, EMA and
strict checkpoint restoration. Self-Flow and its teacher remain disabled.

The complete PiT/norm/activation/RGB tail is checkpointed as one region,
with inner PiT checkpoints disabled only within that region. The FP32
perceptual objective is also checkpointed as a region. Both retain the original
arithmetic, parameters and gradients. This avoids storing all high-resolution
intermediates and permits batch 8 with activations on the 32GB RTX 5090.
Autocast weight caching is disabled for PiT recomputation. Recovery
checkpoints include model/EMA, both optimizers, scheduler/scaler, sampler
counters and per-rank RNG. SIGTERM/SIGINT request a clean paused checkpoint.

Saved-tensor CPU/GPU placement is an execution policy, so resume may change
`execution.saved_tensors` while preserving optimizer identity. The checkpoint
config still records that policy; model, batch, dataset, objective, quality
and schedule remain strict. `stride_preserving_cpu` remains available as a
memory fallback, retaining exact activation layouts.

## GPU0 performance investigation, 2026-10-29

The initial implementation globally offloaded saved tensors to CPU, including
the high-resolution decoder head and perceptual-network activations. Profiling
the real step-748 model found 17.77 GiB copied to CPU per update and roughly
1.61 seconds of combined D2H/H2D device time. Data loading took about 0.2 ms.
The PiT architecture and a second SANA invocation were not the cause.

Controlled benchmarks use the same checkpoint, optimizer, input batch, quality
sequence and BF16 policy, with two warmup updates followed by four measured
updates. Engineering updates are discarded:

| Execution | Mean seconds/update | CPU saved tensors |
| --- | ---: | ---: |
| Initial global CPU storage | 3.151 | 17.77 GiB |
| Skip resident weights/small tensors | 3.112 | 17.03 GiB |
| Complete tail/loss recomputation + CPU storage | 2.146 | 5.66 GiB |
| Complete tail/loss recomputation + GPU storage | 1.848 | 0 |

The selected policy reduces step time by 41.4% and increases throughput by
70.5%. Peak allocated GPU memory is 29.34 GiB. Batch 8, accumulation 1,
architecture, loss weights, quality/rate settings and optimizer are preserved.
The original warmup and the saved 748 training updates are reused.
The production run resumed at update 749 and saved a full recovery checkpoint
at update 760. Both raw and EMA states restored strictly; main and auxiliary
optimizer steps were 760, and both PiT blocks continued to update. Actual
logged steady updates were approximately 1.86 seconds versus 3.18 previously.

`tests/test_bt_recompute.py` compares complete decoder-tail outputs and all
input/parameter gradients in FP32 and BF16, and compares the real frozen
LPIPS/DISTS objective, metrics and image gradients. Exact continuation tests
also cover changing saved-tensor placement while retaining raw/EMA,
optimizer, scheduler and RNG state.

## Validation

The combined CPU regression completed with 79 passed and 4 skipped. Production
GPU0 validation used batch 8 and two main/auxiliary/EMA updates, with zero
initial-image error, zero standalone-packet decoding error and one SANA call
per packet. Peak allocated memory was 29.34 GiB. These checks validate the
implementation and continuation; the 200K-update research run remains active.

`tests/test_precision_pixel_adapter.py` exercises active precision-conditioned
SANA equivalence at zero PiT initialization, gradients through both PiT blocks
and the new-C precision path, six-quality v3 packet decoding by a fresh
receiver, strict architecture/state guards, hash-checked asset relocation,
and exact interrupted/uninterrupted trainer equality including optimizer,
EMA and RNG with CPU activation storage enabled.

`scripts/check_pixel_adapter.py --scope full --init-checkpoint ...` validates
the production pretrained backbone with the actual warmup handoff, two full
image-objective updates, EMA, auxiliary entropy updates, initial parity,
standalone packet parity, and peak GPU memory. Its engineering updates are
discarded before formal training begins.

Machine-specific launch commands, source hashes, test logs and reports are
stored under `/data1/ls/code/runs/newc_pit_gpu0_20261029`.
