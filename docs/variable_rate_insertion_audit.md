# Variable-rate LIC insertion audit

## Decision

This change migrates the complete variable-rate control path described by
FlowCodec into the existing DiT-IC latent codec. FlowCodec cites DCVC-FM for
this mechanism. The migration contains both parts used by the method:

1. one learned per-channel analysis/synthesis scale pair `q_e/q_d` per quality;
2. content-adaptive quantization maps `w_e/w_d`, predicted jointly with the
   first entropy-prior mean and scale from the decoded hyperprior.

The surrounding DiT-IC analysis/synthesis blocks, four-step context model, and
LRP remain the local baseline architecture. FlowCodec has no public reference
implementation, so this is a formula- and upstream-source-faithful mechanism
migration, not a claim that the entire unpublished FlowCodec codec is copied.

## Upstream source map

| Field | Value |
|---|---|
| Upstream | `https://github.com/microsoft/DCVC.git` |
| Audited commit | `819c219b24db34310bbd15c51a720aaaf5eb2e7d` |
| License | MIT; `LICENSE.txt` permits use, modification, and distribution with the copyright/license notice retained |
| Quality interpolation reference | `DCVC-family/DCVC-FM/src/models/common_model.py`, `CompressionModel.get_one_q_scale/get_curr_q`, lines 37-51 |
| Image quality tables and routing | `DCVC-family/DCVC-FM/src/models/image_model.py`, `DMCI`, lines 100-125 and 163-224 |
| Encoder boundary | `DCVC-family/DCVC-FM/src/models/image_model.py`, `IntraEncoder.forward`, lines 31-34 |
| Decoder boundary | `DCVC-family/DCVC-FM/src/models/image_model.py`, `IntraDecoder.forward`, lines 54-57 |
| Quantization-control split | `DCVC-family/DCVC-FM/src/models/common_model.py`, `CompressionModel.separate_prior`, lines 106-116 |
| Four-part quantization order | `DCVC-family/DCVC-FM/src/models/common_model.py`, `forward_four_part_prior/decompress_four_part_prior`, lines 189-244 and 253-290 |
| Current training reference | `src/datasets/image_dataset.py`, lines 58-62; `src/utils/common.py`, lines 166-187 |
| Local symbols | `models/latent_codec.py`: `QualityController`, `AnalysisTransform.forward`, `SynthesisTransform.forward`, `LatentCodec.forward/compress/decompress` |

FlowCodec paper arXiv:2606.21030 was checked on 2026-08-24. No official code
repository was exposed by the paper page or a targeted GitHub search, so no
FlowCodec implementation is copied or claimed.

## Quantization-domain migration

For the variable-rate codec, the first prior head emits `2M+2` channels. The
first two channels are converted exactly as in the DCVC-FM image model:

```text
(w_e, w_d) = split(sigmoid(raw_q) * 1.5 + 0.5)
```

The remaining `2M` channels are the first group's Gaussian mean and scale.
All four masked groups encode `y * w_e`; their reconstructed groups and LRP
corrections stay in that same quantization domain. After the four groups are
combined, the result is multiplied by `w_d` and passed to `g_s(..., q_d)`.
The decoder regenerates `w_d`, means, and scales from decoded `z_hat`, so the
maps are not side information and add no uncounted bytes.

This path is identical in differentiable `forward`, real `compress`, and real
`decompress`. Fixed-rate construction keeps the original `2M` prior head and
does not activate any of these extra channels.

## Current codec boundary audit

### Analysis transform

| Boundary | Shape | Decision |
|---|---|---|
| VAE latent input | `[B, 32, H, W]` | no modulation |
| `g_a.pre1` output | `[B, 192, H, W]` | multiply by `encoder_scale` |
| analysis output `y` | `[B, 320, H/2, W/2]` | entropy path input |

DCVC-FM multiplies the feature after its first encoder feature extractor and
before the remaining analysis body. `g_a.pre1` is the closest local functional
boundary and preserves all existing state-dict keys.

### Synthesis transform

| Boundary | Shape | Decision |
|---|---|---|
| quantized `y_hat` input | `[B, 320, H/2, W/2]` | no modulation |
| final upsample output | `[B, 320, H, W]` | multiply by `decoder_scale` |
| reconstructed VAE latent | `[B, 32, H, W]` | final output |

DCVC-FM multiplies the feature after its main decoder body and before its final
output block. Applying the scale after local `synthesis_transform[3]` and before
the final convolution is the closest functional match and preserves checkpoint
key names.

## Adaptations and invariants

- Quality indices are exactly integers `0..5`; under `D + lambda * R`, `q0`
  uses the largest rate lambda and `q5` the smallest.
- One quality is sampled on rank 0 per optimizer step and broadcast to every
  DDP rank. All accumulation micro-batches for that step reuse it.
- The complete LIC initializes randomly and both scale tables initialize to
  ones. At step 0 all six qualities reproduce each other; there is no legacy
  LIC or M0 checkpoint dependency.
- The hyperprior, context model, Gaussian conditional, and CDF architecture are
  shared. The quality-specific `q_e/q_d` change the features presented to those
  shared components; `w_e/w_d` then adapt quantization to the decoded content.
- Checkpoint schema 6 records
  `variable_rate_mechanism=dcvc_fm_qe_qd_we_wd_v1`. Earlier incomplete LIC
  checkpoints are rejected and cannot be silently resumed or handed to Plain.
- The packet header carries format version and quality index. The header bytes
  are included in real BPP.
- Stage 2 loads and freezes the V2 LIC, loads frozen pretrained SANA VAE and
  DiT backbones, and trains new DiT LoRA parameters plus the fixed-condition
  text projection. VAE LoRA is disabled.

## Required parity and server checks

Local checks cover packet round-trip, invalid quality rejection, source/config
contracts, Python compilation, and structural freeze/sampling rules. The server
must additionally verify all six real rANS round-trips, strict real-BPP
ordering, DDP quality agreement, and byte-identical V2/P0 bitstreams on fixed
image IDs.
