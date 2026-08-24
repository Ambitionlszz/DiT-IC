# Variable-rate LIC insertion audit

## Decision

This change is a source-grounded adaptation of DCVC-FM quality modulation to
the existing DiT-IC latent codec. It is not a FlowCodec reproduction.

The first implementation uses six independent, identity-initialized encoder
and decoder scale vectors. This follows the current official DCVC-FM image
model, which stores one encoder and one decoder scale vector per quality index.
It deliberately does not implement the two-endpoint interpolation proposed in
the planning document because that is not the behavior of the audited image
model source.

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
| Current training reference | `src/datasets/image_dataset.py`, lines 58-62; `src/utils/common.py`, lines 166-187 |
| Local symbols | `models/latent_codec.py`: `QualityController`, `AnalysisTransform.forward`, `SynthesisTransform.forward`, `LatentCodec.forward/compress/decompress` |

FlowCodec paper arXiv:2606.21030 was checked on 2026-08-24. No official code
repository was exposed by the paper page or a targeted GitHub search, so no
FlowCodec implementation is copied or claimed.

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

- Quality indices are exactly integers `0..5`; `q0` is assigned the smallest
  distortion lambda and `q5` the largest.
- One quality is sampled on rank 0 per optimizer step and broadcast to every
  DDP rank. All accumulation micro-batches for that step reuse it.
- Both scale tables initialize to ones. At initialization all six qualities
  reproduce the legacy forward and rANS payload.
- The hyperprior, context model, Gaussian conditional, and CDF architecture are
  shared and receive no explicit quality embedding.
- The packet header carries format version and quality index. The header bytes
  are included in real BPP.
- Stage 2 loads the M0 VAE decoder LoRA structure and weights but freezes them,
  freezes codec parameters, and keeps both modules in evaluation mode. The
  fixed-text projection is also frozen; only configured DiT LoRA parameters
  train.

## Required parity and server checks

Local checks cover packet round-trip, invalid quality rejection, source/config
contracts, Python compilation, and structural freeze/sampling rules. The server
must additionally verify all six real rANS round-trips, V0 payload identity,
strict real-BPP ordering, DDP quality agreement, and byte-identical V1/P0
bitstreams on fixed image IDs.
