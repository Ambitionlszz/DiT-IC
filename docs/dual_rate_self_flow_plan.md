# Dual-Rate Self-Flow Post-Training Plan

## Mechanism

For every image after the 2k-step continuation warmup:

```text
same image/crop/augmentation
            |
      frozen VAE encoder
            |
        clean latent z
          /       \
 frozen codec   frozen codec
   q_a view       q_b view        q_a, q_b independently sampled
          \       /
       spatial token mask         25% selects view B
              |
       one mixed latent grid
              |
     one complete Student pass    token-wise q condition at all 28 blocks
          /             \
 layer 8 feature       reconstruction
          |                  |
 Student projector     MSE + LPIPS + 0.2 DISTS
          |
 all-token cosine loss <--- EMA Teacher layer 20
                              ^
                              |
                    uniform q_H codec view
```

The Student runs once. The Teacher is a separate no-gradient EMA copy and is
truncated after layer 20. The mixed latent is a training view, not a legal
bitstream. Both source views are legal codec forwards and their likelihood BPP
is logged separately. Evaluation and deployment use one legal uniform-q stream.

## Pair Sampling

```text
q_a ~ Uniform{0, 1, 2, 3, 4, 5}
q_b ~ Uniform{0, 1, 2, 3, 4, 5}, independently
q_H = max(q_a, q_b)
```

This covers all 36 ordered rate pairs uniformly and produces same-rate pairs
with probability 1/6. Both source marginals remain uniform. A spatial mask
chooses A or B without sorting the inputs, and all channels of one latent cell
come from the same source. SANA has `patch_size=1`, so one latent cell is one
DiT token.

## Schedule

| Optimizer steps | Student view | q condition | Representation loss | Teacher |
| --- | --- | --- | --- | --- |
| 0-1999 | Uniform batch-shared q | Off | Weight 0 | EMA runs |
| 2000-6999 | IID dual-rate mix, 0 to 25% | 0 to 1 | 0 to calibrated gamma | Synced at 2000, then EMA |
| 7000-99999 | IID dual-rate mix, 25% | 1 | Calibrated gamma | EMA 0.9999 |

The run starts from the P0 checkpoint's EMA model weights. Optimizer,
scheduler, counters, and deployment EMA are reset. Gamma is capped at 0.1 and
is reduced when its initial layer-8 gradient would exceed 20% of the base-loss
gradient. Values below 0.01 fail instead of silently relying on gradient
clipping.

## Fair Ablations

All configs inherit `dual_rate_self_flow_full.yaml`. They share the source
checkpoint, datasets, augmentation, seed 903, effective batch 16, 100k optimizer
steps, reconstruction objective, optimizer hyperparameters, LR milestones,
evaluation interval, and checkpoint interval.

| Config | token q | mixed Student | high-q Teacher | representation |
| --- | ---: | ---: | ---: | ---: |
| posttrain control | no | no | no | no |
| q only | yes | no | no | no |
| mix only | yes | yes | no | no |
| distill only | yes | no | yes | yes |
| same-rate Teacher | yes | no | no | yes |
| full | yes | yes | yes | yes |

The main scientific run is the full row. A two-step run with compressed warmup
is only a DDP/CUDA infrastructure check; it is not reported as an experiment.

## Evaluation Contract

For q0 through q5, run actual `compress()` and `decompress()` separately per
image and q. Report PSNR, MS-SSIM, LPIPS, DISTS, and real rANS BPP for raw and
deployment-EMA Student weights. The P0 and Self-Flow evaluation use identical
source images and codec bitstreams; the frozen-codec hash is stored in every
checkpoint. The primary comparison is the six-rate macro average, with q0/q1
reported explicitly.
