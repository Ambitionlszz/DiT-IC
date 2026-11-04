# Native-endpoint Self-Flow and matched C/D ablations

This implementation replaces the intermediate DiT projector with a
training-only objective on the actual repaired latent. The student uses only
its native low-quality C input. A frozen deployment-EMA C/SANA teacher uses the
next configured quality and completes its own native one-step repair. Both
endpoints are measured through the same current student AE decoder prefix.
There is no new learned head, projector, sampler, side information, or inference
computation. No new full-model GPU validation or RD result is claimed here.

The EMA/self-distillation starting point comes from
[Self-Flow](https://arxiv.org/abs/2603.06507). The use of decoder intermediates
as a reconstruction-relevant feature space is informed by
[Latent Perceptual Loss](https://arxiv.org/abs/2411.04873). This implementation
uses neither paper unchanged: it compares native repaired endpoints at adjacent
compression qualities, through one current AE prefix. Those references do not
establish that this particular cross-rate loss improves compression RD.

## Actual reconstruction and supervision

```text
student C(qL) -> muL, deltaL -> student SANA prediction + native scheduler -> zL
                                                                      |
                    current AE prefix, through up_blocks[3] -> FL -----+--> AE remainder -> D -> RGB
                                                               |
                                                       shared-scale L1
                                                               |
EMA C(qH) -> muH, deltaH -> EMA SANA + its native scheduler -> zH
                                                               |
                           SAME current student AE prefix -> stopgrad(FH)
```

The displayed feature branch is a capture from the actual student AE decode;
it does not rerun the student decoder. The target needs one additional AE prefix
evaluation under `no_grad`. In the six-stage f32 DC-AE, capture is immediately
after zero-based `decoder.up_blocks[3]`, at image resolution H/8 by W/8. Existing
AE latent scaling is applied to both endpoints before decoding. This feature is
before the full-resolution D correction.

The teacher predicts an endpoint, not an update to borrow: `zH =
scheduler.step(predictionH, muH)`. The student keeps `zL =
scheduler.step(predictionL, muL)`. High-rate latent, precision, hidden states,
and target features never enter the student reconstruction path. This avoids
adding a high-rate repair increment to a different low-rate starting point.

The teacher is the same model's deployment EMA for C, SANA, its output adapters,
text projection and precision conditioning. It is always eval/no-grad. It runs
its own quality-dependent analysis and native quantization; high and low inputs
are not assumed to be nested quantizations. It has no independent optimizer or
separately trained history. The metric AE prefix is the current student AE on
both branches, rather than a second EMA AE with different feature coordinates.

## One spatially scaled feature L1

For BCHW decoder features, calculate per-image/per-channel population statistics
over spatial positions from `FL` only:

```python
reference = FL.float().detach()
mean = reference.mean((-2, -1), keepdim=True)
variance = (reference - mean).square().mean((-2, -1), keepdim=True)
scale = (variance + 1e-6).sqrt()
L_SF = ((FL.float() - FH.detach().float()).abs() / scale).mean()
L_total = L_original_RD + L_SF
```

This is equivalently the L1 distance between `(FL-mean)/scale` and
`(FH-mean)/scale` using the **same detached student statistics**. Their shared
mean cancels exactly. The operation is therefore an inverse-spatial-standard-
deviation weighted AE feature L1, not independent standardization of the two
branches, not inverse variance, and not a cosine objective. Teacher/student
mean and amplitude differences are retained; independent positive affine
changes of the two features do not leave this objective invariant. Student
statistics are detached, so the current update cannot reduce the loss by
differentiating through a larger denominator.

There is exactly one additional SF loss, with coefficient one at every active
update. There is no ramp, decay, per-image score, reliability gate, gradient
projection, or claim of automatic selection of low-rate learnable knowledge.
At the highest quality, all target variants use the original RD objective only.
Thus the target controls have exactly the same SF-active quality set as the
adjacent-rate method; inactive top-quality steps are not a loss-weight schedule.

Only existing trainable SANA parameters receive the extra SF gradient, including
any trainable part of its actual output head. The prefix must preserve input
gradients so SF reaches the real repaired latent and SANA. C, shared precision
embedding, text/conditioning modules, AE and D receive their original RD
gradients only. Applying SF by unrestricted `total.backward()` would violate
that parameter policy. Target values and normalization statistics are detached.
Normal optimizer clipping and distributed averaging still apply to the final
combined gradients, so this is not a claim of completely isolated optimization.

The current AE changes under RD training, so it is a moving metric despite
receiving no direct SF parameter gradient. Small spatial variance can amplify
the supervised error even with a finite epsilon. Log the minimum/median scale,
the fraction of near-floor channels, their contribution to SF, the SF magnitude,
and the actual latent and image metrics. These measurements do not gate or
reweight the loss. No guarantee of noncollapse, useful teacher details, or joint
perceptual/structural improvement follows from normalization alone.

## Six matched recipes

| Recipe | D | SF | Target, below top quality |
| --- | --- | --- | --- |
| `c` | absent | off | none |
| `c_d` | precision mu-QK PiT v4 | off | none |
| `c_sf` | absent | on | native EMA endpoint at adjacent higher quality |
| `c_d_sf` | precision mu-QK PiT v4 | on | native EMA endpoint at adjacent higher quality |
| `c_d_same_rate_sf` | same D | on | native EMA endpoint at the student's quality |
| `c_d_gt_sf` | same D | on | frozen-encoder clean AE latent |

The first four recipes are the primary factorial ablation; the last two are
optional attribution controls. All SF target kinds skip quality 1.0. For the
main method `qL=[0,.2,.4,.6,.8]` maps deterministically to
`qH=[.2,.4,.6,.8,1]`, consuming no teacher sampling RNG. Student quality sampling
and crops follow the same stream in all six recipes. Clean-latent control is
not assumed optimal: after AE/D training, clean encoder coordinates need not be
the best decoder input. It tests a distinct reference, not ground-truth teacher
quality filtering.

For each D setting, SF-on and SF-off have identical deployment architecture,
trainable parameter capacity, and initialization. There is no P-only control
because there is no P. Teacher and target prefix are training overhead only;
report that overhead separately. D has the same parameters, seed, zero-output
initialization, and optimizer policy in every D recipe. Initial reconstruction
must match between the four primary groups before any update.

The standalone common YAML snapshots the original precision-v3 joint settings
while removing the old inherited SF layer/projector fields. All groups use
200K successful joint updates, crop 512, two ranks, batch four per rank,
accumulation one, effective batch eight, and the same original RD losses and
milestone learning rates. Use at least two paired seeds for the final result.
SF is active throughout joint training below top quality; it is not a second
training stage after ordinary reconstruction training.

## Shared initialization and launch contract

All six groups must initialize from the **same pure-C `backend_init.pt`**,
prepared from a completed, calibrated common codec warmup. Starting from a
trained C+D checkpoint would leave D adaptation in the shared backbone and is
rejected. The source must retain calibrated distortion scale and passed rate-
and quality-span checks, and contain neither D nor projector weights. Reusing a
previous compatible calibrated C warmup is valid; do not repeat it merely to
introduce SF.

If a compatible backend initialization does not already exist, the ordinary
CPU handoff command prepares one without training:

```bash
python train_bt_self_flow.py \
  --config configs/bt_self_flow/endpoint_sf/c.yaml \
  --prepare-backend /absolute/path/common/complete.pt \
  --output /absolute/path/common/backend_init.pt --device cpu
```

The launcher validates all six recipes even when planning one variant. By
default it reads and validates the checkpoint/assets, hashes local source, and
prints a plan; it neither probes CUDA nor creates the run root:

```bash
python scripts/launch_endpoint_sf.py --variant c_d_sf \
  --init-checkpoint /absolute/path/common/backend_init.pt \
  --gpus 0,1 --seed 903 --run-root /absolute/path/endpoint_runs
```

Only an explicit `--launch` probes the two selected devices and invokes
`torch.distributed.run --nproc-per-node=2`. `--preflight --launch` performs two
real updates in a separate `seed_903/preflight/VARIANT` directory. Formal launch
starts from the untouched common initializer in `seed_903/runs/VARIANT`, rather
than continuing those engineering updates. No launch or GPU allocation has
been performed by adding these files.

The launcher pins the common initialization SHA, shared settings and a copied
source snapshot under the run root, shared by all variants/seeds. It launches
the snapshot, rejects a different initializer or modified source under that
root, and preserves uncommitted implementation files in the snapshot. Training
still verifies its complete optimization and asset contract. Each run retains
its own optimizer, EMA, RNG and exact continuation state. Resume with that
variant's `--resume`, preferably using the frozen launch script if live source
has since changed. A dry run does not write a snapshot or pin a root identity.

## What each evaluation establishes

Train C, C+D, C+SF and C+D+SF independently. Compare SF gain both with and without
D, and compare D gain both with and without SF. Record images before D as well
as final images. Use actual `compress/decompress`, all packet overhead, and the
common actual-bpp interval for RD; equal quality indices do not imply equal
rate after joint C training. Report PSNR, MS-SSIM, LPIPS, DISTS, late-training
stability, paired seeds, inference cost and extra training cost.

SF is a training objective, so there is **no SF inference switch**: turning
feature capture or the loss flag off at evaluation cannot undo learned SANA
weights. Its contribution is determined by the matched independently trained
SF-off checkpoint. D can be bypassed in the same checkpoint for a dependency
diagnostic, which does not replace the independent D-off training group.

Fixed-input diagnostics may cache `(mu,delta)` from one frozen C and compare
backends or D switches on exactly those values. Do not entropy-decode one
model's packet with another model's changed entropy parameters, and do not
present fixed-input diagnostics as the jointly trained system's RD curve.
If freezing C for a decoder-only attribution study, freeze analysis, entropy,
synthesis and shared precision together, and train both compared groups under
that same frozen-front-end policy.

Readout/target unit tests and launcher planning tests are CPU engineering
checks. Passing them does not establish a production GPU memory budget,
full-model numerical stability, or useful compression gains.

## Implementation checks, 2026-11-04

65 endpoint tests passed across CPU runs: configuration/ancestry checks, native
decoder and live-feature parity, full EMA/output-head loading, selective SF
gradients, RNG preservation, actual C packet round trips, production-factory
routing, exact real-trainer continuation, and two-rank batch4/accum1 averaging
and parameter synchronization. 17 existing GAN-contract, precision-recovery,
and single-process spatial-trainer regressions also passed. The two Gloo tests
required host loopback access unavailable in the filesystem/network sandbox;
they passed with CUDA hidden and `GLOO_SOCKET_IFNAME=lo`. No tests were disabled
to bypass that restriction.

The actual pretrained DC-AE **configuration** was additionally instantiated on
the meta device: latent `[1,32,16,16]` produced prefix `[1,512,64,64]` and RGB
`[1,3,512,512]`, with scaling factor `0.41407`. This confirms production shapes,
not production-weight numerical behavior. A dry-run plan accepted the existing
calibrated pure-C initializer
`/data1/ls/code/runs/newc_pit_gpu0_20261029/backend_init.pt` (SHA256
`7286c89f56637c0c94e5d3cb8fb34801f849a44816a8977a5c579973b11efaa8`). It
created no training output and allocated no GPU model. Existing experiments
were not stopped or changed.

Reproduce the endpoint suite on a host with local Gloo sockets available:

```bash
CUDA_VISIBLE_DEVICES='' GLOO_SOCKET_IFNAME=lo PYTHONPATH=.:tests \
  python -m pytest -q tests/test_endpoint_configs.py tests/test_endpoint_decoder.py \
  tests/test_endpoint_self_flow.py tests/test_endpoint_training.py \
  tests/test_endpoint_entrypoint.py
```

Independent reviews covered the trainer/initialization/EMA integration and the
model/decoder/gradient path. An initialization-validation discrepancy between
the launcher and direct trainer was fixed by sharing one validator and adding
six direct-entry rejection regressions. Full-size CUDA forward/backward,
two-GPU memory/throughput, and actual RD evaluation remain to be measured.
