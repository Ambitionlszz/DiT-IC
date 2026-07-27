# Pure SANA AuxED Experiment

This branch adds only the historical auxiliary encoder and decoder to the
clean Pure SANA baseline:

```text
image
  -> SANA VAE encoder + frozen ELIC analysis encoder (AuxE)
  -> learned latent codec and entropy model
  -> one-step SANA flow transformer with a fixed text condition
  -> y_hat-derived latent residual (AuxD)
  -> SANA VAE decoder
  -> reconstruction
```

AuxE adds no side information and runs only while encoding. AuxD uses the
entropy-decoded `y_hat`, so decoding still depends only on the bitstream and
shared model. No latent prompt, reconstruction-variance head, self-distillation,
MeanFlow, or extra alignment loss is restored.

## Required files

- SANA model configuration and weights under `SANA/`.
- A precomputed fixed text embedding at
  `SANA/fixed_text_embed_gemma.pt`, or update `text_encoder_path` in the YAML.
- The frozen ELIC checkpoint configured by `model.params.elic_path`.
- An AuxED checkpoint for inference.

Large model weights and checkpoints are intentionally ignored by Git.

## Inference

Update paths if needed, then run:

```bash
PYTHONPATH=. python compress.py \
  --config_path configs/auxed_lrd1p0.yaml \
  --codec_path checkpoints/best_ckpt.pt \
  --img_path /path/to/images \
  --gpu 0 \
  --use_ema \
  --save_img
```

A legacy all-off training YAML can be used directly for inference. If it already
contains `model.params.codec_path`, `use_ema`, and `use_merge`, those values are
preserved when the corresponding CLI options are omitted:

```bash
PYTHONPATH=. python compress.py \
  --config_path /path/to/legacy_train.yaml \
  --img_path /path/to/images \
  --gpu 0 \
  --save_img
```

## Training

Edit the dataset and model paths in `configs/auxed_lrd1p0.yaml`, then run:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 --nnodes=1 \
  train_nogan_ddp.py --config configs/auxed_lrd1p0.yaml
```

This configuration uses `lrd: 1.0`, `resume: false`, and `codec_path: null`.
It trains AuxED from zero in `checkpoints/auxed_lrd1p0_bs8_200k` and never
loads or writes the existing baseline experiment directory.

`global_batch_size` must be divisible by the number of processes. The supplied
configuration keeps adversarial training disabled and optimizes reconstruction,
perceptual quality, bitrate, and the standard entropy-bottleneck quantiles.
`train_nogan_ddp.py` is also the canonical Stage-2 GAN entry point despite its
historical filename. It supports `disc_start`, gradient accumulation, and
`accumulate_disc`.

Legacy names are mapped automatically: `train.lrd` to `train.rate_weight`,
`loss_params.kl_weight` to `loss_params.rate_weight`, `optimizer.aux_lr` to
`optimizer.entropy_lr`, and checkpoint `opt_aux` to `opt_entropy`. Set
`train.freeze_codec: true` (or `model.params.freeze_codec: true`) for the
Codec-frozen variants.

The paired SANA versus `g_s` residual ablation must start from one Pure SANA
checkpoint and keeps the Codec in evaluation mode:

```bash
bash scripts/launch_ablation.sh /absolute/path/to/common_pure_checkpoint.pt
```

After evaluating both checkpoints with real `compress()` output, verify the
paired bitstreams before comparing reconstruction quality:

```bash
python scripts/compare_bitstreams.py BASELINE_BIN_DIR GS_RESIDUAL_BIN_DIR
```

## Checkpoint contents

Training checkpoints store the latent codec, SANA/VAE LoRA parameters, fixed
text projection, EMA copy, GAN loss/discriminator state, optimizers, and
schedulers. Removed historical heads may still exist as extra keys in older
checkpoints; they are ignored, while missing active Pure SANA weights or
incompatible LoRA keys fail with an explicit error.
