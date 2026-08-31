# Pure SANA Image Compression

This branch contains one deterministic compression path:

```text
image
  -> SANA VAE encoder
  -> learned latent codec and entropy model
  -> one-step SANA flow transformer with a fixed text condition
  -> SANA VAE decoder
  -> reconstruction
```

The model has no secondary image encoder, residual decoder, learned latent
prompt, reconstruction-variance head, or extra alignment loss. The entropy
bottleneck quantile loss remains because it is part of standard learned entropy
coding.

## Required files

- SANA model configuration and weights under `SANA/`.
- A precomputed fixed text embedding at
  `SANA/fixed_text_embed_gemma.pt`, or update `text_encoder_path` in the YAML.
- A Pure SANA checkpoint for inference.

Large model weights and checkpoints are intentionally ignored by Git.

## Inference

Update paths if needed, then run:

```bash
bash infer_pure_sana.sh checkpoints/best_ckpt.pt /path/to/images
```

The equivalent direct command is:

```bash
PYTHONPATH=. python compress.py \
  --config_path configs/pure_sana_infer.yaml \
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

Edit the dataset paths in `configs/pure_sana_train.yaml`, then run:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 --nnodes=1 \
  train_nogan_ddp.py --config configs/pure_sana_train.yaml
```

`global_batch_size` must be divisible by the number of processes. The supplied
configuration keeps adversarial training disabled and optimizes reconstruction,
perceptual quality, bitrate, and the standard entropy-bottleneck quantiles.
`train_nogan_ddp.py` is also the canonical Stage-2 GAN entry point despite its
historical filename. It supports `disc_start`, gradient accumulation, and
`accumulate_disc`.

To start from a completed variable-rate Stage-1 Phase-A checkpoint and run the
adversarial Phase B, adversarial Stage 2, and both real-rANS evaluations without
manual handoffs:

```bash
bash scripts/run_adv_phase2_then_stage2.sh 0,1 \
  checkpoints/variable_rate_lic_v5_main/checkpoints/phase_a_complete.pt
```

The optional third argument overrides the evaluation image directory. Re-running
the same command resumes compatible incomplete Phase-B or Stage-2 checkpoints.

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
