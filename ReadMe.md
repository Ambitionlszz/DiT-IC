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

## Training

Edit the dataset paths in `configs/pure_sana_train.yaml`, then run:

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 --nnodes=1 \
  train_nogan_ddp.py --config configs/pure_sana_train.yaml
```

`global_batch_size` must be divisible by the number of processes. The supplied
configuration keeps adversarial training disabled and optimizes reconstruction,
perceptual quality, bitrate, and the standard entropy-bottleneck quantiles.

## Checkpoint contents

Training checkpoints store the latent codec, SANA/VAE LoRA parameters, fixed
text projection, EMA copy, optimizers, and schedulers. Removed historical heads
may still exist as extra keys in older checkpoints; nonmatching keys are ignored
when loading the model weights.
