# Pure SANA: AdaLN Compression MeanFlow

This branch implements Compression MeanFlow 5A directly on the Pure SANA
pipeline. It intentionally includes only the AdaLN conditioning experiment:

- `t` and `h=t-r` use independent `256 -> 1152 -> 1152` embedders.
- Their sum is added to SANA's native timestep embedding before
  AdaLN-Single.
- The old first-text-token interval adapter is not present.
- The 5B velocity-target control is not present.

The final projections of both interval embedders are zero-initialized, so the
MeanFlow endpoint initially matches the ordinary Pure SANA one-step operator.
The checkpoint records both the 5A mode ID and AdaLN conditioning ID and
rejects mismatched inference configs.

## Joint training

Edit dataset and model paths in
`configs/pure_sana_meanflow_adaln.yaml`, then run:

```bash
torchrun --standalone --nproc_per_node=1 train_nogan_ddp.py \
  --config configs/pure_sana_meanflow_adaln.yaml
```

The training objective adds:

```text
meanflow_weight * ||U(z_t,t,r) - stopgrad(v - (t-r)dU/dt)||^2_adaptive
```

where `z_t=(1-t)z_0+t z_1`, `z_0` is the clean VAE latent and `z_1` is the
codec-decoded latent. Both endpoints are detached only inside the MeanFlow
auxiliary bridge. The ordinary reconstruction/rate objective still trains the
codec normally.

`equal_time_ratio` remains `0.5`. Rows with `r=t` skip forward-mode JVP because
their correction is exactly zero; rows with `r<t` use `torch.func.jvp` and the
math SDPA backend.

## Fixed-bitstream causal comparison

Start both continuations from the same Pure SANA checkpoint:

```bash
bash scripts/run_frozen_meanflow_ablation.sh \
  pure configs/pure_sana_meanflow_adaln.yaml /path/to/pure.pt 1

bash scripts/run_frozen_meanflow_ablation.sh \
  meanflow configs/pure_sana_meanflow_adaln.yaml /path/to/pure.pt 1
```

This mode freezes the codec and entropy model, keeps the codec in evaluation
mode, restarts the optimizer/scheduler, and rejects a common checkpoint that
already contains MeanFlow state. After real rANS evaluation, verify byte
identity before comparing reconstruction quality:

```bash
python scripts/compare_bitstreams.py \
  /path/to/pure/bin /path/to/meanflow/bin
```

## Inference

Use the same MeanFlow YAML and its trained checkpoint:

```bash
python compress.py \
  --config_path configs/pure_sana_meanflow_adaln.yaml \
  --codec_path /path/to/meanflow_checkpoint.pt \
  --img_path /path/to/images \
  --save_img
```

Deployment remains one SANA call:

```text
rANS decode -> z_1 -> z_1 - U(z_1,1,0) -> VAE decode
```

## Server verification

Before a long run, use the training server environment:

```bash
python -m unittest \
  tests.test_compression_meanflow \
  tests.test_sana_meanflow_jvp \
  tests.test_meanflow_config
```

The SANA test checks JVP finiteness, centered finite-difference agreement, and
nonzero gradients for both the AdaLN interval adapter and SANA LoRA.
