# DiT-IC auxE / auxD 消融实验指南

## 实验目标

验证 ELIC 辅助编码器 (auxE) 和 AuxDecoder 残差路径 (auxD) 在 DiT-IC Stage 1 训练中是否有效。

## 前置准备

### 1. SANA 权重

```bash
conda activate ditic
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ENABLE_HF_TRANSFER="1"
hf download Efficient-Large-Model/Sana_600M_1024px_diffusers --local-dir ./SANA_600M_1024px_diffusers

# 将权重文件复制到 SANA 目录下对应子文件夹
cp SANA_600M_1024px_diffusers/transformer/*.safetensors SANA/transformer/
cp SANA_600M_1024px_diffusers/vae/*.safetensors SANA/vae/
```

确认 SANA 目录结构：
```
SANA/
├── transformer/
│   ├── config.json                              # 已有
│   └── diffusion_pytorch_model.safetensors       # 从上面下载
├── vae/
│   ├── config.json                              # 已有
│   └── diffusion_pytorch_model.safetensors       # 从上面下载
└── scheduler/
    └── scheduler_config.json                    # 已有
```

### 2. ELIC 权重

已位于：`/data2/luosheng/hf_models/hub/elic_official.pth`

确保 `configs/train_256_nogan.yaml` 中 `elic_path` 指向正确路径。

### 3. 数据集

修改 `configs/train_256_nogan.yaml` 中的数据路径：
```yaml
data:
  data_path:
    - /your/training/data/path
  valid_path:
    - /your/validation/data/path
  path_size: [256, 256]
```

## 训练配置

在 `configs/train_256_nogan.yaml` 中确认以下关键参数：

```yaml
model:
  params:
    use_aux_encoder: True    # CLI 可覆盖
    use_aux_decoder: True    # CLI 可覆盖
    elic_path: "/data2/luosheng/hf_models/hub/elic_official.pth"
    dit_path: "SANA"

train:
  output_dir: 'checkpoints'
  exp_name: '256_nogan_0.5lrd_100k'
  global_batch_size: 72
  max_steps: 100001
  log_every: 100
  ckpt_every: 10000
```

## 四组消融实验

在项目根目录 `/data2/luosheng/code/DiT-IC` 下执行，将 `N` 替换为 GPU 数量（如 2、4、8）。

### 实验 1：基线（auxE + auxD 全开）

```bash
torchrun --standalone --nproc_per_node=N train_nogan_ddp.py \
    --config configs/train_256_nogan.yaml
```

实验目录：`checkpoints/256_nogan_0.5lrd_100k/`

### 实验 2：去掉 AuxDecoder（仅 auxE）

```bash
torchrun --standalone --nproc_per_node=N train_nogan_ddp.py \
    --config configs/train_256_nogan.yaml \
    --use_aux_decoder 0
```

实验目录：`checkpoints/256_nogan_0.5lrd_100k_noauxd/`

### 实验 3：去掉 auxE（仅 AuxDecoder）

```bash
torchrun --standalone --nproc_per_node=N train_nogan_ddp.py \
    --config configs/train_256_nogan.yaml \
    --use_aux_encoder 0
```

实验目录：`checkpoints/256_nogan_0.5lrd_100k_noauxe/`

### 实验 4：全去掉

```bash
torchrun --standalone --nproc_per_node=N train_nogan_ddp.py \
    --config configs/train_256_nogan.yaml \
    --use_aux_encoder 0 \
    --use_aux_decoder 0
```

实验目录：`checkpoints/256_nogan_0.5lrd_100k_noauxe_noauxd/`

## 一键跑四组

```bash
#!/bin/bash
CONFIG="configs/train_256_nogan.yaml"
NPROC=2  # GPU 数量

echo "=== Experiment 1: Baseline (auxE+auxD) ==="
torchrun --standalone --nproc_per_node=$NPROC train_nogan_ddp.py --config $CONFIG

echo "=== Experiment 2: No auxD ==="
torchrun --standalone --nproc_per_node=$NPROC train_nogan_ddp.py --config $CONFIG --use_aux_decoder 0

echo "=== Experiment 3: No auxE ==="
torchrun --standalone --nproc_per_node=$NPROC train_nogan_ddp.py --config $CONFIG --use_aux_encoder 0

echo "=== Experiment 4: No auxE + No auxD ==="
torchrun --standalone --nproc_per_node=$NPROC train_nogan_ddp.py --config $CONFIG --use_aux_encoder 0 --use_aux_decoder 0

echo "All experiments done!"
```

## 开关机制说明

### auxE (`use_aux_encoder`)

| 开启 (1) | 关闭 (0) |
|----------|----------|
| 加载 ELIC g_a（~4GB 显存） | 不加载 ELIC |
| `latent2` = ELIC 特征 (B, 320, H/16, W/16) | `latent2` = None |
| `AnalysisTransform`: pre1(32→128) + pre2(320→64) → concat→192 | `AnalysisTransform`: pre1(32→192) 直通，无 concat |

### auxD (`use_aux_decoder`)

| 开启 (1) | 关闭 (0) |
|----------|----------|
| `AuxDecoder` 创建（~3M 参数） | 不创建 |
| `res` = AuxDecoder(y_hat) | `res` = None |
| `x_denoised = sched.step(...) + res` | `x_denoised = sched.step(...)` 无残差 |

## 结果对比

训练结束后，每组实验的 `checkpoints/<exp_name>/log.txt` 中包含训练日志。对比指标：

| 指标 | 含义 |
|------|------|
| `train/total_loss` | 总 loss |
| `train/rec_loss` | 重建 loss (MSE) |
| `train/kl_loss` | 码率 loss (bpp) |
| `train/lpips` | 感知 loss (LPIPS) |

如果 auxE/auxD 有效，去掉后应该观察到：
- `rec_loss` 上升（重建质量下降）
- `lpips` 上升（感知质量下降）
- 或相同质量下 `kl_loss` 上升（码率增加）

## 代码改动说明

| 文件 | 改动 |
|------|------|
| `models/latent_codec.py` | `AnalysisTransform` 加 `use_aux_encoder`；`LatentCodec` 加两个开关 |
| `models/DiT_IC.py` | `Codec` 加两个开关，条件加载 ELIC，res/fallback 保护 |
| `train_nogan_ddp.py` | CLI 参数 + config 注入 + 实验名后缀 |
| `configs/train_256_nogan.yaml` | 默认值文档 |

设计参考：FluxCodec (`/data2/luosheng/code/flux2/FluxCodec`) 的 auxE/auxD 开关机制。


---

# DiT-IC 创新点消融实验：验证 SANA Backbone 效果

## 实验目标

增量式移除 DiT-IC 的三个创新点，验证 SANA backbone 本身的有效性：

1. **Variance-Guided Reconstruction Flow** — 基于预测方差的逐元素自适应 sigma。去掉后使用固定 `flow_prediction` 模式。
2. **Self-Distillation Alignment** — 去噪后 latent 与原始 latent 的余弦相似度损失。去掉后 loss 置零。
3. **Latent-Conditioned Guidance** — 从 latent codec 提取特征作为 DiT 的条件。去掉后使用固定 CLIP 文本嵌入替代。

## 前置准备

### CLIP 文本编码器（仅实验 4 需要）

实验 4 使用 CLIP ViT-B/32 文本编码器，路径已在配置中指定：
```yaml
text_encoder_path: "/data2/luosheng/hf_models/hub/clip-vit-base-patch32"
```
无需额外下载，`transformers` 库会自动加载。

## 四组增量消融实验

### 实验 1：基线（所有创新点全开）

```bash
torchrun --standalone --nproc_per_node=N train_nogan_ddp.py \
    --config configs/train_256_nogan.yaml
```

实验目录：`checkpoints/256_nogan_0.5lrd_bs8_200k/`

### 实验 2：去掉 Variance-Guided Reconstruction Flow

```bash
torchrun --standalone --nproc_per_node=N train_nogan_ddp.py \
    --config configs/train_256_nogan.yaml \
    --use_variance_guide 0
```

实验目录：`checkpoints/256_nogan_0.5lrd_bs8_200k_novarguide/`

### 实验 3：去掉 Variance-Guided + Self-Distillation

```bash
torchrun --standalone --nproc_per_node=N train_nogan_ddp.py \
    --config configs/train_256_nogan.yaml \
    --use_variance_guide 0 \
    --use_self_dist 0
```

实验目录：`checkpoints/256_nogan_0.5lrd_bs8_200k_novarguide_noselfdist/`

### 实验 4：去掉所有三个创新点（纯 SANA Backbone + 文本条件）

```bash
torchrun --standalone --nproc_per_node=N train_nogan_ddp.py \
    --config configs/train_256_nogan.yaml \
    --use_variance_guide 0 \
    --use_self_dist 0 \
    --use_text_condition 1
```

实验目录：`checkpoints/256_nogan_0.5lrd_bs8_200k_novarguide_noselfdist_textcond/`

## 一键跑四组

```bash
#!/bin/bash
CONFIG="configs/train_256_nogan.yaml"
NPROC=2  # GPU 数量

echo "=== Experiment 1: Baseline ==="
torchrun --standalone --nproc_per_node=$NPROC train_nogan_ddp.py --config $CONFIG

echo "=== Experiment 2: -VarianceGuide ==="
torchrun --standalone --nproc_per_node=$NPROC train_nogan_ddp.py --config $CONFIG --use_variance_guide 0

echo "=== Experiment 3: -VarianceGuide -SelfDist ==="
torchrun --standalone --nproc_per_node=$NPROC train_nogan_ddp.py --config $CONFIG --use_variance_guide 0 --use_self_dist 0

echo "=== Experiment 4: -All3 + TextCondition ==="
torchrun --standalone --nproc_per_node=$NPROC train_nogan_ddp.py --config $CONFIG --use_variance_guide 0 --use_self_dist 0 --use_text_condition 1

echo "All experiments done!"
```

## 开关机制说明

### `use_variance_guide`

| 开启 (1) | 关闭 (0) |
|----------|----------|
| `DiT_mode = 'scale'` | `DiT_mode = 'flow_prediction'` |
| `AdaptiveNoiseScheduler` 根据 `lq_scale_hat` 逐元素调制 sigma | 使用 DPM Scheduler 的固定 sigma |
| sigma_i = sigma_base * (1 + f(lq_scale_hat_i)) | sigma 对所有元素相同 |

### `use_self_dist`

| 开启 (1) | 关闭 (0) |
|----------|----------|
| `kl_loss = 0.05 * relu(0.5 - cos(x_denoised, lq_latent))` | `kl_loss = 0` |
| 去噪输出被约束靠近原始 latent | 无此约束 |

### `use_text_condition`

| 关闭 (0，默认） | 开启 (1) |
|----------|----------|
| `pos_caption_enc = prompter(prompt)` | `pos_caption_enc = text_proj(CLIP("a high quality image"))` |
| 条件来自 latent codec 特征（图像内容相关） | 条件来自固定文本嵌入（与图像内容无关） |
| prompter 参数参与训练 | text_proj 参数参与训练，CLIP 冻结 |

## 预期结果

如果三个创新点都有效，增量移除后应该观察到：
- PSNR 逐步下降（重建质量下降）
- LPIPS 逐步上升（感知质量下降）
- 或相同质量下 bpp 上升（码率增加）

实验 4 代表了纯 SANA backbone 的能力下界。

## 代码改动说明

| 文件 | 改动 |
|------|------|
| `models/DiT_IC.py` | `Codec` 加三个开关 + CLIP 加载 + 条件切换逻辑 |
| `models/scheduler.py` | 修复 adaptor 创建条件，使 `flow_prediction` 等模式不创建 adaptor |
| `train_nogan_ddp.py` | CLI 参数 + config 注入 + 实验名后缀 |
| `configs/train_256_nogan.yaml` | 默认值文档 + text_encoder_path |
