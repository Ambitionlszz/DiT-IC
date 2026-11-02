# 图像质量评测协议与修复（2026-11-02）

本次修复集中在指标计算、结果身份和可重复性。没有调整训练损失权重、码率系数、模型结构或 best 的 RD loss 选择标准；已有结果数值和历史 run 的 source 快照不作改写。运行中的训练未重启，源码修复从新进程或下一次续训开始生效。

## 指标定义

公共实现位于 `evaluation_protocol.py`，协议版本为 `image_quality_rgb_y_v2`。BT 验证、BT 码流评测、旧压缩入口、LIC 直接评测及目录评测使用相同的 PSNR 和两种 MS-SSIM 定义。

| 输出 | 定义 |
|---|---|
| `psnr` | RGB `[0,1]`，FP32，逐图 MSE，`-10*log10(max(MSE, 1e-12))`，完全相同时为 120 dB |
| `ms_ssim_rgb` | pytorch-msssim 1.0.0，浮点 RGB，`data_range=1`，五尺度、11 像素窗口 |
| `ms_ssim_y` | pyiqa MS_SSIM，显式 YIQ Y 通道、内部 255 尺度取整、FP64、`downsample=False` |
| `ms_ssim` | v2 中始终等于 `ms_ssim_rgb` |

旧 `compress.py`、LIC 直接评测和目录评测的 `ms_ssim` 曾代表 Y 通道，旧 BT 字段则代表 RGB。旧报告不能仅按同名列与 v2 混合，也不能从历史均值推算另一通道的数值。新结果保留两种显式字段，便于直接检查差异。

两种 MS-SSIM 都只对最短边大于 160 的图像计算；不足尺寸时省略字段，不缩放图像、不以零补齐均值。汇总只对实际有该指标的图像求平均，BT 码流评测另外记录 `structure_metric_images`。

训练 `ImageLoss` 的公式与权重保持原样，包括训练日志原有 PSNR 观测公式；验证及独立评测的 PSNR 使用公共定义。LPIPS 继续使用 Alex v0.1，`[-1,1]` 输入；DISTS 使用 pyiqa，`[0,1]` 输入。

## 浮点与 PNG

三个压缩评测入口新增 `--metric-domain float|uint8`，默认 `float`，保留此前对浮点重建评测的方式。`uint8` 会先按 PNG 的量化规则处理重建与原图，再计算所有图像指标。元数据分别记录为 `float` 和 `uint8_roundtrip`。

所有评测 PNG 保存统一使用 `round(255*x)`，不再使用向下取整。浮点评测与 PNG 评测仍是两种不同协议；选择 `uint8` 后指标输入与保存图片对应的像素一致。目录评测使用文件实际解码的 RGB uint8 像素，不猜测外部图片生产时采用的取整规则。

示例（替换 checkpoint、图片目录和输出目录）：

```bash
python eval_bt_self_flow.py --checkpoint RUN/joint/best.pt \
  --input DATASET --output OUTPUT --save-images --metric-domain uint8

python -m eval.evaluate --recon-dir OUTPUT/q00000 --gt-dir DATASET \
  --strip-recon-index-prefix --output-json OUTPUT/folder_metrics.json
```

第二条中的前缀选项只用于 `00000_原文件名.png` 这类 BT 输出命名；普通同名图片不加该选项。具体质量子目录名称以评测输出为准。

## 配对与分布指标

目录评测按大小写敏感的文件 stem 一一配对，允许扩展名不同。缺图、多图、重复 stem、损坏图片、尺寸不一致会在加载指标网络前报错；`ntest` 不会掩盖完整目录的成员不匹配。没有 GT 时只计算无参考指标。

FID/KID 沿用 256 像素 patch 的评测约定，KID 显式固定 `subset_size=50`。是否可以计算取决于实际 patch 数：FID 至少 2 对，KID 至少 50 对；不足时记录 unavailable，避免崩溃。目录成对评测仍要求原图两边至少 256；压缩入口的小图仍可计算逐图指标，但不贡献分布指标 patch。分布指标的样本数与设置单独记录，不能与普通逐图均值混为一谈。

## 结果身份与旧 checkpoint

新结果记录协议参数、实现文件 SHA256、依赖版本、源码 Git 身份，以及按顺序读取图像文件内容得到的 `dataset_content_sha256`。同名同尺寸图片被替换也会使内容指纹变化。独立码流评测保留原有 checkpoint SHA256、raw/EMA 和码流身份记录。

BT 验证身份还包括验证 crop、padding、EMA、实际/估计码率和选模目标配置。与旧 best 身份不兼容时，续训会先重新验证当前恢复的权重，建立新基线；支持数字文件、`paused.pt`、`complete.pt`、`best.pt`。这一额外验证会保存并恢复 RNG，不清空优化器，也不额外执行训练更新。

`compare_bt_self_flow.py` 以及 LIC Gate A/B 会检查协议与内容身份，拒绝新旧口径混用、RGB/Y 混用和 float/uint8 混用。历史 LIC schema 1 之间的流程继续兼容，但明确标记为身份未经完整验证的 legacy；新 LIC 结果为 schema 2。旧报告不会被自动赋予 v2 身份。

## 感知网络的模式

旧 `LPIPSWithDiscriminator.train()` 以及 BT `ImageLoss.train()` 现在都会把冻结的 LPIPS/DISTS 固定为 eval，防止父模块 `.train()` 重新打开 dropout 或更新统计量。图像梯度保留，判别器按请求的 train/eval 模式工作，real/fake 仍分别前向。

这些修复使新结果可以按明确口径复现和比较。它们不代表 MS-SSIM/LPIPS 会自动改善；增加结构监督、调整权重或更换选模目标仍需独立的受控实验。
