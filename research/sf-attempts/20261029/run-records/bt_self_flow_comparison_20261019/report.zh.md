# C/D 与 512 baseline：Kodak24 真实码流比较

C/D 均为 34K deployment EMA；本机 baseline 为 30K EMA；历史 200K baseline 使用原始归档曲线，未重新推理。
原始整图、FP32、全局单步解码。bpp 包括码流头和全部侧信息，以原图像素计费。

## 相同真实码率下的曲线插值

以下数值由六档曲线在 log(bpp) 上做 PCHIP 插值，只使用共同覆盖范围；不是新增实测档位。

| bpp | 模型 | PSNR ↑ | LPIPS ↓ | DISTS ↓ |
|---:|---|---:|---:|---:|
| 0.04 | Baseline 512 / 30K | 23.257 | 0.14890 | 0.08357 |
| 0.04 | C / 34K | 22.895 | 0.15703 | 0.08681 |
| 0.04 | D / 34K | 22.883 | 0.15178 | 0.08247 |
| 0.04 | Baseline 512 / 200K (archived) | 23.863 | 0.12983 | 0.07422 |
| 0.06 | Baseline 512 / 30K | 23.975 | 0.12436 | 0.07203 |
| 0.06 | C / 34K | 23.971 | 0.12134 | 0.06643 |
| 0.06 | D / 34K | 23.870 | 0.12039 | 0.06691 |
| 0.06 | Baseline 512 / 200K (archived) | 24.572 | 0.10831 | 0.06248 |
| 0.08 | Baseline 512 / 30K | 24.245 | 0.11548 | 0.06825 |
| 0.08 | C / 34K | 24.600 | 0.10519 | 0.05771 |
| 0.08 | D / 34K | 24.504 | 0.10493 | 0.05994 |
| 0.08 | Baseline 512 / 200K (archived) | 24.903 | 0.09897 | 0.05890 |

## BD-rate

负值表示相同质量所需比特更少；共同质量区间积分，不外推、不修复非单调曲线。

| 比较 | PSNR | LPIPS | DISTS |
|---|---:|---:|---:|
| D_vs_C | +3.62% | -3.01% | -1.93% |
| C_vs_baseline_30k | +10.57% | +5.81% | +2.54% |
| D_vs_baseline_30k | +12.94% | +0.71% | -6.34% |
| C_vs_historical_200k | +41.85% | +33.69% | +25.60% |
| D_vs_historical_200k | +47.55% | +29.84% | +21.27% |

## 当前判断

相同 34K 下，D 相对 C 的 PSNR BD-rate 为 +3.62%，LPIPS 为 -3.01%，DISTS 为 -1.93%。
当前表现是感知质量与像素保真的取舍，尚不能称为全范围优于 C。训练仍处于 200K 预算的早期。

## 解释边界

- C/D 34K versus local baseline 30K; historical baseline 200K is archived results, its weights are unavailable
- C/D train image loss 2/1/0.5; baseline image loss 4/1/1; frontends and initialization differ
- Preview rates are matched per image to baseline q2; maximum residual mismatch 0.431%
- Kodak is used for checkpoint selection in these runs; it is not a held-out test set

全部 432 个六档推理重建通过有限值、实际文件字节计费和逐图数量核验；额外 6 次重建用于匹配逐图码率的视觉对照。C/D 每张仅一次 SANA 前向。

![RD curves](rd_curves.png)

![Reconstructions](reconstruction_preview.png)

完整逐图记录、包、PNG、CSV、插值和 BD-rate 区间见本目录。

## 训练恢复

主机重启后，已从 C 34K / D 36K 的完整状态恢复双卡 200K 训练。已确认更新 C34001 / D36001，梯度有限且没有新增数值异常报告；完整验收记录见 training_resume_acceptance.json。

局部对比图下方列出的 PSNR / LPIPS 为对应整张图片的指标。
