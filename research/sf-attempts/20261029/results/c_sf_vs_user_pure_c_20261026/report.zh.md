# 纯 C 与 C+SF：相同步数、相同实际总码率

纯 C CSV 共 47 个 checkpoint，2K–94K；两轮最新共同验证步数为 18000。用户确认 warmup、损失和 Kodak24 EMA 口径相同。

## 18K 的直接性能比较

| 总 bpp | 模型 | PSNR ↑ | LPIPS ↓ | DISTS ↓ |
|---:|---|---:|---:|---:|
| 0.02 | 纯 C | 22.0815 | 0.20416 | 0.11154 |
| 0.02 | C+SF | 22.1791 | 0.19876 | 0.10816 |
| 0.04 | 纯 C | 23.6161 | 0.14170 | 0.07653 |
| 0.04 | C+SF | 23.7083 | 0.13704 | 0.07550 |
| 0.06 | 纯 C | 24.5539 | 0.11282 | 0.06188 |
| 0.06 | C+SF | 24.6280 | 0.10895 | 0.06037 |
| 0.08 | 纯 C | 25.0985 | 0.09860 | 0.05456 |
| 0.08 | C+SF | 25.1979 | 0.09563 | 0.05343 |

## 相对各自 10K 起点的额外改善

计算方式：C+SF 的 10K→当前改善量，减去纯 C 的同期改善量。正值表示 C+SF 进步更多；它只是轨迹比较，不是严格因果估计。

| 总 bpp | PSNR 额外改善 (dB) | LPIPS 额外降低 | DISTS 额外降低 |
|---:|---:|---:|---:|
| 0.02 | -0.01892 | +0.000197 | +0.000768 |
| 0.04 | -0.03774 | +0.001087 | -0.001779 |
| 0.06 | -0.03212 | +0.000646 | +0.000344 |
| 0.08 | -0.02833 | -0.001045 | -0.001125 |

## 判断

C+SF 在 18K 的绝对指标更好，但未开启 SF 的 10K 已有优势。扣除原有差距后，PSNR 没有新增收益，LPIPS 仅有局部小幅信号，DISTS 方向混合。当前证据不足以支持 SF 带来明确、稳定的额外压缩收益。
最有判别力的后续对照是从本轮相同 pre_sf.pt 分叉，保持相同数据顺序、batch、优化器、随机性、数值精度和更新预算，单独改变 SF；当前分析不改动正在运行的实验。

## 比较边界

- User confirmed identical 50K latent warmup, 4/1/1 + 0.1 base objective and Kodak24 EMA evaluation.
- CSV contains no dataset fingerprint, commit, seed, optimizer/batch details or checkpoint hash; these identities cannot be independently audited from the supplied file.
- The two runs already differ at 10K before SF activation. Equal named configuration does not establish an identical pre-SF state.
- Raw 18K gaps include the pre-existing 10K gap. Change in that gap is a descriptive comparison of learning trajectories, not proof of a causal SF effect.
- Matched rates use PCHIP on log actual total packet bpp within measured range, without extrapolation. They are interpolated values, not new model inference.
- No per-image metrics or repeated seeds are present, so no paired uncertainty interval or significance claim is available.
- The 94K pure-C curve is contextual only; the main comparison uses the same validation step for both runs.
- Training has not been modified or stopped for this analysis.
