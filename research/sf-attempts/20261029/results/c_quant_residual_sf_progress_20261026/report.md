# C+residual Self-Flow progress / Kodak24

Train step: 19200; latest completed validation: 18000.

| total bpp | model | PSNR ↑ | LPIPS ↓ | DISTS ↓ |
|---:|---|---:|---:|---:|
| 0.04 | 512 baseline / 200K | 23.8634 | 0.12983 | 0.07422 |
| 0.04 | Previous C / 95.5K | 23.2085 | 0.14277 | 0.07878 |
| 0.04 | Previous C+D / 96.8K | 23.3289 | 0.13770 | 0.07782 |
| 0.04 | Current pure C / 10K | 23.6163 | 0.14148 | 0.07569 |
| 0.04 | Current C+residual SF / 18K | 23.7083 | 0.13704 | 0.07550 |
| 0.04 | Previous C+D+SF / 18K | 23.0217 | 0.16430 | 0.08669 |
| 0.04 | Previous C+D+SF / 32K | 23.2785 | 0.15177 | 0.07972 |
| 0.06 | 512 baseline / 200K | 24.5725 | 0.10831 | 0.06248 |
| 0.06 | Previous C / 95.5K | 24.1529 | 0.11420 | 0.06410 |
| 0.06 | Previous C+D / 96.8K | 24.2069 | 0.11176 | 0.06277 |
| 0.06 | Current pure C / 10K | 24.5651 | 0.11196 | 0.06147 |
| 0.06 | Current C+residual SF / 18K | 24.6280 | 0.10895 | 0.06037 |
| 0.06 | Previous C+D+SF / 18K | 24.1513 | 0.12396 | 0.06516 |
| 0.06 | Previous C+D+SF / 32K | 24.3067 | 0.11910 | 0.06315 |
| 0.08 | 512 baseline / 200K | 24.9030 | 0.09897 | 0.05890 |
| 0.08 | Previous C / 95.5K | 24.7800 | 0.09976 | 0.05610 |
| 0.08 | Previous C+D / 96.8K | 24.7806 | 0.09886 | 0.05559 |
| 0.08 | Current pure C / 10K | 25.1538 | 0.09793 | 0.05442 |
| 0.08 | Current C+residual SF / 18K | 25.1979 | 0.09563 | 0.05343 |
| 0.08 | Previous C+D+SF / 18K | 24.8659 | 0.10515 | 0.05572 |
| 0.08 | Previous C+D+SF / 32K | 24.9636 | 0.10264 | 0.05484 |

## Interpretation

At 0.04 bpp the current model still trails the 512 baseline on all three metrics. At 0.06 bpp PSNR and DISTS are better, while LPIPS is close but slightly worse. At 0.08 bpp all three metrics are better.
All three metrics improve from the current pure-C 10K prefix to 18K at the three matched rates, but gains are modest and cannot isolate Self-Flow from continued optimization.
No new forward NaNs or numerical restarts have occurred in this continuation. Two AMP gradient overflow retries lowered the scale and successfully replayed the same data.

## Limits

- Matched-rate points use PCHIP in log actual total packet bpp, with no extrapolation; these are interpolated, not additional model evaluations.
- Historical baseline/C/C+D have different training budgets, warmups, losses, and evaluator implementations. These are reference checkpoint comparisons, not controlled ablations.
- Historical C/C+D losses use 2/1/0.5 and both base/repaired terms; current uses 4/1/1 and only base.
- Current validation uses FP32 deployment EMA, Kodak24 full resolution, six qualities, actual packet accounting on all images; packet decode versus hard forward is checked on two images per quality.
- Historical metrics use pyiqa PSNR/torchmetrics Alex LPIPS; current uses MSE-derived RGB PSNR/upstream Alex LPIPS. Both use pyiqa DISTS.
- 10K pure C to 18K SF includes 8000 additional joint optimization steps. No matched-budget SF-off control has been trained; improvement is not proof of causal SF gain.
- Most of the current advantage versus historical methods is already present at 10K before SF, consistent with the new warmup/frontend/system accounting for much of the gain.
