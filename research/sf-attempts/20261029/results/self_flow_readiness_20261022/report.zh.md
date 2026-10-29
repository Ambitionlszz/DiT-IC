# C / D 当前水平与 512 baseline 对比

快照时间：2026-10-22T11:36:56.072693+08:00

以下依据已保存日志与验证文件；未核验当前 GPU/进程运行状态。checkpoint 步数均指 C/D 的 joint 阶段，之前还各有共同的 50k codec warmup。

| 模型 | 当前训练 step | 最新验证 step | 当前阶段预算 | 最新验证损失 | 最佳验证 step |
|---|---:|---:|---:|---:|---:|
| C | 89200 | 88000 | 200000 | 0.320648 | 86000 |
| D | 90500 | 90000 | 200000 | 0.316259 | 90000 |

主表采用各模型最新完整验证。正 BD-rate＝同指标需更多码率（较差）；负值＝省码率（较好）。共同质量区间 PCHIP 积分，不外推。

| 模型 | 参照 | PSNR BD-rate | LPIPS BD-rate | DISTS BD-rate |
|---|---|---:|---:|---:|
| C / 88k | 512 / 200k（历史归档） | +29.11% | +18.70% | +11.15% |
| D / 90k | 512 / 200k（历史归档） | +24.72% | +13.30% | +4.27% |
| C / 88k | 512 / 30k（之前本机推理） | -1.38% | -8.80% | -14.24% |
| D / 90k | 512 / 30k（之前本机推理） | -5.82% | -13.71% | -20.32% |

C/D 同步数比较（双方 88k）：

| 比较 | PSNR BD-rate | LPIPS BD-rate | DISTS BD-rate |
|---|---:|---:|---:|
| D 相对 C | -2.13% | -4.39% | -4.15% |

最新验证步数 C=88000、D=90000；公平 C/D 比较使用双方同为 88k 的附图和上述同步数行。当前 D 相对老 30k baseline 有优势，对历史 200k baseline 尚有差距。

C 的最佳综合验证 step=86000，D 的最佳综合验证 step=90000。近期处于缓慢改善和波动阶段；综合 loss 最低不表示每个画质指标都同时最好，尚不能据此断言已完全收敛或必定赶上 200k baseline。

本轮评测来源：C/D 是最新 EMA 的 Kodak24 原图验证，画质取 FP32 硬前向、bpp 来自全部完整真实码流字节，每档另抽 2 张解码核对；本次所用验证的核对误差均为 0。没有把本轮称为全量真实解码复测。30k baseline 沿用之前本机全量真实 rANS 推理；200k baseline 是之前 512 非窗口训练、全局解码的历史记录，权重在本机不可用，未重跑。46k 曾完成全量 C/D 真实解码并与验证一致，66k 六张样例真实解码也一致。

训练总预算、前端初始化和损失不同（C/D 图像损失 2/1/0.5，baseline 为 4/1/1），这里衡量当前结果差距。Kodak 用于模型选择，不能当作独立测试或等预算的结构优劣结论。

![最新 RD 曲线](/data2/ls/results/self_flow_readiness_20261022/rd_curves.png)

![同为88k RD曲线](/data2/ls/results/self_flow_readiness_20261022/rd_curves_same_step.png)

![与200k baseline的差距](/data2/ls/results/self_flow_readiness_20261022/bd_rate_vs_baseline200k.png)

![相对 baseline 的训练趋势](/data2/ls/results/self_flow_readiness_20261022/bd_rate_progress.png)

完整六档数据、各指标积分区间与覆盖比例见 comparison.json；可编辑数值见 rd_points.csv 和 bd_rate.csv。
