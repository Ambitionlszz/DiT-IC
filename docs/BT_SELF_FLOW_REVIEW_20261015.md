# BT-SelfFlow 实现复核与两卡运行记录

工作区 `/data2/ls/code/DiT-IC-bt-self-flow`，分支 `codex/bt-self-flow-v1-20261015`。
本次复核承接完整方案 v2，目标是 C/D 两组可训练、可恢复、可独立真实解码；研究收益由相同真实总 bpp 的结果判断。

## 核心机制是否落实

| 机制 | 代码与检查 | 判断 |
|---|---|---|
| 真正改变编码精度 | `codec.steps()` 生成局部 Δ；`_groups()` 统一中心化整数、概率单位、四组上下文和 LRP；旧增益关闭 | 已实现；不是事后掩码 |
| Router 的输入可获得 | 源 latent 的 2×2 重排、分析特征、无已解码上下文的超先验参考码长；3 bit 分配写包 | 无分配/概率循环依赖；解码不运行 Router |
| 步长有失真梯度 | 在量化单位做硬前向 STE，然后乘 Δ 恢复，保留 Δ 与编码器梯度 | 实际梯度检查通过；未使用完整反量化外套 detach 的写法 |
| 真实码流自包含 | λ 先转 q16 再参与计算；分配、超先验、主 rANS、D 尺度、混合掩码和尺寸全部写包 | 独立新进程 rANS 往返通过，文件字节全部计费 |
| 后端获得可靠精度信息 | 编码端从实际恢复 μ 测 2×2 块 RMS，传 4 bit log 尺度；训练也用恢复后的尺度 | 没有误差预测 CNN；没有把量化误差说成高斯 |
| C/D 公平初始化 | 共同前端结束后只生成一次 `backend_init.pt`；D 模块独立 RNG、零初始化 | 实际共享参数逐项相同，数据/裁剪/λ 随机流独立于网络 RNG |
| D 单步修复 | 原生固定时间不变；尺度 Fourier/MLP 调制；μ−stopgrad(σ)u；完整 28 层 | 一次 SANA 前向，无隐藏第二步或重复 scheduler 缩放 |
| Self-Flow 可接 | 学生量化前高低混合、完整高质量 EMA 教师、学生 8 层/教师 20 层、学生继续输出最终图像 | 已有独立配置与训练测试；初始 C/D 不启用 |
| 正确恢复 | raw/EMA/优化器/辅助优化器/LR/scaler/每 rank RNG/数据位置/标定和配置 | 实际中断恢复、AMP 溢出重试、双进程主/辅助更新检查通过 |

这里的理论支撑是**显式量化控制 + 已传输误差尺度条件化 + 残差参数化**。只要 σ 有正下限，`u*=(μ−z0)/σ` 定义明确；尺度停止梯度，前端仍从 μ 接收图像、latent 和表征梯度。

这不证明量化误差加高斯噪声后等于标准 Rectified Flow 轨迹，也不证明局部 RMS 是后验方差。新路径没有加入这样的高斯桥接，更没有把尺度标签冒充 SANA 的原生时间。原始“精确时间同构”的主张不应写入论文结论。

局部分配会改变源特征与熵模型，实际码率未必对每张图片严格单调；共同前端结束必须检查六点实际码率跨度和粗 latent 失真，再进行 C/D 对照。该检查能发现控制失效，不能代替完整 RD 曲线。

## 本轮修复和效率工作

1. Router 的参考概率计算使用 `no_grad`，避免构建最终必定 detach 的梯度图；内容与分配梯度仍保留。
2. 验证按图像复用干净 latent、分析网络、超先验和基础分配图。六个质量点各自执行真实量化/上下文和后端；实际包检查继续走完整压缩/解码接口。
3. 训练指标在设备侧累加，到日志更新才批量转 CPU，减少每个 microbatch 多次同步。非有限损失检查仍保留。
4. checkpoint 使用 8MiB 缓冲写盘、flush/fsync 后原子替换；失败移除临时文件，保留上个完整状态，避免实验卷大量小写入。
5. 初始化 `s_data` 按两个确定裁剪批量编码，不改变采样集合和统计定义。
6. C/D 每卡 **bs=2、累积4、有效 batch=8**；共同前端两卡各 bs=2、累积2，同样有效 batch=8。
7. 保留 SANA block 和 DC-AE decoder stage 的非重入梯度检查点供 Self-Flow 使用。C/D 经完整实测后关闭重算，利用 32GB 显存缩短训练时间。V100 使用 FP16+GradScaler；注意力收缩与归一化、codec 和概率路径显式 FP32，防止 FP16 下溢。
8. 增加 `scripts/launch_bt_pair.py`：锁防重复启动、固定配置/提交、逐阶段恢复、共同初始化后 C/D 分别占 GPU0/1、记录每个进程与日志。失败或暂停不自动伪装阶段完成。
9. 修复硬前向与实际熵解码的张量布局差异：量化后的超先验、四组上下文输入和整数符号统一 NCHW。修复前，实际编码/解码本身逐值一致，但硬前向的 μ 有约 1e−7 的差异，被低码率下的预训练单步后端放大。修复后完整公共模型在 λ=0/0.4/1 的 μ、符号、CDF 索引和最终图像误差全部为 0，证据为 `packet_diagnostic.json`；验收阈值未放宽。
10. BT checkpoint 用 mmap 按需读取，减少恢复时多 GB 状态的预读和 CPU 副本；共享初始化的源文件只计算一次 SHA256。后端/联合阶段结束时直接保存 `complete.pt`，避免重复序列化同一完整状态；共同前端保留标定前恢复点。
11. 正式共同前端标定记录六点粗 latent NMSE。生成共享初始化前，除了至少三个不同实际码率，还要求最高质量平均码率更高、粗 latent 失真更低；失效前端不会自动进入 C/D。14 更新的工程短跑明确关闭此研究质量门槛，其 checkpoint 不参与正式实验。

没有删除 LPIPS/DISTS、降低 LoRA rank、裁减 SANA 层数或用短跑前端替代正式共同训练。

## 验证证据与正式运行

旧 GPU0/1 baseline 已在用户授权后于 2026-10-15 21:57 正常终止，GPU 已释放。旧可用 checkpoint 保留，停止证据：
`/data2/ls/checkpoints/bt_self_flow_v1_launch_20261015/previous_experiments_stop.json`。

优化后的编解码、后端、实际训练恢复及评测回归：21 项通过，报告：
`/data2/ls/checkpoints/bt_self_flow_v1_launch_20261015/review_tests.xml`。
补充行为检查核对“分析复用前后各质量点指标一致”和“写盘中断保住上个 checkpoint”，报告：
`/data2/ls/checkpoints/bt_self_flow_v1_launch_20261015/efficiency_tests.xml`。

完整公共权重、512 裁剪、完整图像损失、bs2 的 GPU 短跑输出：
`/data2/ls/checkpoints/bt_self_flow_v1_launch_20261015/full_gpu_review_fixed/`。
短跑包含两卡共同前端、一次共享初始化、C/GPU0 和 D/GPU1 的后端暂停/恢复及联合阶段、真实码流一致性验证。**只有该目录的 `preflight.json` 为 passed，才算完整 GPU 验收完成。**短跑 checkpoint 不进入正式实验。

该报告现在为 **passed**。两组均完成后端和联合阶段各 14 个真实 optimizer updates，均使用 bs2/累积4。C/D 两个阶段的六质量点实际码流与硬前向图像误差全部为 0。

随后用相同初始权重、数据和损失并行比较关闭重算的完整联合训练，两组各完成 14 更新和真实码流验证，退出码均为 0：

| 联合阶段，512、bs2、累积4 | 开启重算 秒/更新 | 关闭重算 秒/更新 | 关闭重算峰值分配 |
|---|---:|---:|---:|
| C / GPU0 | 6.539 | 4.528 | 22.59 GiB |
| D / GPU1 | 6.492 | 4.566 | 23.50 GiB |

时间取更新 11–14 的平均，未含验证/存盘；正式训练还需更长稳定区间。关闭重算的六质量点图像误差同样全部为 0，1 NFE，未 OOM。
完整记录：`/data2/ls/checkpoints/bt_self_flow_v1_launch_20261015/performance/no_checkpoint.json`，对应目录保存日志、checkpoint 与逐质量验证结果。

正式 C/D 配置据此关闭重算，约减少 30% 单次更新耗时。后续 D_mix/E 对照共同开启重算，为 EMA 教师预留空间；不把这项工程设置当作后端方法收益。

布局修复后的完整回归：118 项通过、0 失败、0 跳过，含新增 CUDA 逐值一致性检查：
`/data2/ls/checkpoints/bt_self_flow_v1_launch_20261015/all_tests_fixed.xml`。
最初 `full_gpu_review/preflight.json` 保留为发现问题的失败记录，不能用作通过凭据。
最终标定/保存调整后的训练、恢复与效率行为补充回归 9 项通过：`final_training_tests.xml`。

正式输出根目录：`/data2/ls/checkpoints/bt_self_flow_v1`。
`pipeline.json` 是阶段/进程/日志的实际状态，`manifest.json` 固定提交、GPU、解释器和配置。
共同前端 50k → 训练集尺度标定/共享初始化 → GPU0 C、GPU1 D，各后端 5k + 联合 25k。
只有 `pipeline.json` 已记录两个后端训练进程和实际日志更新，才能称为 C/D 已启动；共同阶段期间应明确说“两组实验在等待共同前端”。

正式启动命令（在 tmux 内执行）：

```bash
cd /data2/ls/code/DiT-IC-bt-self-flow
/data2/ls/code/venvs/ditic/bin/python scripts/launch_bt_pair.py \
  --root /data2/ls/checkpoints/bt_self_flow_v1 --gpus 0,1
```

同命令重新执行会从各自阶段最近保存状态恢复，不覆盖未完成目录。启动前需提交代码，恢复时提交和配置必须与 manifest 相同。

显存和耗时以完整 optimizer update 的实测记录为准；短跑估计不包括定期验证和存盘。正式训练后再用稳定区间更新预算，不只给其中一组额外训练。
