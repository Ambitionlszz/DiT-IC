# C + Self-Flow：真实局部编码精度实验

本实验从原纯 C 的 joint10K 续训，目标是改善正式单质量 C 的实际总 bpp—画质曲线。SF loss 下降、梯度非零和 teacher 更深都不作为压缩收益的结论。

## 数据流与损失

每图只以预算质量 qA 分析一次，得到 y、hyper 和 router allocation。独立抽 qB；每卡四图中随机两图为原生 qA，另外两图的随机 25% 编码 cells 使用 qB。各 cell 使用原 C 步长公式，并经相同四组熵编码/解码和精度条件合成得到唯一 student 输入。

这份输入完整经过 DiT28、原 C endpoint、VAE，计算一次 `4*MSE + LPIPS + DISTS + .1*base + beta(qA)*R`。H8 从该次正式前向读取，没有额外 student 前缀、P、跳层解码或 Dmix。

teacher 输入在同一个 y、hyper、allocation 上以全图 max(qA,qB) 重新量化和合成；整个参考构造无梯度。teacher 本体为 EMA DiT20、文本投影和精度输入投影，没有第二套 codec 或 VAE。

student 和 teacher 都将可由 packet 恢复的局部 log(delta) 最近邻映射到 AE token 网格，经零初始化的 `Linear(1,1152)` 加入 patch embedding。新增参数 2304 个，属于正式部署网络，同时加入主优化器、部署 EMA、teacher EMA 和 checkpoint。

SF 采用空间中心化后逐 token 通道归一化的 cosine 距离，只监督 student 步长大于 teacher 的位置。每图按有效 token 数取均值，无有效 token 时贡献零，再按整个 batch 平均。SF 梯度经实际 student 进入 codec；teacher 参考全部 detach。SF 权重1，10K→12K 线性 ramp，不做权重校准。

## 编码与计费边界

显式 `local_precision_protocol` 扩展原 precision-v2，`LocalPrecisionControl.analysis_quality` 仅用于编码端 g_a/router 分析锚点；继承的双 q 与 mask 字段定义实际量化。旧协议继续拒绝 mixed 或跨 q 复用分析结果。

decoder 只需原 BT packet 的 q16、allocation3 和 mixed mask，没有原图、qA 分析锚点或未传输的连续条件。全局 qA 始终用于 RD 的价格；不能误用排序后的 packet.low 或 teacher 的 qH。

训练 rate 由实际 student 的 main/hyper likelihood 加全量 side bits 构成。半 batch 原生样本按单 q packet 计费，混合样本加 mask；六档相同 qA=qB 时全部退化为原生单 q。512 图对应 8×8 编码网格，混合 mask 为64 bit；主流和 hyper 的字节成本另算。评测使用实际全部 packet 字节。

全局单 q 部署仍有 router 造成的局部精度变化，部署精度条件也来自同一 packet。这是局部观察训练与正式单 q 解码的接口。局部双 q 仍改变训练分布，不能声称完全等同旧 C。

## 训练配置

- 原始来源：`c_precision_v2_lambda16_C_joint10k_20261023/0010000.pt`，SHA256 `21d23b3beef59a9b30e08a11776b14720041f994f742c6de961c0ee2375c8ab6`。
- 保留原 raw/EMA、Adam/aux、scheduler、现有各 rank RNG 和80000张全局样本进度；仅精度投影新增零参数及空 Adam 状态。新 FP16 GradScaler 起始64。
- 两张 V100S 32GB，每卡4图、积累1、有效batch8，输入512。每卡本地 EMA teacher，无 teacher 跨卡传输。
- 六档 q 独立均匀抽样；beta 为 `[16,8,4,2,1,.5]`。qB 独立 counter stream，不改变原 qA 或数据采样流。
- LR1e-4，aux1e-3，clip4，原200K milestone契约。第一段到20K；12/14/16/18/20K 在 Kodak24 六档做 EMA 全图实际码流评测。
- 日志100步，常规保存10K；验证发现新最佳时亦保存，正常分段退出保存完整 paused 状态。

## 稳定性修复与验证

旧13.6K故障批次的首个非有限 leaf 为 DiT block8 self-attention `to_out.0.lora_B.default`：FP16 输入有限、输出 inf。全 FP32 前向正常。旧 post-projection clamp 发生在 LoRA 中间运算之后，无法保证其内部有效。

将 self-attention 的投影、LoRA乘积、归一化及输出保持在 FP32，保留权重和运算拓扑；其它 AMP 运算仍正常使用 FP16。原故障批次经修复前向有限，H8 最大幅度约3242.72；全 FP32 约3242.68。此修复同时应用 teacher，并非依靠降低 SF 权重或无限自动重启。

真实512初始化验证：六档原生零投影输出与共享相同数值修复的原 C 完全相同；六档及高低两个方向混合 packet 的 hard forward/decode 最大误差均0。SF-only 梯度进入 g_a、g_s、router、精度投影和 DiT前8层；后20层/VAE/teacher在此项中无梯度，仍由主重建任务训练。

初始同批内容诊断：正确目标约0.510，错图0.814，空间打乱0.980；这是目标可区分内容的证据，不是压缩收益证据。

完整双卡满权重200次连续更新检查已通过，无非有限前向或 AMP 跳过更新；另已真实加载完整检查点，从10200续至10201。两卡 NVIDIA 实际占用各22828 MiB（约22.3 GiB），PyTorch峰值分配约20.1 GiB，每步约2.55秒。检查另存于实验目录 `stability`。其权重不用于正式初始化；正式仍从原纯 C10K 开始并使用2K ramp。最终测试与显存结果记录在运行目录，不将未完成检查写成通过。

运行脚本：`scripts/migrate_local_precision.py`、`scripts/check_local_precision_gpu.py`、`scripts/launch_local_precision.py`；配置：`configs/bt_self_flow/c_local_precision.json`。

本实验同时改变局部编码训练、正式精度条件和 SF 目标。外部旧纯 C 可评估整个配方；若要把收益单独归因 SF，需要相同新配方但关闭 SF 的对照。本次只启动已授权的一条 C+SF 双卡训练。
