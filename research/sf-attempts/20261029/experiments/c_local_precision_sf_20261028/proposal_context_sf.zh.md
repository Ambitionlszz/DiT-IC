# C 的 Self-Flow 上下文条件模块：候选方案

状态：设计建议，尚未实现、预检或启动。当前训练不受影响。该方案能从结构上移除已发现的 SF 直接增码率路径与混合 RD 任务改变；能否提升压缩效率仍须实测。

## 主张

让浅层从已经收到的码流预测深层上下文，并把预测出的同一份上下文直接用于后续正式解码。Self-Flow 监督这份实际参与生成的条件，不再仅要求整个 H8 与 H20 相似。

当前 C 使用固定文本 embedding 经投影构造 cross-attention 条件；不是随图片变化的语义条件。图像信息主要由输入 latent 和 self-attention 提供。这里新增的是从当前图像的压缩证据导出的上下文条件，不增加发送的语义码流。

## 原生出图

1. 图像在原生 qA 下经过完整 precision-v2 C 编码，得到实际可解码 cA、likelihood、allocation 和 deltaA。
2. cA 经同一 student 的第 1–8 层，得到 H8。主空间状态 H8 保留，接着继续第 9–28 层。
3. 新模块 G 把 H8 的每 2×2 个 AE token 汇聚为一个 coding-cell 上下文 token，再用小型残差 MLP 变换。可将 packet 可恢复的 log(deltaA) 经线性映射加入模块输入。输出 U_A 的宽度为 1152；512 输入时形状是 B×64×1152，任意尺寸按真实 coding grid 扩展。
4. 第 9–28 层增加一条从当前空间状态读取 U_A 的 cross-attention 残差。原本的固定文本条件路径继续工作。图像条件读取复用各层现有 cross-attention 的 Q/K/V/输出投影权重，新增可学习残差强度，零初始化保证初始完整输出等价于源 C。
5. 仍经过全部 28 层、原 C 更新公式和 VAE 出图。没有 H8→H20 的状态替换，也不跳过第 9–20 层。

G 采用 LN→1152→256→1152 的小残差 MLP，共约 59 万权重，而不是 531 万参数的训练专用 P。context 的 2×2 pooling 与 coding grid 对应，不是人为指定固定个数的语义槽。G 输出保留其内容和幅度，供稳定归一化后的注意力读取；SF 的空间中心化只用于损失计算。

可学习残差强度是普通的模型参数，由主图像 RD 训练；不是根据 PSNR、损失梯度或 teacher 正误打分构造的门控。新增 image attention 应复用已修复的 FP32 投影与注意力数值路径。零初始化只保证起点等价，不保证该分支随后一定被有效利用。

## teacher 与 SF 任务

- 随机独立采样六档中的 qA、qB，保留相等情况，不强制小质量差。
- 同一原图在 qB 下构造完整、独立的 C 分析与解码视图 cB；不能沿用 precision-v2 的 qA 分析结果跨档重量化。辅助 codec 运算均无梯度。
- EMA teacher 输入 cH，即 qH=max(qA,qB) 的完整视图。teacher 是 EMA DiT 第 1–20 层，加上其实际执行所需的 G 和图像条件读取；不复制 codec 或 VAE。teacher 在 H8 也生成自己的条件并继续到 H20，保证与 student 是同一架构的前缀。
- teacher 目标为 T=Pool2×2(H20_teacher)。Pool 是固定操作，没有一个可通过单独调整而把目标变容易的 teacher predictor。
- 对全部四张原生出图的 U_A 计算 SF；这些 token 正是后 20 层读取的条件。最高档也接受同质量的浅层→深层监督，不要求一定存在比它更高的一档。
- 每卡随机选两张，把 detach(cA)、detach(cB) 按随机 25% coding cells 的 mask 混成 cMix。各 cell 的所有通道共同选择，mask 对齐到 2×2 AE token；log(delta) 按相同 mask 混合。这个混合观察只经过共享 student 第 1–8 层和同一个 G，得到 U_mix，对齐同图的 T。
- qA=qB 时复用原生观察，不重复执行同一混合 prefix。

这样保留随机双质量、student 内局部不同信息量、teacher 全图较完整观察，以及浅层→深层自蒸馏。混合观察只作为上下文预测任务，不宣称完整复刻原文在混合输入上同时生成与对齐的训练。

辅助 cMix 是训练观察，不是原生低率 packet；它不进入正式 RD，也不进入正式出图作为高质量侧信息。

## 损失与梯度合同

    L_C = 4*MSE + LPIPS + DISTS + 0.1*base + beta(qA)*R_A
    L   = L_C + 1.0*L_SF

R_A 包含原生主码流、hyper 和所有原生 side bits。beta 仍为 [16,8,4,2,1,0.5]，六档 iid 原生质量采样不变。repaired-latent loss 为零。

L_SF 使用空间中心化后的上下文 token cosine，与固定 pooled teacher 目标比较。对每张图先平均其可用原生/混合观察，再平均 batch；没有另一个混合重建损失、逐区域图像指标门控或梯度校准出的 SF 系数。

- L_C 正常训练完整可训练 C、新模块和 VAE decoder LoRA。
- L_SF 只更新 DiT 第 1–8 层 LoRA 与 G，不能更新 g_a、g_s、router、熵模型、编码侧精度条件或 VAE。原生 log(delta) 的 SF 梯度同样不能进入 router。
- 原生出图仍保留完整 RD 到 codec 的图，不能通过把 cA 整体 detach 来实现隔离。实现可对 SF 使用明确参数白名单的 autograd.grad，再与主 RD 梯度累加；双卡必须在最终累积后进行一次一致的梯度平均。
- teacher 和所有辅助视图构造 stop-gradient。EMA 包括 teacher 前缀真正使用的 LoRA、新 G 与残差参数。

因此 SF 不再直接奖励编码器放大 latent 或改变 allocation。联合阶段 RD 仍然可以改变码率：这个方案移除直接 SF 梯度通道，不承诺数学上锁死最终 bpp。

前端的联动来自正式图像损失：G 和 DiT 若能利用同一份码流恢复得更好，完整 RD 梯度才决定哪些信息可以少传。不能把“前端有 SF 梯度”本身当作压缩适配成功。

## 训练与资源建议

候选起点仍是用户提供的纯 C 10K，不能接当前已漂移的 SF 权重。512，两卡每卡 4、积累 1，有效 batch 8；从 10K 后的第一个更新起启用 G、图像条件读取和权重为 1 的 SF，不设 SF ramp。

按用户修正，从 10K 起直接端到端联合训练到第一段 20K，不设 codec 冻结或解冻阶段。codec/router/熵模型及其原有精度条件从第一个更新起正常训练：原生 L_C 和原有 entropy auxiliary objective 保持原职责；L_SF 始终只更新前 8 层 LoRA 与 G。新条件读取的残差强度零初始化以继承源 C 的初始出图，由主 RD 从第一步开始学习，不是单独的预训练或延迟启用阶段。继续按 12K/14K/16K/18K/20K 的原生真实码流评测跟踪工作点与同 bpp 收益。

每卡成本：4 张完整 student + 2 张混合 prefix8 + 4 张无梯度 teacher prefix20 + 必要的无梯度 qB codec。teacher 可分块。只有一次完整 student/图，但确实存在额外 prefix8；不能承诺“总共只算一次 student”，也不能在预检前保证 4×1 的显存。辅助 backward 顺序执行，避免同时保留原生和混合计算图。若 4×1 不足，需按原有约定采用 2×2，并重新保证混合抽样与有效 batch 语义。

## 必须证明什么

1. 零残差初始化时，原生 C 出图、真实 packet 和源 10K 一致；关闭新条件应精确回到原路径。
2. SF 单独反向：codec/router/熵模型梯度严格为零，prefix 和 G 有梯度；主 RD 反向仍能到达所有原可训练 codec 部分。
3. 在评测时固定同一份 packet 和解码模型，比较正确上下文、关掉上下文、错图上下文和空间错位上下文的真实重建；这只是离线诊断，不引入固定 codec 的训练阶段。观察“输出变化”或“SF 很低”仍然不够。
4. residual 强度、图像条件贡献和上下文内容应有实际作用。如果分支被忽略，即便它在部署代码中，也不能称已解决旧版收益转移问题。
5. 联合阶段按相同实际总 bpp 比较低/中/高码率，尤其最高档额外 bits 能否收回。架构改进与 SF 的因果贡献需要“同模块、无 SF”的对照；只与已有纯 C 比较只能评价整个配方。

未解决的研究风险：teacher 的深层空间上下文未必是最有用的生成条件；注意力可能学会忽略它；混合 prefix 可能带来不良表征压力；压缩后的上下文在高码率上的边际收益可能很小。该方案增加了合理的归因路径和可证伪验证，不等于保证涨点。

## 与已有工作的关系

- Self-Flow 提供随机双时间步、局部信息差、更完整 EMA 观察和浅深层自蒸馏。https://arxiv.org/html/2603.06507v1
- OneDC 使用 hyperprior 提供图像语义条件，并引入语义蒸馏，说明压缩证据生成图像相关条件是已有方向；本提案不能将“latent-conditioned generation”本身称为新贡献。https://www.microsoft.com/en-us/research/publication/one-step-diffusion-based-image-compression-with-semantic-distillation/

这里要验证的具体新主张是：用 Self-Flow 把压缩观察的浅层特征训练成后续正式解码直接读取的条件，同时保留原生 RD 任务并隔离 SF 对传输表示的直接激励。是否具有论文层面的新颖性仍需更完整的先行工作检索。
