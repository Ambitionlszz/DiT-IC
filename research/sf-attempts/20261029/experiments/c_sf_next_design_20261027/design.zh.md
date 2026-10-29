# 下一版 C + Self-Flow：把空间预测接回真实压缩路径

> 当前主推方案见 `single_pass_design.zh.md`。用户随后明确要求一次 student 前向完成全部图片的重建和 H8 对齐；本文关于两次 student 前向的建议已被取代，失败证据与文献核查仍可参考。

状态：研究方案。2026-10-27 核对已有代码、训练记录和论文后整理。没有修改正在运行的代码，没有启动、停止或替换实验；没有执行本方案的 GPU 原型。双卡现有运行经宿主机只读检查仍在运行。

## 结论与边界

建议下一次先验证一个小而明确的假设：高精度 EMA teacher 的**空间结构**，能否通过真实 C 的第 8 层，改善受码率约束的编码表示和最终重建。

保留原始 Self-Flow 的局部双质量输入、浅层 student 对深层 EMA teacher，以及完整 student 的主任务。主要修改是：

1. 同时约束真实单码率前向和局部混合前向，真实前向中的 SF 梯度进入付费的 codec；
2. 对齐 teacher 的图像内空间关系，避免巨大公共分量掩盖内容；
3. 混合前向只在相对 teacher 信息更少的位置计算预测损失；
4. 原生视图与混合视图都由共享的完整 C 执行图像重建，空间关系直接从 H8/H20 计算，不再设置可训练预测头或专用辅助读出。

这些是可检验的改进假设。空间归一化不能保证 teacher 有有用信息，前端接到 SF 梯度也不能保证同 bpp 收益。若短诊断发现 teacher 的空间关系没有内容信号，或 C10K 的 H8 已经包含了与 H20 相同的有效结构，就缺少开展长训的依据。

## 已有失败证据

### 差分／高斯 NLL 版本

旧记录的 target RMS 为 0.001522，预测 std 约 0.050001，后者贴近设定下界。日志 SF loss -1.497561 与“预测零、方差下界、半数参与”的理论值 -1.497866 很接近。它支持“这项辅助目标近乎退化”的判断，不支持“整个 C 已经坍塌”的判断。

更有力的任务证据是：六档测例中，直接注入真实 teacher 差分的重建收益均未出现。这个目标的下游用途没有通过验证。当时仍继续 ramp 和长训，判断不充分。

在已有 18K 对比中，该版同 q 平均总 bpp 增加约 3.47%；在重叠码率范围内插值后，PSNR 约 -0.103 dB、LPIPS 约 +0.00299。SF 输入是 detached，不能说“SF 直接对 encoder 的梯度导致了涨码率”；具体因果未证明。

### 当前完整特征＋辅助读出版本

当前方法的实际主线是完整 C；辅助任务则经过“混合输入 → 1–8 层 → P → 21–28 层 → VAE”。辅助读出权重冻结，但输入梯度保留；辅助输入整个 detached，所以 codec 收不到 SF 梯度。推理不用 P。

这不能直接推出“辅助头必然无效”——OneDC 等工作也使用训练期预测头。问题在于当前头、特征目标和梯度路径是否真的改变了真实 C 的任务能力。

已有记录显示：

- 正式更新 C 之前，冻结 C 只训练头 64 次，日志 cosine 已从约 0.162 降至约 0.022。这些点来自不同训练样本，只能说明任务可较快适配，不能当成固定测试集上的 87% 下降。
- 18K 时 cosine 约 0.00049，而同总 bpp 对纯 C 的平均 PSNR 差约 -0.018 dB、LPIPS 差约 +0.00080，尚无明确压缩收益。
- 预检把正确 teacher 特征换成其他图片／空间打乱后，辅助重建图像的 MSE 变化仅约 7e-6～1.4e-5；正确 teacher 并非每个测例都改善 D411。

之前把“读出变化非零”作为通过标准太弱。上述读出结果发生在混合输入基底上，不能证明 native 全 C 的相同层一定无用；但它足以要求进一步测量，而不能继续用小 cosine 作为有效性依据。

目前尚未测量 H20 的空间方差、公共分量占比、不同图片之间的中心化距离和有效秩。因此“公共分量主导 cosine”是重点假设，不是已证明事实。

## 论文给出的约束

- Self-Flow 原论文使用双 timestep 的局部信息差，完整 student 同时执行生成与表征预测；预测头不是替代后半网络的捷径。原论文还显示，某些异质扰动会产生训练推理分布差距。其生成收益不能直接当成压缩收益保证。
  https://arxiv.org/html/2603.06507v1
- iREPA 研究发现，空间结构比整体语义分类能力更能解释其生成实验中的对齐收益；其改进包括卷积映射和目标空间归一化。这启发这里检查公共分量，但它研究的是生成，不是本 codec 的 RD。
  https://arxiv.org/html/2512.10794
- RKD 已提出蒸馏关系而不是逐个特征数值；DINOv3 的 Gram 目标进一步说明，可约束图像内 patch 关系并保留特征坐标的自由度。下面使用这类已有思想，不声称 Gram loss 是本项目的新发明，也不照搬它们的完整训练配方。
  https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html
  https://arxiv.org/html/2508.10104v1
- OneDC 把蒸馏接到从已解码 hyperprior 得出的语义条件，其条件参与实际解码，且压缩模块继续承受 rate loss。这启发“SF 要影响实际传输表示”，不意味着必须复制 OneDC 的结构或额外码流。
  https://arxiv.org/html/2505.16687v2
- When the Teacher Has More Bits 的作者机构摘要指出，直接逼低码率表示匹配高码率表示可能不合适，并采用方差归一化蒸馏。这里只读到了官方摘要，不能据此复述其未读到的具体公式。
  https://www.eurecom.edu/en/publication/8851
- REPA-E 支持研究“对齐梯度进入前端”这一方向，同时也展示了错误目标可能使表示变得容易预测、却损害任务。这里的前端是有熵编码的 codec，区别于该文的 VAE；保持原始图像和 rate 约束仍然必要。
  https://arxiv.org/html/2504.10483v1

## 具体模型与两种 student 视图

### 真实部署视图：所有样本都完成 C 的 RD 任务

令 z 为冻结 AE 编码的 clean latent。按源 C 的方式独立均匀抽 q，完成真实 codec 的精度条件分析、硬量化模拟、熵似然和合成，得到 c_q。

完整 student F_theta(c_q) 仍运行 1–28 层，经原 C scheduler 与 VAE 得到 x_hat_q。额外读取这次真实前向的 H8(c_q)，不改主线输出。

第一轮最小实现直接沿用源纯 C 架构，不新加六档 quality conditioner；codec 内已有的 precision-v2 conditioning 保留。辅助混合图只用于表征预测，不要求它成为真实可解码的混合码流。这样不再把新增 decoder conditioner 的收益混进 SF。

### 局部信息不对称视图

每张被选中样本另外独立抽 q_b，必须执行该 q_b 对应的独立分析与量化，不能复用 q 的 g_a 分析。

构造 c_mix = (1-M)c_q + M c_qb，其中 M 精确覆盖 25% latent 单元，单元内所有通道同进同出。两个 q 不排序，允许相等。保持最初双码率构造。

teacher 取名义质量较高的完整视图 c_H，q_H=max(q,q_b)。q_H 是质量控制值，不能假定每张图的实际 bit 数或重建质量都严格排序。

student 混合视图运行共享的完整 1–28 层，取 H8 做对齐，同时经原 C scheduler 和共享 VAE 重建同一原图，保留“带局部信息不对称的 student 同时做重建与对齐”的要求。它使用混合 latent 自己作为 scheduler 的 sample，不能偷偷换成 teacher 的高质量 latent。

teacher 用 EMA codec 和 EMA SANA 前 20 层产生 H20(c_H)，全程 no_grad。EMA codec 是此次开放前端 SF 梯度后的配套安排，避免 teacher 输入直接随 student codec 的一次更新跳变；冻结 AE 无需复制，teacher 不需要 VAE decoder。

局部监督位置 Ω={i:q_mix(i)<q_H}，由质量来源直接确定。若 q=q_b，则 Ω 为空，本样本没有局部信息差，混合项跳过；真实前向仍有 8→20 的深度不对称。在最高档，真实视图与 teacher 可能输入同质，但混合 student 仍可能有低质量局部。

## 对齐目标：直接预测图像内空间关系

对单张图的特征 H∈R^(N×D)，沿空间去掉每通道公共均值，再求归一化 Gram 矩阵，全部使用 FP32：

    Z(H) = H - mean_tokens(H)
    e(H) = mean_i sum_channel Z(H)[i]^2
    K(H) = Z(H) @ Z(H).T / (e(H) + eps)

K[i,j] 表示两个位置在当前表征中怎样相关。归一化对整张图进行，不对每个微弱通道或平坦 token 单独除以微小方差。它去掉公共平移，对正整体缩放近似不变；对通道的正交变换严格不变，不对任意可逆变换保证不变。

令 S=K(H8_student)，T=stopgrad(K(H20_teacher))。去掉对角线，对有监督的行集合 Ω 计算相对关系误差：

    I = {(i,j): i∈Ω, j≠i}
    d(S,T;Ω) = mean_I (S-T)^2 / stopgrad(mean_I T^2 + eps)

真实视图用所有行，混合视图用质量低于 teacher 的那些行。混合行与所有其他位置建立关系，不要求所有邻接位置同质，因此保留利用较清楚上下文补偿受损位置的任务。

每张选中图片对有效视图取平均，再对图片取平均，得到 L_SF。equal-q 样本只计 native 项。这个归一化让“预测全部非对角关系为零”在 teacher 有足够关系能量时大约得到 1 的误差，不会仅因目标绝对值小就看起来已经学好。

没有可训练 P：辅助目标的下降必须由实际 native codec／SANA 表征的变化实现。两个 student 视图均运行完整 28 层，SANA 第 8 层不用提前变成与第 20 层逐通道相等的向量；这避免强行统一不同层的通道坐标和幅值。目标也可能因 teacher 退化而变容易，故不能只监控 student loss。

不把矩阵计算解释成“恢复了所有缺失信息”。相同 Gram 可以对应不同内容或颜色；不可恢复的纹理仍可能影响关系；固定位置模式也可能产生关系。原始图像 RD 负责内容身份、颜色、细节和码率，SF 仅提供空间结构约束。正确图、错图和空间打乱 teacher 的对照在这里尤其必要。

若 teacher 的中心化能量或非对角关系能量接近数值噪声，这个目标无效，不能靠分母放大后继续长训。EMA 加主 RD 也不提供严格的防坍塌保证：要监测 teacher 目标的能量、有效秩和跨图片差异。

## 梯度与主损失

    L_C = beta(q) R_total + 0.1 base + 4 MSE + LPIPS + DISTS
    D_mix = 4 MSE(x_hat_mix,x) + LPIPS(x_hat_mix,x) + DISTS(x_hat_mix,x)
    L = L_C + ramp(step) * 1.0 * (L_SF + D_mix)

base 与源 C 完全同义，repaired-latent loss 为 0。beta(q)=[16,8,4,2,1,0.5] 对应 q=[0,.2,.4,.6,.8,1]。

- 原生 SF：H8 来自实际 native C 图，不 detach c_q。梯度经过原 SANA 1–8 层、codec g_s、量化 STE 和 g_a，也可沿实际可微路径进入 router／熵上下文。所有涉及的 native 编码表示同时受这次 R_total 和 base 约束。
- 局部 SF：mix 的两份输入 detach，关系对齐更新共享前 8 层，混合重建更新共享完整 SANA 和 VAE LoRA。它不负责单独改写 codec，也没有未计费的高质量辅助编码器梯度。
- teacher：包括高质量视图在内完全 stopgrad。
- SANA 9–28 层和 VAE decoder：由两种视图的图像重建更新。关系矩阵只在训练时计算；原生 SF 的 trunk／codec 修改保留到部署。

rate loss 是软约束，不能保证 q 相同就 bit 数不变。开放前端之后，模型仍可能用更多 bit 换对齐；只有实测同总 bpp 的 RD 曲线能判断这种取舍是否有价值。

## 双卡可执行路径（尚未实测本版显存）

从原始纯 C 10K 完整 checkpoint 重新开始，保留 optimizer、EMA、LR 和样本计数，不从已失败的 SF 权重接着改。512 输入；2 卡×每卡 4×累积 1，有效 batch 8；不够则每卡 2×累积 2 保持 batch。

每卡持有完整 student 与自己的冻结 EMA codec／teacher 前 20 层副本；teacher 无反向激活，不把单卡专门划给 teacher。具体能否保持 4×1 必须实测，不能沿用当前不同计算图的显存值作为保证。

先对每卡随机半个 batch 准备 detached q_b/mix 与 teacher 目标。native 前向计算 L_C+native SF，一次 backward 后释放大图；再计算 detached mix 的完整重建与局部 SF 并 backward。只 step 一次，沿用已验证的统一梯度同步。两项关系损失按“每图有效视图平均”预先给定系数；D_mix 按选中图片求均值，不因半 batch 自动乘 2。equal-q 样本仍有混合重建，但关系项只计 native。

源 C10K 已完成 student warmup。teacher EMA decay 先沿用 0.999，随后用原先 2K ramp 到 SF 权重 1；保留源 C 的独立六档抽样、每 100 步日志、每 10K 完整保存与 LR milestones。V100 使用已有 FP16＋FP32 敏感运算修复，如实记录与源 BF16 的差异。

相较当前版，删除预测头和跳层的辅助读出，混合重建改走共享完整 C，并增加真实 H8 的监督、关系矩阵和 EMA codec。混合重建增加了部分中间层与可训练读出的反向，不能声称比当前版更省显存。512 输入对应 16×16 AE 单元时，每张图关系矩阵仅为 256×256，FP32 约 0.25 MiB；矩阵大小对 token 数是平方增长。理论上具备双卡实现空间，但不能在完成真实 teacher+student forward/backward 前承诺具体显存或吞吐。

## 长训之前必须回答的研究问题

这些是实验判据，不是新的用户审批流程。

1. teacher 目标是否有图像内容？在预留数据上记录公共分量／中心化能量、不同图与同图 token 距离，以及正确、错图、空间打乱 teacher 的预测差异。平坦区域需单独报告，不能把 tiny variance 归一化后伪造强信号。
2. 是否有新信息？在 C10K 的同一预留集合上比较 H8 与 H20 的图像内关系；若本来已高度一致，且 H20 没有显示更多有效内容结构，就没有足够增益空间。旧版 head 准备的不同 batch 日志不能代替这个测量。
3. teacher 能传什么？用轻量 probe 比较 H8 与 H20 对 clean latent 空间结构的可预测性，并比较错图／shuffle。probe 仅用于分析；不加进正式损失。原生输入和混合输入分别报告，不使用跨 latent 基底直接注入 H20 的即时收益作为硬门槛，因为那会混入坐标不兼容问题。
4. 梯度是否到了正确路径？单独记录 SF→g_a、g_s、router、SANA 前缀的范数及与 native RD 梯度夹角。夹角作诊断，不引入梯度投影算法或自动 alpha。
5. 最终有效吗？先看 10K→14K/16K 的机制和三档到六档真实 bpp 曲线，再决定是否扩到 20K/30K。短试没有收益不等于长期一定无效；但信号退化、只有辅助 loss 变化或持续 RD 恶化，不应靠延长训练解释。

纯 C 的已有结果继续作参照。要宣称对齐本身带来收益，还需要对照：保留相同混合重建、仅关掉关系项；完整方案再关掉 native 对齐到前端的梯度。必要时用关系项但去掉 D_mix 的短试分离增强收益。该文档没有启动这些对照，也没有另外占用 GPU 训练纯 C。

Kodak 用于检查趋势；选型应另留验证集，最终在未参与调参的 Kodak／CLIC 等数据上报告。逐图配对 bootstrap 给出不确定性；同一步与同总 bpp 都要报告，增加的训练计算也要单独记。一个 seed 的微小差异不能宣称稳定收益。

## 创新应该如何表述

高码率教低码率、EMA、空间中心化和关系矩阵蒸馏各自都有已有工作；把它们罗列在一起不能当成足够的创新。

本项目值得验证的实质是：**以真实编码前向为锚，把局部双精度的预测任务约束到接收端实际使用、同时受熵码率约束的表示；用更有内容区分力的空间目标，让它参与“有限比特保留什么”的学习。**

如果最终只有辅助 SF loss 改善，或改善仅来自额外重建计算，这个论点就不成立。当前方案去掉新增 decoder 模块和训练专用读出，但仍然有混合视图额外重建；所以同样增强、仅关闭关系项的对照不可省略。

## 核查过的本地来源

- `/data2/ls/experiments/c_sf_redesign_20261027/evidence.json`
- `/data2/ls/experiments/c_dual_rate_task_sf_20261027/gpu_preflight.json`
- `/data2/ls/checkpoints/c_dual_rate_task_sf_20261027/joint/head_preparation.json`
- `/data2/ls/experiments/c_dual_rate_task_sf_20261027/comparisons/trend_12000_18000.json`
- `/data2/ls/code/DiT-IC-c-dual-rate-task-sf-20261027/models/bt_self_flow/dual_rate_task.py`
- `/data2/ls/code/DiT-IC-c-dual-rate-task-sf-20261027/models/bt_self_flow/codec.py`
- `/data2/ls/code/DiT-IC-c-dual-rate-task-sf-20261027/models/bt_self_flow/model.py`
- `/data2/ls/code/DiT-IC-c-dual-rate-task-sf-20261027/models/bt_self_flow/losses.py`

当前最有把握的结论是“既有对齐指标不能代表压缩收益，下一版必须验证内容、路径、码率三者的联系”，不是“这套新方案已经证明 work”。
