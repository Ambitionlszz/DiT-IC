# 从 C 出发：对生成修正分支的局部观测做 Self-Flow

状态：2026-10-27 待修订的历史讨论方案。用户明确要求参考原文随机抽两档信息等级，固定 2*Delta 的局部粗化建议撤回，不作为实施依据。原文独立抽取 t,s~p(t)，student 按随机 mask 分配两档，teacher 取 min(t,s)；向 C 的质量 q 迁移时应讨论随机双 q 以及 teacher 的较细一档，并单独解决实际码率与主重建的对应关系。当前运行代码 auxiliary_views 已有双 q 视图，因此随机双 q 本身不是下一版的新改进。本文未实现、未做 GPU 预检、未变更现有训练；以下保留为此前讨论记录。

## C 提供的结构

本机 models/scheduler.py 的真实更新是：

    z_hat = c - sigma_C * SANA(c)

其中 c 是实际码流解出的 coarse latent，sigma_C 是 C 原有的固定单步系数。c 直接进入结果，SANA 预测修正。这提供了一个自然的设计边界：完整编码表示继续作为基底，让生成修正分支学习从不完整的局部观测预测完整条件下的表示。

## 唯一核心改动

在同一份已经量化、上下文解码并做完 LRP 的 coded latent y_hat 上，构造局部较粗的训练观测。设 Delta 是本次真实 codec 使用的局部量化步长，M 是与图像指标无关的局部随机 mask：

    y_view = (1-M)*y_hat + M*Q_(2*Delta)(y_hat)

这里 Q 是再量化训练增强，通过 STE 提供近似梯度。它不是当前四组编码符号的严格最低位截断，也不是合法的低码率替代码流；不得宣称因此省下了任何编码比特。因为 y_hat 已含预测均值和 LRP，Q 的零点网格只是一个明确的局部粗化算子，不把它冒称为原量化区间的嵌套关系。

量化发生在 coded latent 的单位里，避免把 Delta 直接加到不同单位的 AE latent。仅重用现有 synthesis 得到 c_view；针对再量化的位置以 2*Delta 作为增强视图的局部精度条件，其余用 Delta。原生 c 及其 Delta、router code 和熵率保持来自正常 codec 路径。上述增强条件不是实际 packet 精度，不能混入真实码率或测试解码。

g_s 含空间卷积，因此“局部”指 y_hat 上选中的 coding cells，不承诺 c_view 的变化严格限制在对应的 2x2 AE cells。

## 数据流

    x -> frozen AE -> z
      -> original C codec(q)
         -> y_hat, Delta, R_native
         -> g_s(y_hat, Delta) = c

    y_hat -> local coarsening -> shared g_s -> c_view
    v, H8 = full_student_28_layers(c_view)  # 一次完整 student
    z_hat = c - sigma_C*v                 # 基底严格用原生 c
    x_hat = VAE(z_hat)                    # 一次图像解码

    H20_T = EMA_SANA_prefix20(stopgrad(c)) # 同 q、同一实际码流
    L_SF  = mean(1-cosine(P(H8), stopgrad(H20_T)))

四张图全部重建、全部对齐。P 只用于原版 SF 特征预测，主重建正常执行第 9–28 层。teacher 不运行图像 VAE。没有高 q teacher、没有另一个 g_a、没有第二次 student 前向或辅助图像重建。

teacher 的输入与 student 来自同一个实际码流。teacher 的优势是看到未粗化的编码观测，并使用 EMA 的更深层，而非默认高码率图像输出或某个位置的特征一定更正确。teacher 同样可能产生无用目标，主 RD 共享训练并不构成收益保证。

## 目标与前端联动

    L_C  = 4*MSE(x_hat,x) + LPIPS(x_hat,x) + DISTS(x_hat,x)
           + 0.1*base(c,z) + beta(q)*R_native_total
    L    = L_C + ramp*1.0*L_SF

R_native_total 计入完整原生 packet 的所有符号和边信息；训练似然是 rate 近似，最终比较真实编码总 bpp。coarse 训练视图不独立计费，也不提供比原生码流更多的源信息。其作用是削弱生成修正分支的观测，完整码流始终保留在 C 的重建基底中。

L_SF 从 H8 经 c_view、共享 g_s、STE 再量化和原生 codec 的量化近似回到 g_a/前端可训练参数。teacher 的输入与参数均无梯度。实际编码预算由原有 R 约束，图像保真由同一次 4/1/1 重建约束。前端可能为了 SF 编码更多信息，不能把直接联动自动解释为码率下降。

## 原生任务与训练推理差异

每卡 batch 4 可以由两张 M=0、两张局部粗化组成，四张一起经过唯一一次 student 并全部对齐。粗化样本初始可在 8x8 coding grid 选 25% 的 cells，一次只降低一级观测精度；这些是待短试验证的起始设置，不是理论最优值。

M=0 时 c_view 必须与 c 数值一致，关闭 SF 后应精确恢复源 C 的行为。推理全用 M=0、无 teacher，原生量化、码流、SANA 和 CStep 路径不变。加入正常 C 样本可持续训练原生任务，但增强样本仍有训练推理差异，不宣称消除该差异。

从用户给的纯 C 10K 完整 checkpoint 初始化，原生参数、优化器、LR、有效 batch 8、iid 六档 q 与原 C 损失保持一致。新增 P 和 EMA teacher 单独初始化，SF/增强可用 10K→12K ramp 接入。此时间表是候选方案，尚未实施。双卡每卡 4、积累 1 为目标配置，真实显存需预检。

## 压缩假设与待验证事项

低码率：在较少局部证据下学习利用邻域结构形成修正表示，目标来自同预算下完整观测的 EMA C，不要求恢复一个额外高码率 teacher 的所有细节。

高码率：已编码细节继续通过原生 c 进入结果，正常 C 样本持续训练完整观测下的修正，降低“为了对齐而抹掉细节”的压力。仍不能保证高码率无退化。

相比此前运行版本，本方案将 reconstruction 与 SF 合到一条真正的 full student 路径，恢复 SF 到前端的梯度，同 q 同编码坐标构造两档观测，取消 P 冒充 H20 的辅助 readout 和额外图像目标。原始 cosine 仍可能被无用公共分量主导，需在不增加训练门控的情况下诊断，不能以 cosine 很低判断成功。

直接与现有纯 C 比较可以判断整套训练配方是否改善系统。把收益单独归因 SF，还需要相同局部粗化、关闭 SF 的对照；本讨论不启动额外本机纯 C。

参考：Self-Flow 的双噪声局部调度、更完整 EMA 视图和浅层到深层特征预测 https://arxiv.org/html/2603.06507v1 。本方案沿用其信息不对称思想，以 C 的生成修正路径适配压缩，不宣称照搬原文噪声边缘分布。
