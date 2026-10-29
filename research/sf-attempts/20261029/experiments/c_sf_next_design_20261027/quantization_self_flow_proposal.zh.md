# 以真实量化信息差为基础的 C + Self-Flow

状态：2026-10-27 已被用户进一步澄清的“以 C 为起点、重点结合生成式压缩与信息不对称”取代，不作为当前实施建议。后续讨论见 c_native_generative_sf.zh.md。本文件保留为未实施的研究记录。用户否决了以重建梯度对 teacher 建议评分的方案，该方向同样撤回。

## 核心假设

把 Self-Flow 的局部高低噪声换成同一编码表示的粗细量化。student 缺失的是其实际码流没有携带的细化信息，重建和特征预测使用同一次 student 前向，SF 梯度通过量化近似回到编码前端。

teacher 的信息优势来自同源量化信息的包含关系，不以其图像指标更好为前提；信息更多也不保证每个 teacher 特征都更有用，压缩净收益仍是待验证的研究假设。

## 编码视图

1. 用一个全局 q 控制原有 RD 权衡和一次 analysis：z = AE(x)，y = g_a(z, q)。当前 precision-v2 的 g_a 依赖 q，不能把另一个 q 的 analysis 当成同一份 y。
2. 在同一份 y 上定义嵌套量化层级：粗量化符号加条件编码的细化符号。细层包含粗层的符号，不把两个独立量化器的重建值大小关系当作信息包含关系。
3. 按原版 dual scheduling 的思路抽两个精度层级 a、b，在编码网格上以局部 mask 分配给 student。两个层级相同时自然得到整图统一精度的样本。student 的局部层级图是 codec 的正式输入和码流的一部分。
4. student 实际量化/重建使用混合层级；teacher 在全部位置获得 max(a,b) 层对应的信息。两路源于同一份 y 和同一套量化区间树。teacher 的细化信息只用于训练，不进入 student 的 sample、skip 或图像重建。
5. 真正的熵编码必须按粗层和条件细化符号实现，局部层级图、hyperprior、码流头均计入学生总 bpp。不能在两个独立 decode 结果上拼图并沿用单档 rate 作为本方案。

全局 RD 质量 q 与局部量化层级是两个控制量：前者决定总体权衡及 encoder 条件，后者决定各位置收到多少量化信息。具体层级范围及其与现有六档精度的映射尚需设计和验证，不能声称原 C 的 q→bpp 映射会自动保持。

## 单次 student 前向

512、每卡 batch 4 时：

    z:        [4,32,16,16]
    y:        [4,320,8,8]
    c_S:      [4,32,16,16]     # 混合精度码流实际解出的输入
    H8_S:     [4,256,1152]     # 从唯一一次完整 28 层前向读取
    x_hat:    [4,3,512,512]    # 同一前向的 CStep + VAE 重建
    H20_T:    [4,256,1152]     # no_grad teacher prefix20

四张图全部重建、全部对齐，一一对应。P(H8) 是原版 SF 的训练预测头，后续重建继续使用真实第 9–28 层，不让 P 冒充第 20 层。

teacher 使用 EMA synthesis/条件模块及 EMA SANA prefix20，输入为同一份源 y 生成的、stopgrad 的细层符号。无需另跑 EMA g_a 生成另一个 latent：否则两边就失去“同一符号层级”的定义。teacher 不运行图像 VAE，也不计算另一份图像 loss。

## 简单目标与梯度

    L_C  = 4*MSE(x_hat,x) + LPIPS(x_hat,x) + DISTS(x_hat,x)
           + 0.1*base(c_S,z) + beta(q)*R_student_total
    L_SF = mean_(image,token) [1 - cosine(P(H8_S), stopgrad(H20_T))]
    L    = L_C + lambda_SF*L_SF

SF 外层最终权重 1。没有重建梯度评分、teacher 门控、残差特征目标、额外辅助图像 loss 或新的关系 loss。

L_C 更新原有可训练前端、后端及图像 decoder。L_SF 经 H8、c_S 和量化近似更新 synthesis、analysis 以及 SANA 前八层的可训练参数；教师分支整体 stopgrad。量化梯度是训练近似，并非离散码长的精确导数。

这种耦合让 encoder 在率约束下学习可用于重建和上下文预测的表示。它不保证码率一定下降：encoder 也可能为降低 SF 而多编码信息，需以相同真实总 bpp 验证。

## 研究边界

- 原版 Self-Flow 提供 dual scheduling、EMA、更完整视图和浅层→深层对齐；本提案的压缩适配是将视图差落实到实际量化/熵编码瓶颈，并使 SF 训练前端。
- 嵌套量化已有 PLONQ 等先例，不是本提案独创。现有 precision-v2 的均值中心化分配图及四组上下文/LRP 不自动满足这里的层级语义，需要改量化与条件熵编码。
- 当前 C 权重可作初始化；量化/熵模型变更后，旧 10K 的训练状态与旧 rate 模型不能冒充新 codec 的精确续训。需要先适配新增模块，再进入联合训练；适配时长未确定。
- 公平隔离 SF，需要同一新 codec、同一局部精度训练分布关闭 SF 的对照。现有纯 C 曲线可比较整套系统，但不足以归因 SF 本身。不因此启动本机纯 C。
- 主干仍是一次 full student 加一次 no_grad teacher prefix20；双卡每卡 4、积累 1 是目标配置，显存、吞吐、真实码流及稳定性均未预检，不作已能开训的承诺。
- 高码率保留细化信息并由后段完成重建，低码率借上下文预测表示。两端同时改善是待验证目标，不是由 cosine loss 降低即可推出的结果。

参考：

- Self-Flow: https://arxiv.org/html/2603.06507v1
- PLONQ: https://arxiv.org/html/2102.02913v1
