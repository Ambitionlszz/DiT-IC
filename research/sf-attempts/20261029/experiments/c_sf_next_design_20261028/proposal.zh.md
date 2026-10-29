# 下一版 C + Self-Flow：在真实编码精度上做局部不对称

2026-10-28，研究方案，尚未实现或启动。目标是正式单质量C的实际总bpp—画质曲线超过纯C；不把SF下降、teacher更深、或梯度非零当作压缩成功。

## 设计判断

下一版优先改变任务发生的位置：局部双质量由C的真实量化环节产生，这一份student输入完整承担生成、RD和Self-Flow。随机双质量应作用于同一个分析表示的编码精度，不能把两个质量条件下独立分析/合成后的latent拼接，解释成一条原生低码率输入。

Self-Flow原文的混合观察同时承担生成和浅到深的特征对齐，且网络得到局部时间条件。我们上一版的mixed观察只经过前8层做关系学习，正式生成由另一种输入完成。因此它没有检验完整的“从局部信息不对称中学习生成”的假设。

另一个主线连接是C已经有局部精度：即使全图使用同一q，router的allocation也会改变各编码cell的量化步长。把这个码流可恢复的精度条件交给DiT，才让双质量训练学到的“利用更可靠上下文”有机会在正式单q解码中被调用。

## 1. 一份分析表示，两种实际量化精度

每次抽原C的主质量qA，以及独立的qB，均来自六档。qA保留原来的独立均匀抽样和全图RD系数。每张图只在qA下执行一次原有分析变换和router，得到y、hyperprior与allocation a_i。

这是有意保留的C分析端：qB改变局部量化精度，不重新执行另一套qB分析，也不把两套C的合成输出做拼图。

对一张混合训练图，在编码网格上随机选择25% cells令q_i=qB，其余q_i=qA：

    log delta_i = (1-q_i)*log(4) + q_i*log(0.5) - kappa*a_i

这是原C的步长公式；随机的是质量等级，不固定把步长乘2。随后按delta_i执行原有分组量化、熵模型、局部精度条件合成变换，得到唯一的student输入cS。

这一路必须支持真实packet编码/解码。训练的rate来自这一路的main/hyper likelihood及全部side bits，评测用实际字节。两档q和局部mask需能从packet恢复；原有packet已经有相关字段，但precision-v2实现目前禁止mixed control，需要一个显式的新协议扩展。512时编码网格8×8，若使用现有逐cell 1bit mask，额外mask为64bit=8byte，约0.000244bpp；main/hyper rate的变化另算，不能只报mask成本。

混合流通常可能比全低q花更多bits，必须按其实际码流计费。teacher的训练参考信息不传给student，也不计入student码流。

## 2. Teacher：同一表示的更细量化参考

设置qH=max(qA,qB)，使用同一个y、hyperprior和allocation，在全部cells使用delta_T,i=delta(qH,a_i)，经过共享当前codec的无梯度量化/合成得到cT。

teacher自身仍只有EMA DiT前20层、文本条件投影，以及新增的DiT精度输入模块；没有第二套EMA codec、VAE或熵模型。共享codec只负责构造无梯度参考。teacher在其实际delta_T条件下输出H20T。

同一分析表示使两份观察的差异主要落在量化精度，而非同时改变分析变换的表示。注意，C的六档量化网格不是已证明嵌套的码流层级，合成变换/上下文也存在非线性；更细量化不能被宣称为逐图重建必然更好。这仍需测量，不把teacher输出当作原图替代物。

## 3. Student：同一次完整前向生成与对齐

每张图的cS经过完整DiT28层、原C endpoint更新、原VAE，计算一次原来的4/1/1重建损失。在这个完整前向中读取H8S，对齐H20T。不增加第二次student前缀、不使用专用P或辅助跳层读出。

双卡各batch4、积累1：每卡4张全部完整重建。建议每卡固定2张整图单q、2张局部双q；后者的25% cells取qB。整图单q就是M=0，使用同一条模型/码流路径。每token的q边缘分布仍来自同一组随机质量，但训练的空间联合分布和部分RD任务与旧C不同，需要如实作为配方改动报告。

保留单q样本是为了直接训练最终测评情形；不能声称局部双q与单q完全没有分布差异。单q时router自身仍提供局部精度差异，是上下文利用能力通向部署的具体联系。

## 4. 给DiT原生可恢复的局部精度条件

decoder可以从q、allocation、必要的mixed mask恢复每个编码cell的log delta_i。将它按既有合成空间尺度映射到AE token网格，以一个零初始化的小型输入投影加入DiT patch embedding；该模块属于正式DiT，在部署中也使用，teacher同步EMA它的参数。

log delta描述编码精度，不是已知的真实量化误差。它在g_s合成后也不是每个AE token实际误差方差的精确值。不得用原图误差、训练专用clean latent或未传输信息作为部署条件。

零初始化允许原生单q C10K的初始出图保持不变；无需丢弃50K warmup或换用已经失稳的SF权重。新增条件的收益必须与SF的收益区分。

## 5. 一个简单的SF目标

先回到逐token特征预测，避免把完整Gram关系方向当作唯一学习目标。使用无可学习参数的空间中心化与token归一化：

    phi(H)_i = normalize_channels(H_i - mean_tokens(H))
    d_i = 1 - cosine(phi(H8S)_i, stopgrad(phi(H20T)_i))

在student实际使用较粗质量、teacher使用较细质量的cells所对应token上取平均；每张图没有此类位置时SF为0，最后按整个batch取平均。选位置依据抽样出的精度图，不依据原图误差、梯度utility或人为重要性评分。编码cell到AE token的对应沿用空间上采样；由于g_s具有感受野，这只描述该区域的编码精度标签，不代表误差严格局限于该区域。

这样要求低信息位置预测具体内容特征，逐token的通道关系也受到约束；它不再容许student仅作任意通道正交旋转就保持同一Gram目标。空间中心化旨在减弱共同成分；它不保证目标一定对纹理和RD有用。

归一化只用于比较，不代替数值稳定性修复，也不保证内部激活范数有界。不加第二份图像损失或对比学习loss，不用目标交换诊断量动态开关训练。

    L = 4*MSE + LPIPS + DISTS
        + 0.1*base + beta(qA)*R_student_total
        + ramp*1.0*L_SF

beta(qA)沿用[16,8,4,2,1,.5]。qA是整张图的预算/价格锚点，不把高q teacher的beta误用给student；也不把不同区域再拼出一串图像loss权重。混合图和单q图都在同一主重建项中，每张图只计算一次图像损失。

## 6. 为什么这一版更值得检验

- 混合输入的H8就在实际完成生成的同一次student前向中，不能由训练专用读出单独解决任务。
- 局部信息差来自同一个C分析表示的实际量化；更可靠的student局部也付出真实rate代价。
- SF和图像RD均经过这份实际输入与量化STE进入codec，编码器有机会学习如何在预算内保留有助于上下文恢复的信息。
- DiT直接看到正式码流可恢复的局部精度，训练出的上下文利用机制能在单q+router的精度差异中使用。
- 较粗区域接受自监督，较精细位置仍由真实图像任务约束；不强行要求高质量可见细节对码率完全不变。
- 使用逐token预测后，目标比全图Gram方向更具体；实际会不会改善压缩仍是实验问题。

这些给出机制上的理由，不提供C+SF>C的保证。尤其不能保证有限bpp可恢复任意teacher细节；teacher不可靠、目标被共同位置结构主导，以及低码率目标难以预测的问题仍需诊断。

## 7. 与上一版的可核对差别

| 项目 | 上一版 | 本提案 |
|---|---|---|
| 双质量位置 | 两个独立q的C输出后混合 | 一个分析y内的局部量化精度 |
| 混合student | 前8层，只做关系学习 | 完整28层+VAE，承担主重建与SF |
| student计算 | 全部原生+额外mixed prefix8 | 每张一次完整student |
| mixed rate | 辅助输入不对应正式packet | 对mixed自身的真实码流完整计费 |
| DiT条件 | 固定原C时间，无显式局部精度 | 正式输入增加码流可恢复log delta |
| SF目标 | 全图Gram方向 | 较粗区域的逐token内容特征 |
| teacher构造 | qA/qB各自独立分析后的较高q | 同一y的全图更细量化参考 |
| 单q部署训练覆盖 | 全部原生重建 | 明确保留半batch单q原生重建 |

## 8. 训练与验证顺序

1. 恢复可复现的持久运行环境，定位13.6K checkpoint首个非有限算子。明确FP32修复边界，不把减小GradScaler、增大eps或无限重启当作稳定方案。旧SF checkpoint仅用于诊断。
2. 从同一纯C10K初始化。先验证新模型的原生单q出图与原C一致，新的uniform/mixed packet均能实际往返、side bits无遗漏。确认精度条件仅来自可解码信息。
3. 固定同批图片/q，检查student、teacher逐层激活及精细参考的量化误差；检查正确/错图/空间打乱目标在新损失中有区分，且最新teacher没有丢掉内容。它们是诊断，不是主loss打分器。
4. 用完整损失、权重1、Adam状态和EMA做连续数百到上千次更新的稳定性与显存检查，覆盖六档以及宽窄q差。不能再次以两步预检替代稳定性验证。4×1的实际能否装下仍需实测，FP32修复可能改变性能。
5. 正式实验从10K起，SF用2K ramp；20K为第一段终点，12/14/16/18/20K记录正式单q六档Kodak24真实总bpp、PSNR、LPIPS、DISTS，并分别观察低中高码率。混合流成绩不能冒充单q低bpp结果。
6. 该提案同时改变局部精度训练、DiT条件和SF目标。旧外部纯C能评估整体配方是否更好，但不能独立归因SF。若要主张SF有效，需要同样的新局部精度接口、条件、采样、步数与精度设置但关闭SF的匹配对照；未经授权不启动额外训练。
7. 没有固定bpp上的一致收益就不自动接长训；不以SF很低作为成功信号，也不因单次早期回落立即判失败。

## 来源与证据

- Self-Flow：双时间步局部观察、显式时间条件，同一混合student同时生成和对齐，浅层预测更完整EMA观察。https://arxiv.org/html/2603.06507v1
- iREPA：支持考察空间信息与空间归一化，但不证明本提案可改善压缩。https://arxiv.org/html/2512.10794v1
- Lossy Image Compression with Foundation Diffusion Models：将压缩中的量化误差恢复作为生成去噪任务的相关先例。https://arxiv.org/abs/2404.08580
- C实现：`models/bt_self_flow/codec.py`的步长、全局分析条件及局部synthesis precision；`precision_conditioning.py`；`packet.py`的mixed字段。
- 本次失败诊断：`/data2/ls/experiments/c_native_relational_sf_20261027/diagnostics/diagnosis_20261028.zh.md`。

上述是下一版可执行研究方案，并非已完成实现、已确认原创性或已验证收益。
