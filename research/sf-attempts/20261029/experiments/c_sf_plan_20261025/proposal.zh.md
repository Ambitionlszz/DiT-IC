# C + Self-Flow：让精度差分预测进入真实重建

2026-10-25。设计提案，未修改训练代码，未启动或停止 GPU 任务。

主方案：保留 student 内部的局部精度不对称、student8 → EMA teacher20 的深度与输入不对称；把“对齐整份 teacher 特征”改成“预测更细量化带来的特征修正”，并将预测均值直接用于实际重建。辅助蒸馏只训练解码侧，前端 C 通过真正的重建和码率目标决定该保留哪些信息。

这是一项待验证的研究假设。下文给出完整的视图、目标、梯度与双卡执行定义；当前并无已经通过 GPU 验收的实现。CPU 原型只验证了量化公式与梯度隔离，不能证明实际 RD 收益。

## 1. 旧实验究竟教会了我们什么

证据来自 `/data2/ls/checkpoints/cd_sf_after10k_b4a2_20261023` 的已有验证与日志；本次只进行 CPU 读取与同码率插值，没有新增模型推理。精确数值见 `historical_evidence.json`。

- 已有 32K 评估中，0.04 bpp 下 PSNR / LPIPS / DISTS 为 23.278 / 0.15177 / 0.07972，历史 512 baseline 200K 为 23.863 / 0.12983 / 0.07422。0.08 bpp 时 PSNR 24.964、DISTS 0.05484 已优于这个 baseline 的 24.903 / 0.05890，但 LPIPS 0.10264 仍落后于 0.09897。这提示低码率与各指标的收益不均衡，不能以高码率一个指标改善宣称解决问题；32K/200K 的不同预算也不构成 SF 的因果证据。
- 接入前 10K 到后续 checkpoint 的真实 RD 在改善；这包含更多训练和 mixed 采样，没有同预算 SF-off，不能据此证明 SF 有效。历史 C/C+D 还使用不同的 loss、有效 batch、前端及训练预算。
- “SF 很快学会，所以 loss 太小”不是已证实的诊断。原始 SF loss 约从 25K 的 0.0035 到 30K 的 0.0105，32K 日志约为 0.013–0.014；不是一直下降。29.5K 带权 SF/coarse-RD 梯度范数比约 8.8%，30K 约 3.2%，并非完全没有梯度。这些是 coarse 接口的范数，没有测全参数梯度或方向。
- 原 projector 仅服务辅助对齐。原第 8 层本身仍在重建主路径，不能断言 SF 对主干无作用，也没有证明 head-only 捷径已经发生。新方案的任务是缩短这条联系，并用实际干预验证。
- teacher 始终 q=1，且以 clean-latent RMS 更低作为允许对齐的依据。它不能证明 teacher 最终图像更好，更不能证明低码率值得花比特模仿其全部细节。新方案不再依赖这个 gate。
- 旧 teacher EMA=0.9999，半衰期约 6931 次 optimizer update。它可能加重初期目标滞后，但尚无对照证明这是失败原因。新方案选择 0.999，半衰期约 693 次 update，并从共同 10K student 当场初始化。
- C/C+D 在没有 SF 时已经存在低码率弱于 baseline 的现象。SF 不能自动解决主潜变量冗余、hyperprior 或固定 side bits 的成本。评估必须拆分真实包成本，不能把去掉 D 的 scale map 所节省的比特算成 SF 收益。

因此，不采用“把 SF 权重开大，然后等先变差再恢复”的策略。概率头也不能单独解决问题：不确定性反映可预测难度，不等于任务价值；必须让预测经过真实解码检验。

## 2. 复用哪一个 C

使用新下载的 precision-v2、lambda16、50K warmup：

`/data2/ls/checkpoints/ditic-50k-warmups/c_precision_v2_lambda16_50k_20261022/backend_init.pt`

这是 joint 初始化文件，不是用 `common/complete.pt` 恢复 warmup 的 optimizer。C 指用户要复用的压缩前端；后端固定为相同的 SANA 一步重建，不混入 D 后端的 distortion-scale 改动。

以下载包的 `configs/C_joint.json` 为训练定义：precision conditioning 已作用于 analysis / Router / synthesis，不能把增加 q conditioning 当作新贡献。图像硬量化，rate likelihood 使用 noise。六档 q 为 `[0,.2,.4,.6,.8,1]`，对应 rate coefficient 为 `[16,8,4,2,1,.5]`。

本地最接近的源码为 `/data2/ls/code/DiT-IC-c-precision-20261022`。已确认 365 个 codec 参数名称和形状吻合；下载包声明的确切 Git commit 尚未取得，不能宣称源码逐字一致。现有 precision-v2 loader 禁止 SF，C 后端没有 8 层捕获接口，需显式实现本方案，不能直接打开旧 SF 开关。

## 3. 两层不对称如何构造

### 3.1 Student 内部：扰动真实精度分配，而非拼接重建 latent

对一次训练图像，采样全局 q，按原 C 完整执行一次 quality-conditioned analysis，得到 y、hard hyperprior、Router 的 3-bit 整数分配图 k。512 输入时编码图为 8×8。

普通视图使用原 k。SF 视图均匀抽取 25% 编码 cell，将这些 cell 的 allocation code 置为已有的最低档 0，其余 code 保持原值，得到 k_S。它没有把图像或 latent token 清零；扰动的是原系统本来就支持的局部量化精度。

必须重新由 k_S 计算去均值 allocation、完整空间/通道上下文量化和 g_s，得到 student coarse。不能在 g_s 输出后拼接两份 coarse，因为那样不对应同一真实符号/context 解码过程。

令原本与扰动后的量化步长为 Δ_N、Δ_S：

\[
a(k)_i=\frac{2}{7}(k_i-\bar k),\qquad
\Delta_i=\Delta_g(q)\exp[-(\ln 2)a(k)_i].
\]

**去均值是一个关键细节。** 若 d_i=k_i M_i，则

\[
\log\Delta_{S,i}-\log\Delta_{N,i}
=\frac{2\ln2}{7}(d_i-\bar d).
\]

未选中的位置通常变细；选中位置也并非全部必然变粗，必须看实际 Δ。CPU 例子已确认这一点。它保持的是平均 log-step，不是实际码率。

k_S 仍是 0..7 整数图，直接占用现有 3-bit allocation 字段。Student 的混合精度视图可以实际熵编码和解码，不新增 mask 字段；主流和 hyperprior 的比特数仍可能变化，需要完整计费。全局 q 始终只有一个，保留 precision-v2 的 analysis 定义，不开启旧 global mixed-quality 分支。

当某图没有产生实际粗/细差异时，该图 SF 项为零，不凭空制造目标。

### 3.2 Student 与 teacher：teacher 保留所有更细精度

定义训练专用 teacher 步长：

\[
\Delta_{T,i}=\min(\Delta_{N,i},\Delta_{S,i}).
\]

用同一次、同一个全局 q 的 analysis，在这个步长下重新完整执行 hard quantization/context 和 precision-conditioned synthesis。teacher 在 student 已经更细的位置保持相同精度，只在实际变粗的位置补回精度。

这是训练时的特权量化视图；不声称 Δ_T 能直接表示成当前格式的另一份普通 packet，也不声称 context quantization 构成严格嵌套码流。部署时只存在 student 的合法 packet。名义上更细的量化同样不保证最终图像更好，后文安排直接的重建检查。

两次 teacher 特征提取使用**同一个**冻结梯度的 EMA20：

- 输入 student 的实际 coarse，提取 h_T,L^20。
- 输入上述更细的 coarse，提取 h_T,H^20。

两个 forward 串行执行，不复制两份 teacher。teacher 只复制 DiT 前 20 层及对应 text projection，不复制 AE 或整套 codec；特权量化使用当前 C 的参数并全程 no_grad。teacher 特征都在新修正接入点之前捕获。

这样，每个全局 q 都有同样定义的局部任务。q=1 仍能学习被局部降低精度区域的修复，不会因为已经处于最高全局 q 就失去输入不对称。teacher 也不必始终使用固定 q=1 的远距离目标。

## 4. 模型只增加一个真正用于重建的预测头

令 N 为无可学习参数、逐 token 的 RMS normalization。teacher 给出的目标是

\[
r_T=\operatorname{sg}\big[N(h_{T,H}^{20})-N(h_{T,L}^{20})\big].
\]

这表达“同一个解码器看到更细量化后，表示怎样改变”，共同内容先相消。它不等于有严格统计保证的纯语义/纯纹理分解，也不预先指定边缘、文字或高频 mask。

Student 第 8 层接一个共享小 MLP，输入为 h_S^8 和由真实 packet 可恢复的 log Δ_S 条件，输出逐 token/channel 的均值 μ 与方差 v：

\[
(\mu,v)=P(h_S^8,\log\Delta_S),\qquad
v=0.05^2+\operatorname{softplus}(s).
\]

编码 8×8 的 log-step 图按几何对应关系提升到 DiT 16×16 token 图。它是已知的量化条件，不被解释成 DiT token 的精确噪声方差。

Student 原本第 9–20 层照常运行；第 20 层后，直接接入同一个 μ：

\[
h_{S,\mathrm{used}}^{20}
=h_S^{20}+\operatorname{sg}(\operatorname{RMS}(h_S^{20}))\,\mu.
\]

随后照常经过第 21–28 层、原一步 scheduler 和 AE decoder，产生最终图像。

```text
实际码流 → C 的 g_s → S1..8 → S9..20 → 加入均值修正 → S21..28 → AE → 重建
                       └── P(h8, logΔ) ──────↑

同一个 EMA20：student coarse / 更细量化 coarse → 两份特征之差 → 监督 P
```

均值输出层零初始化，因此启用新模块时重建保持原 C 输出。方差初始化为 std=0.5，仅为数值起点，不作图像价值判定。v 不用于给主路径加置信门，也不把大方差直接解释成应该放弃的细节，避免所谓置信度本身成为绕过任务的开关。

MLP 可以沿用 1152→2304→1152 的均值结构，增加同宽方差输出和轻量 log-step embedding。约 8M 个训练参数；推理移除方差输出，保留约 5.3M 的均值支路。实际参数数目实现后记录。推理仍为原 28 个 DiT 块、一次生成更新，不需要 teacher、原图、搜索或额外语义码流。

该支路仍可能被后部网络忽略；必须在固定同一码流下进行 μ=0 / 打乱 μ 的重建干预，不能仅以“接了主路径”推断它已经有用。

## 5. Loss 与梯度：让实际 RD 决定前端该改什么

主目标沿用用户要求：

\[
L_{RD}=4\,MSE+LPIPS+DISTS+0.1L_{base}+\beta(q)R_{S,\mathrm{total}}.
\]

repaired-latent loss 权重为 0。R 包括主流、hyperprior、allocation 与 header 等该 C 包格式的全部成本。训练以可导 likelihood 加固定 side bits 估计，验证用实际 packet 字节数。

SF 使用归一化差分的 Gaussian NLL，所有项按 token/channel 平均并用 FP32 计算：

\[
L_{SF}=\frac12\operatorname{mean}
\left[\frac{(r_T-\mu)^2}{v}+\log v\right].
\]

log v 会惩罚无限增大方差，方差下界避免零误差下 NLL 无界和极小方差的数值问题；这仍不保证学出的不确定性校准。它的作用是区别可预测修正与不可预测差异。**有用与否由实际图像 RD 检验，不由方差宣判。**

梯度分工明确：

| 路径 | C：analysis / Router / synthesis | Student DiT、均值支路、text | 方差支路 | AE decoder LoRA |
|---|---|---|---|---|
| 实际图像 + base + rate | 按各项的真实依赖端到端优化 | 图像项优化 | 无 | 图像项优化 |
| SF NLL | 无 | 前 8 层及共享预测头优化 | 优化 | 无 |
| Teacher | 全部 no_grad；输入量化只作目标 | EMA20 无反传 | 不需要 teacher 方差头 | 不需要 teacher AE |

实现上，不使用容易在 DDP/AMP 下弄错的梯度覆盖：SF microbatch 另跑一次共享权重的 student 前 8 层，输入为 `coarse.detach()`；SF 头的 `log_delta` 条件也必须 detach。这一支不跑 9–28 层或 AE。主重建的前 8 层和 μ 仍保留完整端到端梯度。SANA dropout=0，两份前缀的前向表示值应一致。

RMS 只承担归一化与残差尺度匹配，不再被用来判断 teacher 最终图像是否更好。概率头的方差也不能替代该判断。

这样，SF 不能直接驱动 C 增加比特去追逐特权答案；C 能收到 μ 改善最终图像所产生的真实 RD 梯度。它不保证辅助梯度永不与任务冲突，也不保证全局最优 RD，但比把 NLL 一并无约束压向 encoder 更符合本实验要解决的问题。

### SF 权重的唯一校准

接入时用固定、与最终测试隔离的训练样本，六档均衡，在均值输出 μ 这个共享接口测量图像 D411 和 NLL 梯度范数。取各档比值的中位数，定义

\[
\alpha_0=\min\left(10,
0.05\operatorname{median}_q
\frac{\|\partial D_{411}/\partial\mu\|}
{\|\partial L_{SF}/\partial\mu\|+10^{-12}}\right).
\]

必须至少包含每档 4 张非退化样本；有效目标不足时不得靠零分母得出权重。初始 μ 为零时，h8 的 SF 梯度可能也是零，因此在 μ 输出校准，不在 h8 上除零。校准只为选取一个初始辅助强度，之后固定 α0，并用 2K 线性 ramp。不要反复补大 α 抵消模型学到的不确定性。

校准使用独立的数据索引和随机数状态，不推进主训练的数据/噪声序列。teacher 特权量化不计算无用的 noisy rate likelihood，不消耗 student 的 rate-noise RNG；保留硬量化和相同的 context/LRP 重建过程。

这是初始梯度尺度，不是后续全参数 5% 的保证；校准输出、实际 α、梯度统计均写入 checkpoint。监测方差、μ 的使用及实际 RD，不能把 α 校准当作收益保证。

## 6. 训练与双卡分配

共同前缀 0–10K 从新 warmup 的 `backend_init.pt` 初始化，只训练原 C 的 RD；在第 10K 步保存完整状态及验证，再为实验分支添加相同、零初始化的均值支路。新参数单独加入 optimizer，保留已有参数的 optimizer 状态。teacher 从这个 10K 的当前 raw student 创建，而非从 50K warmup 或长期平均状态创建。

| 项目 | 首轮设定 |
|---|---|
| GPU | 两张 V100S 32 GB，2-rank DDP；每卡一份完整 student + 本地 EMA20 |
| 输入 | 512×512 训练，Kodak 验证原尺寸 |
| Batch | 每卡 8，accumulation=1，全局有效 batch=16 |
| 显存回退 | 每卡 4，accumulation=2，仍保持全局 16 |
| 主/辅 microbatch | 每次 update 一半样本用原生分配，一半用局部扰动；两类均完整计算重建 RD，SF 只在局部扰动类计算 |
| q | 六档均衡打乱循环，同一 update 的所有 rank/microbatch 使用同 q 与对应 β；覆盖所有档位 |
| SF 接入 | 10K 保存完成后接入，10–12K 线性 ramp |
| 扰动 ramp | 同一 2K 进度把“辅助一半样本采用扰动”的概率从 0 升到 1，25% cell 比例固定 |
| Teacher | EMA=0.999，每个成功 optimizer update 后更新一次；两份目标串行 forward |
| 学习率 | 依新 C 配置：codec / DiT LoRA / AE decoder LoRA / 新头 1e-4，entropy aux 1e-3 |
| 数值 | FP16 autocast + GradScaler；codec、likelihood、NLL 与关键归一化 FP32；grad clip=4 |
| 总预算 | 先到总步数 30K 检验，若通过再延长至 200K；沿用 100K / 160K / 190K milestone |

accumulation=4 时，原生/局部 microbatch 各两个。每个 microbatch loss 仍按 accumulation 正确缩放，最后一个 microbatch 才进行 DDP 同步。两类数据、q、扰动 mask 的随机状态均可恢复。

teacher 没有参数梯度、optimizer state、AE 激活。两个目标 forward 共用同一份 teacher 参数，特征缓存很小。SF 的额外有梯度路径只有 8 层前缀与小头。相对旧“每个 microbatch 一次 teacher20”的实现，新方案一半 microbatch 做两次 teacher20，平均 teacher 层数相当，但增加了一半 microbatch 的可反传 8 层前缀。

旧实现日志的 max_memory_bytes=27,846,560,768，约 27.85 GB / 25.93 GiB；这只是历史参考。新 precision-v2 codec、两份目标、重复前缀和新 optimizer state 的实际显存仍需实测，不能据此保证 batch=4 不 OOM。首个完整 optimizer step 后才分配齐的 Adam 状态也必须计入。

## 7. 先确认 teacher 的修正值得学

启动长实验前，在不用于超参选择的测试集之外，固定训练留出图像和共同 10K 模型，做一次直接的反事实检查：

1. Student 同一真实 packet、μ=0，记录 D411 和各指标。
2. 同一 packet，把训练特权目标 r_T 临时作为 μ 接入 20 层，记录最终图像。

这是对“这个修正方向能否改善实际解码”的低成本检验，不进行逐 cell 搜索、不产生在线 oracle 分配标签，也不把该结果视为理论上界。至少分别检查低 q 和高 q。如果这个 privileged correction 都系统性损害最终图像，本方案的 target/接入点就没有通过前提验证，应暂停该 SF 长跑，不能仅靠把 NLL 训低来掩盖。

GPU smoke test 需包含 q=0 / q=1 的 native 与局部视图、两次 teacher、完整 4/1/1 + base、NLL、反传、optimizer.step、EMA 更新，以及保存/恢复。用满 α 跑至少 32 次 update 检查稳定性和峰值显存。b4 不合适时按表回退 b2/a4，并重新测量，不降低分辨率冒充通过。

## 8. 两张卡如何给出可归因的结论

优先顺序执行两段 DDP 短试，每段都用两张卡，不将 teacher 独占一张卡：

- A：共同 10K checkpoint → 30K，相同均值支路、native/局部视图与训练序列，SF-off。
- B：同一 10K checkpoint → 30K，同结构、同样本预算和视图，开启本方案 SF。

比较 B-A 才是新增 SF 的作用；不能把 head 增容、局部扰动或换新 C warmup 的收益算给 SF。teacher/辅助计算造成的 wall time 开销单独报告。历史 512 baseline、C、C+D、旧 cosine SF 继续作为性能参照，不作为因果对照。

评估每 2K 步进行，12K / 20K / 30K 保留明确里程碑。使用真实 packet 总 bpp 的共同覆盖区间，分别报告低、中、高码率的 PSNR / LPIPS / DISTS。沿用全尺寸 Kodak24 方便历史参照，另加固定的 DIV2K/CLIC 留出集；不继续用 Kodak 选择超参后把它称为独立测试集。

必须看四件事：

1. 同总 bpp 的曲线是否比 A 稳定改善，而非同 q 码率上涨换画质。
2. 固定 packet 下把 μ 置零或打乱，图像是否变差；是否至少在低/高两个区间实际使用了修正。
3. 方差的数值、目标误差与 μ 幅度是否合理；只训 head 是否几乎取得全部 NLL 改善。这是诊断，不仅看 loss 标量。
4. 主流 / hyper / allocation / header 成本分别怎样变化。SF 不会凭空消除固定 side bits，也不应为 teacher 的细节使主流更昂贵却不改善实际 RD。

“先变差再恢复”允许发生，但不是继续训练的自动理由。30K 时若 B 仅优于自己的 10K、却没有优于相同步数的 A，不能证明 SF 有效。若只有一个指标、一个码率段受益，应明确报告取舍，不宣称同时解决高低码率。若均值支路有效而 SF 无额外贡献，应保留结构结果，否定本轮 SF 假设。

首轮得到正向证据后，最有价值的一个后续消融是把 NLL 换成相同归一化差分的 MSE，判断不确定性是否必要；不在主实验开跑前堆叠更多 Router 搜索、语义分类器或对抗损失。

## 9. 实现落点与状态

- `codec.py`：同 q 的整数 allocation override；独立的 teacher-only delta override；完整 group/context 重算；student override 的 hard-forward / 真包解码一致。
- `model.py` / SANA forward helper：原 C 路径的 8/20 层捕获、20 层修正注入、相同 C 语义的 EMA20；不引入 D conditioner。
- 新概率头：共享均值直接消费；方差仅训练；log-step 来自真实 packet；FP32 NLL、RMS 与 floor。
- `experiment.py`：新增独立 SF 模式的显式合法性检查，继续保留 q-dependent analysis 的约束；不得盲删旧 mixed-quality 检查。
- trainer：两个 loss 的梯度边界、一次 α 校准、10K handoff、同一 ramp、q 均衡采样、DDP/AMP 与完整恢复状态。参数可复用的前缀需使用兼容共享调用的 non-reentrant checkpoint；每次只做一个合并 backward，DDP 的条件分支参数也要明确处理。
- evaluator：actual packet 全部计费；mu intervention；teacher correction 反事实检查；原生和覆盖 allocation 两种包的编解码一致性。

当前已完成的是公式/数据审计和 CPU toy 验证。真实模型加载、这些接口修改、完整 DDP、GPU smoke test、训练与实际收益均未完成。本文件不表示已启动实验，也不表示已经证明收敛或有效。

## 参考依据与本方案的边界

- Self-Flow：保留局部输入信息差、浅层 student 对深层 EMA teacher、同时做主重建任务。原论文研究 flow generation，不直接证明本项目的变码率 RD 收益。https://arxiv.org/html/2603.06507v1
- VID：概率预测而非强迫每个 teacher 特征都精确可恢复。本文使用的条件方差差分预测、实际残差注入与梯度分工是本项目设计，不冒充 VID 原方法。https://arxiv.org/abs/1904.05835
- OneDC：已解码的压缩表示应在生成解码中真正发挥条件作用。其 hyperprior/tokenizer 结构与本文不同。https://arxiv.org/html/2505.16687v2
- KDIC：LIC 的知识迁移需要考虑能量集中、码率约束与端到端任务，普通特征匹配不充分。本文不复现其模块替换蒸馏。https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Knowledge_Distillation_for_Learned_Image_Compression_ICCV_2025_paper.pdf

这些研究支持设计取舍，不构成对本方案优于 baseline 的保证，也不支持对尚未实施的新组合宣称首创。
