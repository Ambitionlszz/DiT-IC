# 把 Self-Flow 改成面向实际压缩收益的训练机制

日期：2026-10-24。本文是代码审计、论文阅读和待验证方案；没有修改训练配置、进程或正在使用的冻结源码。

文中的 C 始终指用户所说的压缩前端。比较时固定同一个重建后端，避免把历史配置里 backend=C/D 的内部命名混进来。

## 结论与优先级

我最看好的方向是：把 teacher 的知识接到实际使用码流的解码路径，同时让 teacher 教前端在给定码率下怎样分配信息。8→20 的浅深层自蒸馏可以保留，但优化目标必须从“整个特征像不像”转向“哪些知识能够通过当前比特预算改善最终重建”。

建议分两步验证：

1. 低成本的结构修改：让目前仅用于辅助对齐的 student projector 输出成为后续 DiT 块实际使用的条件；它同时接受重建和 SF 的梯度。
2. 研究核心：用真实重建的改善及新增码率，为局部信息补充评分；再把有价值的分配决策教给 Router，并逐步改用满足相同总码率预算的 teacher 分配图。

第一步有明确的工程依据，第二步直接针对前端编码效率。两者都需要消融，不能预先承诺能胜过 512 baseline。

## 1. 当前实现告诉了我们什么

审计路径：`/data2/ls/code/DiT-IC-self-flow-sf10k-recovery-20261023`。

- `train_bt_self_flow.py::sample_control`：普通更新与 mixed 更新交替；mixed 中约 25% 编码位置使用固定 q=1，其余位置使用本次随机 q。
- teacher 始终接收 q=1 的重新量化结果。它与 student 共享分析结果，但主潜变量重新量化，不能视为 student 码流的真实渐进增强层。
- `models/bt_self_flow/model.py::forward_analysis`：student 第 8 层经过 `representation_projector` 后参与辅助 loss；实际图像由原始完整 DiT 路径生成，不读取这个 projector 输出。第 8 层本身仍然参与主路径，因此不能据此断言 SF 梯度没有作用，也不能断言 projector 已经发生捷径学习。
- `cross_rate_loss`：teacher 的 clean-latent RMS 更低就允许对齐。它没有检查 teacher 的最终图像是否更好，也没有检查得到这种优势需要增加多少码率。
- `RateRouter` 的当前 BTv1 分支不接收 q。相同图像在不同 q 下采用相同的 allocation codes；q 只在后续改变量化步长。最优空间分配一般可能随码率变化，这里存在表达能力限制，是否是主要瓶颈仍需验证。
- allocation 去均值只约束平均 log-step 的偏移，不约束实际总码率。熵模型、空间上下文和符号分布使“平均精度不变”不能推出“比特数不变”。
- 现有 hard STE、真实 packet 编解码检查已经存在；新方案不应把这些当成尚未实现的创新。

对“对齐学得很容易”需要修正：当前日志约 25K 时 SF loss 为 0.0035，30K 时约为 0.01–0.012。29.5K 校准点，带权 SF/RD 的 coarse 接口梯度范数比约为 8.8%；30K 附近又下降。这不是全参数梯度比例，也没有测梯度方向。现有证据不足以把问题简单归因于“SF loss 太小”。

30K 验证已生成；此前对 24K 与历史 checkpoint 的比较，不是 SF 的因果实验：训练步数、图像 loss 比例、base/repaired 项等有差异。10K→30K 的改善也包含额外训练和 mixed 采样。

## 2. 论文中哪些部分值得迁移

| 来源 | 可以依赖的观察 | 对本项目的启发 |
|---|---|---|
| Self-Flow 原论文 | 两档噪声来自同一采样分布；局部混合构成信息不对称，teacher 使用较干净一档。它同时优化生成任务。 | 不能仅保留浅层→深层 cosine；要保留与主任务一致的视图机制。 |
| KDIC，ICCV 2025 | 分通道能量压缩和码率约束使普通特征蒸馏难以直接迁移；其方法将端到端 RD 纳入模块训练。 | 特征相似不能替代 RD；需要让被指导的表示经过真正的压缩/解码任务检验。 |
| When the Teacher Has More Bits，ECCV 2026 | 作者明确研究同结构高码率 teacher 指导低码率 student，并提出方差归一化 latent 蒸馏。 | “高码率教低码率”本身已有工作；我们的新意需要落实到局部决策、码流使用和实际预算。公开作者摘要已核对，未取得全文，不引用具体公式或表格数字。 |
| OneDC，NeurIPS 2025 | 从收到的 hyperprior 提取语义条件，供实际解码器 cross-attention 使用，再进行辅助语义蒸馏。 | 监督一个实际用于解码的码流条件。它的辅助预测头也可在推理时丢弃，因此问题不是“存在 projector 就一定不好”。 |
| StableCodec，ICCV 2025 | 利用压缩表示直接解出结构分支，并与生成分支协作。 | 结构保真和细节生成有不同作用，不能要求每个高码率特征都由低码率重现。首版不叠加它的全部双分支结构。 |
| DPICT，CVPR 2022 | 对渐进编码单元考虑 RD 优先顺序。 | 用新增比特的边际价值指导补充信息；现有 codec 并非它那样的真正嵌套码流。 |
| ELIC，CVPR 2022 | 变换能量集中和空间/通道上下文都影响码率效率。 | 即使 SF 改善了表示，熵建模及 side information 的成本仍然需要单独核对。 |

Self-Flow 的逐 token 噪声边际保持，不等于我们的量化上下文联合分布保持。用两个同分布 q 构造视图是合理对照，但不是“已经证明正确”的压缩版 flow matching。量化误差也不能直接当成独立高斯 RF 噪声。

## 3. 第一步：让对齐的结果真正被解码器使用

保留 8→20，复用现有 student projector：

```text
图像 → C → 硬量化/熵编码 → 码流 → 熵解码 → g_s → student 第 1–8 层
                                                        │
                                                   c_s=P(h_s^8)
                                                        │
                                    后续 DiT 块的条件适配器 → 重建图像

teacher 的量化视图 → EMA 第 1–20 层 → h_t^20
                                      │
                         stop-gradient 的 SF 监督 → c_s
```

具体建议：

- `c_s` 作为第 8 层以后若干 DiT 块的空间内容条件，使用带门控的 attention/残差适配器。先小范围接入后部块，初始化保持原有输出；不同时重做整个后端。
- `c_s` 完全由收到的量化表示产生。teacher 仅在训练时提供监督，推理中不出现原图、高精度潜变量或 teacher feature。
- 同一个 `c_s` 同时接收图像 RD 与 SF 梯度，使 auxiliary head 与实际重建形成更直接的联系。模型仍可能忽略条件，因此必须做置零/打乱条件的消融。
- teacher 的新条件分支也做 EMA。普通训练仍可截断于第 20 层，不需要新增完整 teacher AE。
- 首版继续使用归一化 cosine，以减少变量。方差归一化值得单独测试，但不能把 SANA 的高方差通道未经验证就解释成“无用纹理”。不直接换成无约束 raw L2：原 Self-Flow 的消融报告过特征范数不稳定。

使用原有主体目标：

\[
\mathcal L_{RD}=4\,\mathrm{MSE}+\mathrm{LPIPS}+\mathrm{DISTS}
  +0.1\,\mathcal L_{base}+\beta(q)R_s,
\qquad \beta(q)=8\cdot32^{-q}.
\]

\[
\mathcal L=\mathcal L_{RD}
 +\alpha\frac{\sum_i w_i\,[1-\cos(c_{s,i},\operatorname{sg}(h^{20}_{t,i}))]}{\sum_i w_i+\epsilon}.
\]

不启用 repaired-latent loss。`w_i` 在下一步由实际任务价值产生；第一步的结构消融先复用现有 gate，避免同时更换所有变量。

新增条件分支不需要额外传输一个独立语义码流，但会增加一些推理参数/计算；训练还可能改变主码流与 hyperprior 的熵，所以不能宣称“完全没有成本”或“码率天然不变”。

## 4. 第二步：teacher 教局部信息的价值，而不仅是特征答案

对同一图像，固定分析结果和评分解码器，对编码位置做少量可实现的精度调整。512 输入时当前编码网格为 8×8；以编码 cell 或小组为动作单位，保持与真实量化、上下文组顺序一致。

令 B 是当前分配，B' 是一个候选调整。定义：

\[
U(B\to B')=
 [D_{411}(x,\hat x_B)-D_{411}(x,\hat x_{B'})]
 +0.1[\mathcal L_{base}(B)-\mathcal L_{base}(B')]
 -\beta(q)[R(B')-R(B)].
\]

其中 \(D_{411}=4\mathrm{MSE}+\mathrm{LPIPS}+\mathrm{DISTS}\)。同时单独记录图像项的改善，防止一个候选仅因 base 更好而看似有益。

这个定义把三个问题分开了：

1. teacher 更接近 clean latent，不代表重建更好。
2. teacher 重建更好，不代表多花的比特值得。
3. 一处量化更细会影响后续上下文，不能只把局部高斯代价当成完整码率变化。

### 4.1 先用离线 oracle 判断有没有收益空间

- 在未用于选择超参的训练验证图像上，从现有 allocation codes 产生候选：某些 cell 提高一档，另一些降低一档。也可交换两个 cell 的 code，保留 code 直方图。
- 保留直方图仍不保证相同码率。先用完整熵模型估计筛选，再对少数最好候选做实际 rANS 编码，计算 main + hyperprior + allocation + scales + header 等全部成本。
- 用当前完整 student 解码器在 `no_grad` 下评分，权重在一次搜索内固定。这不要求常驻第二套完整 AE。评分器和提取表示的 EMA teacher 是两个角色，不能混称为“现有截断 teacher 可以直接评分最终图像”。
- 优先选满足目标总码率容差、且图像重建更好的候选。以全图重建指标计分，避免假定 cell 的影响局限在局部。
- 每图只试少量组/交换，先验证空间；它是有限候选搜索，不是全局最优 oracle。若连这样的分配也无法稳定改善 RD，先不投入在线分配蒸馏。

### 4.2 再教 Router 复现更好的分配

- 给当前 BTv1 Router 增加轻量的 q/log-step 条件；使用现有码流中的全局 quality 字段，不增加新的传输字段。不直接切换完整 precision-v2 codec，因为其现有实现禁止 mixed quality，会同时改变过多组件。
- 搜索得到的优选分配 `a*` 作为停止梯度的监督。可用原 Router 连续分数的排序 loss，指导应提高精度的位置排在应降低精度的位置前面；实际 forward 仍经过原来的 3-bit codes 和 STE。
- 显式 RD 仍约束 student。新增分配监督的梯度在 Router 自身参数上校准；现有 coarse 接口的 SF 范数测量不能直接替代这个校准。
- teacher 的收益来自训练时额外搜索/前瞻，并非“teacher 可以看原图、student encoder 看不到”：两者的编码端本来都能看原图。
- 优选分配图用于 teacher 量化视图，EMA20 提取表示。只在 teacher 局部更可靠、候选整体 RD 确认有益的位置上蒸馏；teacher 被降精度的位置不要求 student 模仿。

这样 student 与 teacher 仍有局部信息不对称，但 teacher 的总体优势尽量来自更有效的信息配置，而不是始终拿一个昂贵的 q=1 输入。高码率 teacher 保留为有用的对照，不把“高码率蒸馏一定有害”当成结论。

### 4.3 把表示知识和编码决策分开检验

联合目标可以扩展为：

\[
\mathcal L=\mathcal L_{RD}+\alpha\mathcal L_{SF,task}+\gamma\mathcal L_{allocation}.
\]

两项辅助监督有不同职责：前者指导解码器如何使用收到的内容条件，后者指导前端把有限比特分给哪些内容。如果只有 allocation 监督有效、加 SF 没有进一步收益，应如实得出“分配方法有效，SF 尚未有效”的结论。

## 5. 训练和计算安排

已有 latent warmup 可复用，1W 的纯 RD checkpoint 可作为受控比较起点。新模块做一致初始化，并给所有对照相同的短 RD 适应期；teacher 从这个共同状态建立。接入时间、优化器、有效 batch、数据顺序和学习率安排保持可比较。

第一轮只验证“被对齐条件进入主路径”是否有价值，必须有相同新结构的 SF-off 对照。不能拿加了条件适配器的模型直接对旧模型，然后把收益全部归给 SF。

待离线分配搜索有结果后，第二轮使用以下 2×2。四组共享相同的 q-aware Router、内容条件分支、重建后端、4/1/1 和 base 项：

| | 无分配监督 | 有分配监督 |
|---|---|---|
| 无 SF | A：共同结构的纯 RD | C：RD + allocation |
| 有 SF | B：RD + SF | D：RD + allocation + SF |

为使这轮成为严格的 2×2，四组使用相同的候选视图生成规则；“有分配监督”只决定是否训练 Router 去预测搜索结果，“有 SF”只决定是否加入同一定义的 feature loss。原有固定 q=1 teacher 的 SF 单列为方法参考，不混作这个 2×2 的严格对照。

双卡可以分两批做 A/B、C/D。若每组单卡，用 batch=4、accumulation=4 保持有效 batch=16；teacher 各卡本地持有。当前双卡训练则是每卡 batch=4、accumulation=2、world=2，同样有效 batch=16。不同 DDP 分组的数值和样本顺序差异要记录，不把“总 batch 一样”当成完全同一随机实验。

在线常规路径仍使用截断 EMA20，离线搜索另占评估预算。第一轮不得每个 microbatch 对所有 cell 逐个完整解码。若后续在线化，采用稀疏采样、候选小批量和缓存/缓慢更新；必须同时报告额外 forward 数与 wall time。

## 6. 在启动新长训练前，哪些证据最有用

1. **当前对齐是否有内容辨识能力**：同图对应位置、同图打乱位置、不同图、只训 projector 的对比；测中心化后的特征相似度和通道方差。原始 cosine 高不能单独证明 collapse。
2. **teacher 是否真的在教更好答案**：给定相同图像和 q，比较最终图像的 D411 优势，不只比较 latent RMS。高码率优势、相同码率分配优势分别记录。
3. **SF 梯度是否支持 RD**：在 codec、Router、后端分别测夹角/内积与范数；当前只有 coarse 范数比例，不知道是否冲突。
4. **新内容条件是否被使用**：推理时置零/打乱该条件，观察实际重建指标；只看 SF loss 下降不能证明有效。
5. **真实比特花在哪里**：main/hyperprior/allocation/scales/header 分解。固定 side bits 的改善和主潜变量的改善分开报告。
6. **有限搜索是否已有等码率收益**：先证实可以通过局部比特交换改善重建，再学习这样的决策。

## 7. 如何判定 work

- 主要看真实总 packet bpp 下的 RD 曲线，尤其 0.03–0.06 bpp；同时报告 PSNR、LPIPS、DISTS，不用固定 q 下的一组数替代相同 bpp 比较。
- 与相同训练预算的 SF-off 比较，至少在 Kodak24 之外增加一个保留验证集。超参/候选选择不使用最终测试集。
- 可将稳定 2–3% 实际 BD-rate 改善作为短试验继续投入的门槛；这是工程筛选标准，不是收益预测，也不是统计显著性的替代。
- 新结构收益看共同结构 SF-off 与旧结构 RD；SF 净收益看 B-A、D-C；分配监督收益看 C-A、D-B。
- 如果低码率 bpp 上涨、只有较高码率感知指标改善，就没有解决当前压缩效率问题。
- 如果 C 有收益而 D 无进一步改善，保留分配方法，停止把它包装成 SF 的成功。

## 8. 对后续实现的具体边界

不直接修改正在训练的冻结树。实验实现放在独立副本，重点修改：

- `models/bt_self_flow/model.py`：实际消费 projector 条件的路径、teacher 的相应 EMA 分支。
- `models/bt_self_flow/router.py`：轻量 q 条件、可导的分配分数用于排序监督。
- `models/bt_self_flow/codec.py`：显式、可审计的 allocation override；所有候选均用相同 hard quantization/context 实现。
- `train_bt_self_flow.py`：独立的视图生成、SF 开关、allocation 开关与各自日志，保证消融可解释。
- 独立 oracle/evaluator：冻结的全图重建评分与真实 packet 码率，不把 local proxy 当成最终答案。

需要核对：新条件能否仅从真实 packet 解码重建；header/side bits 是否全部计入；teacher 分支无梯度且 EMA 范围一致；关闭新机制时可复现共同结构的基线路径；分配 override 在 hard forward 与实际 encode/decode 中完全一致。

## 原始资料

1. Self-Flow, *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis*：https://arxiv.org/html/2603.06507v1 ，官方项目：https://black-forest-labs.github.io/Self-Flow/
2. Chen et al., *Knowledge Distillation for Learned Image Compression*, ICCV 2025：https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_Knowledge_Distillation_for_Learned_Image_Compression_ICCV_2025_paper.pdf
3. El Mennaoui et al., *When the Teacher Has More Bits: Self-Teacher Latent Distillation for Learned Image Compression*, ECCV 2026，作者机构摘要：https://www.eurecom.edu/en/publication/8851
4. El Mennaoui et al., *Variance-normalized latent distillation (VNLD) for domain-specific learned image compression under JPEG AI constraints*, ICPR 2026，作者机构摘要：https://www.eurecom.edu/en/publication/8901
5. Xue et al., *One-Step Diffusion-Based Image Compression with Semantic Distillation*：https://arxiv.org/html/2505.16687v2
6. Zhang et al., *StableCodec: Taming One-Step Diffusion for Extreme Image Compression*：https://arxiv.org/html/2506.21977v1
7. Lee et al., *DPICT: Deep Progressive Image Compression Using Trit-Planes*, CVPR 2022：https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_DPICT_Deep_Progressive_Image_Compression_Using_Trit-Planes_CVPR_2022_paper.pdf
8. He et al., *ELIC: Efficient Learned Image Compression With Unevenly Grouped Space-Channel Contextual Adaptive Coding*, CVPR 2022：https://arxiv.org/abs/2203.10886
9. Shi et al., *DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression*：https://arxiv.org/html/2603.13162v1 ，其中 latent-conditioned guidance 也直接支持“压缩表示应提供实际解码条件”这一方向。
