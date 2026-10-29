# C 上的双质量与空间关系 Self-Flow

状态：已被 `dit_only_native_bridge.zh.md` 的候选方案取代，不作为实施依据。用户进一步厘清：Self-Flow 的作用网络是 DiT，teacher 只应是 EMA DiT；本文件中的 EMA codec 与真实局部混合码流是额外改动，未获确认。以下保留为历史讨论，没有实现、预检或替换训练。

## 原文的对应

原文独立抽取两个时间步 t,s；学生根据随机 mask 混合两档噪声，teacher 全图用 min(t,s)。本设计独立抽取 C 原有六档中的 qa,qb，学生根据同一空间规则使用两档名义压缩精度，teacher 全图用 max(qa,qb)。q 越大名义量化步长越小。相等抽样保留，此时自然成为同视图的深浅层关系监督。

两档来自原图 latent z 的 C 编码，不再先解码一份 c 再重压缩。更高名义质量不保证每个位置更正确，也不意味着两份有条件分析产生的表示严格嵌套。

## 真实混合精度，而非拼接两个解码 latent

定义 coding-grid 上的 q_map=(1-M)*qa+M*qb，M 先沿用 25% 随机 cells。两档不按高低排序后固定分配；packet 的规范化存储可以排序 low/high，但必须对应翻转 mask，不能改变原抽样。

学生 codec 在分析、router、量化和合成中使用一致的空间精度条件，产生一份合法的混合包及其 cS。这里需要扩展 precision-v2 的空间条件支持；不能直接删除 mixed 拒绝开关，也不能把独立编码 qa/qb 后拼出的 mean 冒称为合法混合包。

该包的 main、hyper、allocation、mask 和 header 全部计入 rate。训练用 likelihood 估计；验证用实际 packet bytes。

## 每卡四张的建议分配

随机两张使用全图 qa，另两张使用局部 qa/qb。四张都由共享 student 正常运行第 1–28 层、CStep 和 VAE，并全部读取 H8。可以按 codec 控制类型分组编码，再拼为一个四张的完整 SANA/VAE batch。每张只有一次学生重建，完整保留单图空间布局。

原生两张用于持续训练部署时的全图单质量路径；混合两张用于学习从较完整区域推断较弱区域。所有四张做关系对齐。常规评测继续使用全图单质量。

teacher 使用 EMA codec 在全图 qT=max(qa,qb) 下得到 cT，随后由 EMA SANA 前 20 层生成 H20。全 teacher 分支 stop-gradient，无 VAE、无图像重建。teacher 的高质量表示只作为目标，不直接喂入学生。

## 空间关系目标

对每张图分别计算，不跨图片拼关系。H 的形状为 N×D，512 输入的当前 SANA 对应 N=256。

    Hc = H - mean_tokens(H)
    K(H) = Hc @ Hc.T
    r(H) = vectorize_off_diagonal(K(H))
    L_rel = mean_images[1 - cosine(r(H8_student), stopgrad(r(H20_teacher)))]

关系和归一化用 FP32，并用 eps 处理数值零。去掉对角线，避免只匹配 token 自身能量。归一化关系向量让整体特征缩放不决定 loss 尺度，不设置可训练 P、不跳层、不增加关系预测网络。关系目标与原始特征向量 cosine 二选一，不叠加。

它约束位置之间的相对结构，允许跨深度采用不同的通道坐标。它不是语义的充分统计，也不能保证忽略不可恢复纹理或位置模板。teacher 的空间内容和有效能量必须先诊断；先前公共分量主导 cosine 的解释仍是未测实的假设。

## 一份重建损失，不额外添加 D_mix

旧草案的 D_mix=4*MSE+LPIPS+DISTS，是额外一次混合重建的图像损失。旧设计先对四张算 L_C，再对另外的半 batch 加一份按均值计的 D_mix，改变了图像重建相对 rate/base 的权重。

这里四张各自只有一次重建，统一平均：

    L = mean_images[D411(x_hat,x) + 0.1*base(cS,z) + beta(qa)*R_total]
        + lambda(step)*L_rel

    D411 = 4*MSE + LPIPS + DISTS
    base = 原 C 按 s_data 归一化的 coarse latent MSE
    lambda 最终为 1；repaired-latent loss 为 0

qa 是本次主码率档，beta(qa) 沿用原 C 的六档系数。qb 是局部精度探索档；整份学生包的所有比特仍承受同一个 beta(qa)。这是一个明确的全图 RD 选择，不引入逐位置 beta 或额外共享比特分摊规则。qa 不能从规范化 packet 的 low_code 推断：当 qa>qb 时它们不同。当前 rate_beta(control) 默认读 low_code，因此实现必须显式传入主档 qa。

这里不对不同档位的 R 做均值或插值来冒充混合包码率。不同图片是否有 mask 的 side bits 也必须分别计费。

## 前端联动

学生的 H8、cS 和 codec 输入链不 detach。原生与混合学生的关系梯度都可经过 g_s、量化 STE、router 与 g_a；重建也沿同一路径，rate 约束同一份被使用的编码表示。AE encoder 仍冻结。teacher 的输入和参数均无梯度，EMA 仅跟随学生更新。

低码率收益假设：有限比特优先支持能帮助其他位置完成生成的结构关系。高码率收益假设：原生样本继续训练细节重建，混合样本加强局部证据不足时的上下文利用。两者均不是收益保证，SF 也可能推动前端增加 rate。

## 实施前的检查与短试

从指定纯 C 10K 完整 checkpoint 起步，512、双卡每卡 4、积累 1、有效 batch 8，qa 保持原 C 的 iid 六档。候选接入安排为 10K→12K 将关系权重从 0 增至 1；不是重新进行 10K 原生预热。

必须先验证 uniform-q 路径能恢复源 C，mixed packet 实际压缩/解压与训练重建语义一致，mask/side bits 无遗漏，主档 beta 未被 canonical low/high 替换，原生与混合关系梯度确实进入 codec，以及 teacher 全 stop-gradient。真正带 teacher 的 512 forward/backward 通过后才能确认双卡显存和吞吐。

先检查源 C10K 的 H8/H20 关系差距、中心化能量，并与错图/打乱空间的 teacher 对比。loss 下降不能代替内容检查。再考虑短训到 14K/16K，与外部纯 C 的同 step、同真实总 bpp 比较低中高码率。

整套训练超过已有 C，只能支持系统收益。要把收益单独归因于关系 SF，仍需相同混合精度训练而关闭关系项的对照；本设计不授权另外启动纯 C 或任何训练。

## 来源与创新边界

- Self-Flow 双时间步、局部信息不对称和深浅层自监督：https://arxiv.org/html/2603.06507v1
- DINOv3 的图像内 Gram 关系约束是已有方法；本文的中心化关系 cosine 不是其原始 Gram MSE 配方：https://arxiv.org/html/2508.10104v1

不把 EMA、Gram 或跨码率蒸馏各自宣称为新发明。待验证的研究主张是：让原版局部信息不对称作用于真正付费的编码精度，并以同一路径上的重建、关系与 rate 训练可用于部署的压缩表示。
