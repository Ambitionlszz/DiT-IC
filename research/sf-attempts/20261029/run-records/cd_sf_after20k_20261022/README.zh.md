# C 前端＋D 后端：20K 后接入 Self-Flow

2026-10-22，Asia/Shanghai。用户确认的单个双卡实验。

## 初始化与训练预算

- 使用修改前的 BT v1 压缩前端，不启用 precision V2。
- 复用 `/data2/ls/checkpoints/bt_self_flow_v1/common/complete.pt` 的 50K raw codec 和尺度校准。
- 实际入口为原 C/D 共同的 `/data2/ls/checkpoints/bt_self_flow_v1/backend_init.pt`。已逐张量确认其 codec/scales 与上述 warmup raw 完全一致，并核对 source SHA256 及 SANA/AE/文本资产身份。
- 后端从共同的预训练 SANA/AE 与初始 LoRA、文本投影开始，D 的尺度条件模块按原规则初始化。没有加载已训练约 90K 的联合模型。
- 新的联合训练总计 200K optimizer updates，前 20K 包含在其中；此前的 50K latent warmup 直接复用。
- 学习率、优化器动量、数据顺序和计数在 20K 切换时连续，不新开后训练日程。

## 双卡与优化

两张 V100S 32GB 做 DDP，每卡 batch=2、累积2次，全局有效 batch=8。

每卡都有完整 student。前 20K 没有教师；达到 20K 后，每卡各自从已同步的当前 raw student 复制一份教师，服务本卡数据。教师只保留第20层目标所需的后端，不含 AE 图像解码器和 student projector。

更新模块为 codec/Router/熵模型、DiT LoRA、D 条件模块、文本投影、AE decoder LoRA；SF 接入后还更新 student projector。projector 在前20K不参与前向、无梯度且无Adam状态，SF接入后单独对累计梯度做跨卡平均；其余student主参数使用DDP。这样保留coarse→feature的直接求导路径，不使用DDP的find_unused输出包装。AE encoder 与预训练基础参数冻结。教师全部停止梯度。

AdamW，betas=(0.9,0.999)，weight_decay=0，max_grad_norm=4。学习率峰值：codec=5e-5，后端=1e-4，AE decoder LoRA=1e-5；主学习率500步 warmup，随后 cosine 至峰值的10%。熵模型 quantile 独立 aux optimizer=1e-3，沿用原训练器。

开启梯度检查点。V100 使用 FP16 AMP/GradScaler，codec/熵模型和感知损失保留既有 FP32 语义。

## 主损失

`L = 4*MSE + LPIPS + DISTS + 0.1*base + beta(q)*R + alpha(t)*SF`

- 图像 MSE/LPIPS/DISTS 权重为 **4/1/1**。
- base 为前端 coarse latent 相对 clean latent 的归一化 MSE，权重0.1。
- `repaired_weight=0`：修复后 latent NMSE 仅作日志指标，不参与优化。
- `beta(q)=8*32**(-q)`，q 在[0,1]抽样，保留原 C/D 的采样流。
- R 包含原 likelihood 估计及完整侧信息；混合视图也计入 mask 侧信息。

## 20K 前后与局部不对称

更新1…20000：正常整图质量输入，只做主 RD 训练，不创建教师，不产生二值质量混合输入。

完成20000步后复制当前 raw student 的后端作为教师。更新20001开始启用原有8→20对齐：学生完整执行后端恢复，并将第8层特征通过残差MLP projector 与教师第20层特征做 cosine 对齐。层号为配置的一起点编号。

后续正常/混合更新交替：20001为混合，20002为正常，依此类推。混合更新在每图编码网格上随机选25%位置使用q=1，其余75%用当前q；在codec量化时构造局部质量差异，再经完整解码得到学生coarse latent。不同rank使用独立mask随机流。

这同一个学生视图同时承担图像重建、base、码率和SF任务。正常更新也计算SF，有助于保留部署时的正常输入训练。

教师看同一张图的全图q=1视图。旧前端的分析结果可以共享，但会完整重执行教师对应的量化与恢复路径。教师输入由当前codec生成并停止梯度；教师后端是EMA，非另外一套EMA前端。师生使用各自实际误差生成的尺度条件，以及原生固定SANA时间。

只在教师coarse误差小于学生的位置进行SF监督；这是输入可靠性门控，不保证最终图像或感知指标必然改善。SF梯度通过学生后端进入前端，教师分支不反传。

教师EMA decay=0.9999，每次成功的学生optimizer update后更新LoRA/文本投影/条件模块。

SF权重从更新20001起用2000步ramp逐渐引入。每500步在coarse latent接口校准辅助梯度，以RD梯度范数的5%为目标、10%为上限，并保留标量系数上限；比例不是后端参数或Adam实际更新的比例。校准使用两卡汇总的范数，无有效SF梯度时不放大系数。

校准使用训练AMP scale/累积次数缩放，在autocast环境外执行autograd.grad，再以FP32除回scale并计算范数；孤立校准若溢出，两卡共同降低探测scale，有限次重试后仍异常则停止。真实GPU诊断确认：在forward的autocast环境内调用backward会使RD/SF接口梯度均为NaN；移出后同一图的梯度恢复有限值。另加小梯度下溢和autocast精度回归测试。日志保留校准scale、重试次数及RD/SF梯度范数。此前失败的工程短跑各自归档在preflight_*目录；正式训练使用当前修正实现。

## 验证与可恢复性

原四个训练集、512裁剪。每2K对Kodak24原图做六档验证，使用deployment EMA和完整真实packet bpp，每10K进行全验证集真实解码检查，其余节点每档抽2张。每2K保存，保留最近3份及最佳权重。

训练checkpoint保存raw/deployment EMA、主/aux optimizer、scheduler、AMP scaler、每rank随机状态、数据进度。SF激活后另保存教师、教师创建步数和辅助权重，恢复时不重新初始化已有教师。

`scripts/launch_delayed_self_flow.py` 使用独占supervisor锁，固定配置与代码SHA256清单；重启会恢复最新断点。停止监督进程时只通知实际trainer workers在更新边界保存，保持torchrun/DataLoader活到保存完成。

实际入口：

```bash
/tmp/bt-self-flow-runtime-20261016/linux/venvs/ditic/bin/python scripts/launch_delayed_self_flow.py --gpus 0,1
```

工作目录为 `/data2/ls/code/DiT-IC-self-flow-start-20261022`。配置、初始化身份、日志和运行状态分别在 `configs/joint.yaml`、`initialization.json`、`logs/joint.log`、`pipeline.json`。代码来自独立源码副本，原始来源与文件校验记录在源码目录 `source_snapshot.json`，启动时另记录当前文件哈希。

真实模型工程短跑单独保存在 `preflight/`，将20K边界缩为2步，验证RD前缀暂停、混合SF接入、教师及优化状态恢复、实际码流和显存；不计入正式200K训练。

旧 precision V2 双卡任务已正常暂停在4333步，断点为 `/data2/ls/checkpoints/c_precision_v2_D_v100_20261022/common/paused.pt`。

工程验证结果：17项单元/回归检查通过；真实双V100短跑6步（2步RD＋4步SF，中途暂停并恢复），无AMP跳步，SF梯度和系数均为正；六档实际码流验证通过，显存峰值19,887,452,672 bytes（约18.52 GiB）。这些结果只证明训练实现可运行，不代表200K实验的质量收益。
