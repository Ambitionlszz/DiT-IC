# C precision v2 implementation review — 2026-10-22

结论：本次自查和工程测试覆盖了完整实现，未留下已知阻断训练的代码问题。尚未运行 5090 完整模型的双卡 smoke，也没有新 50K / 200K 收敛结果，不能将工程通过写成 RD 提升已证实。方案、完整参数和可执行命令见 [C_PRECISION_V2.md](C_PRECISION_V2.md)。

## 实际通过的验证

| 检查 | 结果与范围 |
| --- | --- |
| CPU / Gloo 回归 | **53 passed**，188.91 秒；codec、loss、backend、效率、trainer、evaluation、supervisor、data 和 precision v2 的 9 个测试文件 |
| CUDA 回归 | **8 passed**，45.33 秒；V100S 32GB，实际 FP16 + GradScaler；6 个已有 GPU 测试、2 个新增 v2 C/D trainer 测试 |
| 最终 v2 集成回归 | **14 passed**，76.65 秒；最终修改后整份 `test_c_precision_v2.py` 重跑，包含 Gloo、CUDA、共享 embedding 第二步梯度和 mixed quality 拒绝检查 |
| 完整公共模型 512 | **C / D 均通过**；完整公共 SANA / DC-AE、LoRA、4MSE + LPIPS + DISTS + base + R、主优化器和熵分位数更新；CPU FP32，各 1 次更新后真实编码 / 序列化 / 解码 |
| 共享 embedding 学习 | 零初始化条件头首步有梯度；追加的第二步检查确认共享 embedding 本身出现有限非零梯度 |
| 静态检查 | 修改的 Python 文件编译、shell 语法、git diff whitespace 检查通过 |

CPU 回归调用：

```bash
GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q tests/test_bt_codec.py tests/test_bt_losses.py \
  tests/test_bt_backend.py tests/test_bt_efficiency.py tests/test_bt_training.py \
  tests/test_bt_evaluation.py tests/test_bt_supervisor.py tests/test_bt_data.py \
  tests/test_c_precision_v2.py
```

GPU 回归调用：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q tests/test_bt_cuda.py tests/test_c_precision_v2.py \
  -k 'cuda_amp or GPUPrecisionTest'
```

两卡 Gloo 在默认执行沙箱中因回环 socket 被禁止而失败，使用允许本机 socket 的执行环境后通过；不是跳过分布式检查。完整 512 首次尝试遇到 pyiqa 感知权重缓存路径只读，映射到已有完整缓存后重新运行通过。检查脚本随后把默认缓存设置提前到模型导入前，避免 pyiqa 过早固定错误目录。

## 完整 512 检查记录

原始 JSON 见 [test_reports/c_precision_full_512.json](test_reports/c_precision_full_512.json)。使用固定 Kodak 图的 512 裁剪，仅为工程检查，没有训练集调参或 RD 结论。

| 指标 | C | D |
| --- | ---: | ---: |
| 前后向、更新和码流检查总用时（CPU） | 155.79 秒 | 141.91 秒 |
| 裁剪后的 Router 梯度范数 | 0.0209611 | 0.0101687 |
| analysis precision 条件头梯度 | 0.0756432 | 0.0432372 |
| synthesis precision 条件头梯度 | 1.12997 | 0.637149 |
| 前向与真实码流重建最大误差 | **0** | **0** |
| 解码 SANA NFE | **1** | **1** |
| Router 图 / D 尺度图字节 | 24 / 0 | 24 / 32 |

首步共享 precision embedding 梯度为 0 是零初始化下游条件头的预期行为；条件头本身首步即更新，第二步梯度回到 embedding 的回归检查通过。该表的 CPU 用时包括加载、损失、编码、解码，不能用作 5090 训练每步耗时估计。

## 检查过的关键契约

- q：训练与码流统一 canonical q16；六个质量点均覆盖，当前 q 改变内容相关决策；条件分析不允许跨 q 缓存复用；v2 不接受 mixed quality 图。
- 输入融合：z / y 投影、归一化、融合及条件投影都有有限非零 RD 梯度；参考成本仅为无上下文提示，与正式 R 区分。
- 硬量化：训练重建及上下文保留硬符号，只有 R 使用噪声概率；eval 恢复硬概率。增加噪声不改变硬重建，训练噪声的 RNG 随断点精确恢复。
- 解码：非恒等条件下检查六档真实 rANS；符号和索引一致；新进程只有 codec 权重和包也能恢复；封禁 Router / clean encoder 后仍能解码。
- 传输：3-bit 图、D 4-bit 图、固定头和两种熵流全部计费；新增精度条件不引入额外包字段。
- 后端：C / D 均一次预测；D 只使用已发送尺度；冻结的 encoder 无梯度；repaired NMSE 只记录。
- 优化器：主参数和辅助分位数分离、两 rank 更新一致；LR 边界匹配配置；累积后的有效 batch 与启动设备契约一致。
- 交接：warmup EMA 标定并进入唯一共享 `backend_init.pt`；C / D 使用相同前端、公共 LoRA 和文本初始化；不同 λ 或 codec 配置拒绝交接。
- 恢复：连续与暂停后继续的 raw / EMA / scheduler 一致；保留既有计数器采样、原子保存、checkpoint 别名、批量 EMA 更新和守护进程行为。
- 验证：仅冻结 AE 缓存可跨 q 共用；真实 bpp 用序列化总字节；PSNR / MS-SSIM / LPIPS / DISTS 走完整图协议。

## 额外发现并修复的 EMA 问题

原 codec 的下采样 / 上采样层在 state_dict 同时导出 `down` / `branch1`、`up` / `branch1` 等同一参数的别名。原 EMA 通过默认去重的 named_parameters 建立平均集合，只对一个名字平均，另一个名字被拷贝 raw；load_state_dict 随后加载别名时会覆盖已平均的值。

本次改为对所有导出参数名字执行相同 EMA，并增加“加载后权重仍等于预期 EMA”的测试。CUDA 批量更新与逐参数参考连续三次更新完全一致的测试也通过。旧 checkpoint 仍可读取；该修复不会重建历史缺失的 EMA，更不能据此认定此前 RD 退步由它造成。

## 测试环境与剩余验证

本机 Python 3.12，torch 2.12.0+cu126、diffusers 0.35.2、peft 0.18.1、pyiqa 0.1.16、compressai 1.2.8、tensorboard 2.20.0。`requirements-c-precision.txt` 单独记录实际测试的 Python 依赖，避免沿用旧清单中的 peft / pyiqa pin；保留目标机器上能正常支持 5090 的 torch / torchvision CUDA 构建。

本机只有正在运行原实验的两张 V100，因此完整公共模型在 CPU 上检查，CUDA 只跑约小模型规模的短程验证。完整 512 的双卡 CUDA、5090 原生 BF16、batch8 warmup 和 batch2 / accumulation4 e2e 的显存、吞吐、暂停恢复，应在 5090 用随代码提供的 `preflight_bt_self_flow.py --smoke` 验证。代码不会自动把这些未跑的检查标为通过。

0.01–0.1 bpp 覆盖、Router 是否形成更有价值的分配、相对历史 baseline 的 BD-rate，都仍是新训练需要回答的问题。未强制八档使用比例，也没有通过伪造曲线或改变评测协议保证结果。
