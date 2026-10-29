# 新 C + cross-rate evidence Self-Flow

正式训练：E3，物理 GPU1、GPU2，每进程 batch 4、累积 1，有效 batch 8；200K 次成功更新。初始化使用与新 C 相同的已完成 50K codec EMA 和共享原生后端初始化。初始化文件虽然位于 PiT 运行目录，其内容是 `backend_init/shared/step=0`，没有 PiT 参数或联合训练优化器。

## 模型与目标

同一当前模型承担两个角色，不创建或更新独立 teacher/EMA teacher。冻结 DC-AE encoder 只编码一次原图；同一个质量条件 codec 分别在 q_low、q_high 做完整 analysis+quantization。高码率不能重用低码率 analysis。

两条生成路径都从 mu_low 出发，使用 delta_low 做原生 precision modulation 和 scheduler anchor。Student 的 cross-attention 证据为低码率 `(mu_low, delta_low)`；teacher 的证据为高码率 `(mu_high, delta_high)`。证据 token 为共享 SANA patch embedding 加零初始化的 precision projection，然后经过原生 caption_norm；不再经过 caption_projection。固定 text projection 保留用于旧初始化兼容，但冻结并从优化器排除。

Teacher 完成与 student 相同的整图修复任务，最终重建损失训练共享后端。SF 为正式 student H8 与 teacher H20 的逐 token FP32 cosine 距离；仅 H20 用作 SF target 时 stop-gradient。没有额外 P/G、辅助读出或区域混合。

全局梯度裁剪前，codec 所有主参数（包括共享 precision embedding）只接收原来的 `D_student + 0.1*base_NMSE + lambda_R*R_low`。共享后端接收 `0.75*D_student + 0.25*D_teacher + gamma*SF`。`D=4*MSE+LPIPS_alex+DISTS`。lambda_R、量化器、熵辅助更新、学习率日程与原生 C 保持一致。全局 norm=4 裁剪仍可能间接改变前端步长；训练日志记录裁剪系数，不承诺最终 bpp 不升。

q_low 的原生六档采样、数据/裁剪 counter stream 不变；q_high 使用独立 stream 9001+rank，均匀选取严格更高档。高码率额外量化使用独立 RNG scope，不消耗学生的随机数。q_low=1 或 E1 时只执行一次生成，teacher 重建等于 student，SF=0。部署只接收原有 v3 低码率包，执行一次生成；无需高码率证据或新增 side bits。

## 梯度同步和恢复

参数分别求导需要 `autograd.grad`，因此本方法不用不支持它的 DDP reducer。两进程启动时广播模型；每次更新对累积的主梯度显式 all-reduce 并取平均，然后沿用原生统一裁剪、AdamW、熵辅助更新及部署 EMA。全局未使用的参数保留 `grad=None`。训练仍使用 torchrun/NCCL 两进程数据并行。

沿用原生完整 checkpoint：raw/部署 EMA、两个优化器、调度器、AMP、每 rank RNG、数据 counter 和合同。新增 context projection 纳入严格恢复和 EMA。恢复要求 world size、batch、优化配置一致。

## 四个开关配置

- `e0_native.yaml`：相同数据、两卡/总 batch 和 RD 设置，原生新 C，无证据接口。
- `e1_interface.yaml`：同证据接口，teacher 使用低码率；无 SF。
- `e2_privileged.yaml`：高码率 teacher 整图修复训练；无 SF。
- `e3_self_flow.yaml`：E2 加 H8→H20 SF。

本次只启动 E3。E1/E2/E3 的架构和初始化一致；E0 的接口不同，不声称初始出图完全相同。对照须用同一 50K EMA、共享初始化及相同两卡划分。训练步数和独立图片数匹配，双前向的计算量不匹配。

## 启动和查看

环境 `/data1/ls/envs/cvpr`，缓存复用 `/data1/ls/pretrained/pretrained/torch`。从本工作树运行：

```bash
/data1/ls/envs/cvpr/bin/python scripts/launch_evidence_self_flow.py \
  --run-root /data1/ls/code/runs/newc_self_flow_gpu12_20261030 --gpus 1,2
```

启动器脱离终端运行，记录 `launch.json`、`launcher.log`；训练记录 `joint/resolved.json`、`joint/train.jsonl` 和 TensorBoard。第 10 步额外保存恢复点；之后每 2K 步验证并存储，保留最新两份数字 checkpoint 及验证 best。论文主结果固定使用最终 200K EMA，不能逐码率挑 best。完整恢复显式传 `--resume .../joint/paused.pt`，不能与原始初始化混用。

验证沿用 Kodak24 整图、六档实际序列化总 bpp，每次做两张真实包解码，每 10K 全量包解码。最终同码率增益仍需训练后在共同 bpp 区间比较，当前预检不证明研究收益。原研究计划的独立 DIV2K-valid/CLIC-valid 测试清单尚未落地，正式外部测试前须固定清单；Kodak 是开发验证集。

工程检查记录在运行目录：梯度与原生 RD 独立参照、两进程主/辅助更新及精确恢复、fresh receiver、EMA 和原生 C 回归；真实模型预检额外执行完整非零 SF 更新和最高质量无 teacher 更新，检查每卡 batch 4 的显存、复制一致性与 EMA 包解码。预检更新不用于正式初始化。
