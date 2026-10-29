# Self-Flow 各版尝试归档

归档日期：2026-10-29。保存服务器上与 Self-Flow（SF）相关的源码、配置、评测和研究记录，便于换服务器后继续定位与复现。

## 从这里开始

| 版本 / 尝试 | 原 GitHub 分支 | 本归档中的入口 |
| --- | --- | --- |
| BT Self-Flow 早期 C/D 联训 | [bt-self-flow-v1](https://github.com/Ambitionlszz/DiT-IC/tree/codex/bt-self-flow-v1-20261015) | [源码快照](code-snapshots/DiT-IC-bt-self-flow)、[运行记录](run-records/bt_self_flow_v1) |
| 低码率 C 与从 30K 接入的 SF | [lowrate-c](https://github.com/Ambitionlszz/DiT-IC/tree/codex/bt-lowrate-c-ddp-20261019)、[from30k](https://github.com/Ambitionlszz/DiT-IC/tree/codex/bt-lowrate-c-from30k-20261020) | [低码率评测](run-records/bt_self_flow_lowrate_20261019)、[对照](run-records/bt_self_flow_comparison_20261019) |
| 10K / 20K 后接入、batch 与恢复尝试 | 源目录没有独立 Git 元数据，按服务器文件快照保留 | [start](code-snapshots/DiT-IC-self-flow-start-20261022)、[b4](code-snapshots/DiT-IC-self-flow-b4-20261022)、[10K recovery](code-snapshots/DiT-IC-self-flow-sf10k-recovery-20261023)；运行记录见 `run-records/cd_sf_*` |
| 量化残差 SF 与稳定性 / 恢复尝试 | [quant-residual](https://github.com/Ambitionlszz/DiT-IC/tree/codex/c-quant-residual-self-flow-20261025) | [源码](code-snapshots/DiT-IC-c-sf-residual-20261025)、[设计](experiments/c_sf_plan_20261025/proposal.zh.md)；运行记录见 `run-records/c_quant_residual_sf_*` |
| 有效 batch 8 的 C / C+SF 消融与 weight=1 | [ablation-b8](https://github.com/Ambitionlszz/DiT-IC/tree/codex/c-sf-ablation-b8-20261026) | [实验说明](experiments/c_sf_ablation_b8_20261026/README.zh.md)、[源码](code-snapshots/DiT-IC-c-sf-ablation-b8-20261026)、[weight1](run-records/c_sf_weight1_20261026) |
| 双质量辅助任务 SF | [dual-rate-task](https://github.com/Ambitionlszz/DiT-IC/tree/codex/c-dual-rate-task-sf-20261027) | [源码](code-snapshots/DiT-IC-c-dual-rate-task-sf-20261027)、[28K 评估](experiments/c_dual_rate_task_sf_20261027/assessment_20261027_28k/assessment.zh.md) |
| Native Relational SF | [native-relational](https://github.com/Ambitionlszz/DiT-IC/tree/codex/c-native-relational-sf-20261027) | [源码](code-snapshots/DiT-IC-c-native-relational-sf-20261027)、[诊断](experiments/c_native_relational_sf_20261027/diagnostics/diagnosis_20261028.zh.md) |
| Local Precision SF | [local-precision](https://github.com/Ambitionlszz/DiT-IC/tree/codex/c-local-precision-sf-20261028) | [源码](code-snapshots/DiT-IC-c-local-precision-sf-20261028)、[码率漂移诊断](experiments/c_local_precision_sf_20261028/failure_analysis_12k.zh.md) |
| Context SF | [context-sf](https://github.com/Ambitionlszz/DiT-IC/tree/codex/c-context-sf-20261028) | [源码](code-snapshots/DiT-IC-c-context-sf-20261028)、[20K 结果](experiments/c_context_sf_20261028/comparisons/status_20000.zh.md)、[运行状态](experiments/current_c_sf_run.json) |
| 未开训 / 中间修订的 SF 方案 | — | [下一版设计](experiments/c_sf_next_design_20261027)、[重设计](experiments/c_sf_redesign_20261027) |

## 保存了什么

- `code-snapshots/`：服务器各 SF 工作目录的源码、脚本、配置和说明的文件快照。包含已提交版本及本地修改，不依赖原服务器的 `.git` 指针或 ext4 挂载路径。早期 `DiT-IC-bt-self-flow` 中 `models/bt_self_flow/checkpoint.py`、`validation.py`、`train_bt_self_flow.py` 的本地修改也保存在快照中。
- `experiments/`：各版方案、比较表、诊断脚本和实验状态。
- `run-records/`：从训练目录提取的配置、启动/迁移记录、训练日志、验证指标与 checkpoint 元数据。
- `results/`：SF 对照与准备评估的文本 / 表格结果。
- `reference/pure-c/all_quality_points.csv`：当前 SF 评估引用的外部纯 C 基线。
- `manifest.json`：每份原始文件的来源相对路径、归档路径、字节数及 SHA256。
- `patches/early-bt-uncommitted.patch`：早期 BT 目录相对原提交的本地改动，便于单独检查或恢复。

目录快照用于保存当时实现。原分支链接用于追踪已有提交历史；两者有差异时，以快照和 manifest 所记录的服务器文件为准。归档分支根目录沿用 context SF 的代码。

## 最新已记录状态与结论边界

截至归档时，context SF 已正常保存并暂停在 20,000 步，其本分支最佳验证点为 16,000 步。20K 同 bpp 对照未超过纯 C；16K 部分感知指标曾小幅改善。稳定训练和 SF 对齐 loss 下降均不等价于压缩收益，具体数据、限制与解释以各版原始报告为准。

Native Relational 的记录包含 FP16 前向非有限值故障；Local Precision 的记录包含码率漂移和梯度诊断。保留这些失败和未证实的假设，避免以后重复同一条路线，或把局部诊断误写成最终结论。

## 换服务器后使用

先在上表找到目标版本，查看其源码快照、`configs/` 和相应 `run-records/`。配置中的 `/data2/ls`、解释器和 checkpoint 路径是原服务器记录，需要按新机器调整。早期各版依赖可能不同，使用对应分支或快照中的 requirements 与运行脚本。

本归档不包含 training-free 尝试、模型权重、数据集、生成图像、码流、张量载荷、运行环境或缓存。续训需要另行迁移对应 `.pt` 文件、初始化权重、SANA / ELIC 等依赖模型及数据集；运行记录中的校验和与相对目录可用于核对。

归档过程没有修改原实验代码或启动训练。验证以文件内容校验和、上传文件范围检查及远端提交核对为主，没有重新运行这些历史实验。
