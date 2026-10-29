# 10K纯重建后接入Self-Flow

用户在2026-10-23指定：完成10000个optimizer updates并保存checkpoint后，从10001次更新接入8→20 SF。

## 实际接入顺序

1. 继续student纯重建训练至10000步。
2. 按既有日程完成Kodak24六档原图验证和完整码流解码检查。
3. 原子保存10000步的raw/deployment EMA、optimizer、scheduler、scaler、各rank随机数与数据进度，文件为joint/0010000.pt。
4. teacher创建之前另发布joint/pre_sf_0010000.pt，保留纯RD基线，不受最近3份checkpoint清理影响。只有写盘成功才继续创建teacher。
5. 从当前raw student初始化每卡teacher；第10001次更新开始混合SF，之后混合/正常交替，2K ramp，EMA=0.9999。

## 其余训练设置

512×512；每卡batch=4，梯度累积2，两张V100S，global batch=16。

主损失仍为4*MSE＋LPIPS＋DISTS＋0.1*base＋beta(q)*rate，repaired_weight=0。同一个student同时做重建与SF对齐。教师全图q=1，学生混合更新25%编码位置q=1、其余为当前q。

学习率、优化器、200K总更新数、SF的2K ramp、每500步校准、RD接口梯度5%目标/10%上限均保留。

## 重启恢复

10:14:57机器重启。之前9056步的暂停保存未完成，临时文件报I/O错误，不能当作恢复断点。原/tmp运行环境随重启丢失。

本实验从最近完成的8000步checkpoint恢复：`/data2/ls/checkpoints/cd_sf_after20k_b4a2_20261022/joint/0008000.pt`，保留该步所有权重、优化器、EMA、学习率和数据进度。8K至中断点之间的更新重新执行。旧checkpoint及异常临时文件保留，没有修改源盘镜像或运行磁盘修复。

运行环境从已有原始ext4镜像复制到/tmp，仅对可丢弃副本做文件系统检查；关键Python/Torch/CUDA文件按原SHA256核验。

## 路径

- 配置：configs/joint.yaml。
- 正式日志：logs/joint.log、joint/train.jsonl。
- 运行状态：pipeline.json。
- 10K纯重建基线：joint/pre_sf_0010000.pt（到达10K时生成）。
- 起点变更与恢复身份：sf_start_migration.json、reboot_recovery.json。
- 实现改动：sf10k_changes.patch。

源码：/data2/ls/code/DiT-IC-self-flow-sf10k-recovery-20261023。
启动器：scripts/launch_delayed_self_flow.py --gpus 0,1。
原batch4含teacher的GPU短测记录：/data2/ls/checkpoints/cd_sf_after20k_b4a2_20261022/preflight/verification.json；完整SF更新和恢复均通过，驱动显存约27.7GiB/卡。
