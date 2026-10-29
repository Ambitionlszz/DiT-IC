# C+D，512输入，每卡4，累积2，双卡

用户在2026-10-22将每卡batch从2改为4，累积仍为2。新全局有效batch为16。

- 原2/2/2任务安全暂停于472个optimizer updates：`/data2/ls/checkpoints/cd_sf_after20k_20261022/joint/paused.pt`。
- 本目录接续该断点，从第473次更新继续；raw权重、deployment EMA、主/aux optimizer、scheduler、AMP scaler、各rank随机状态、更新次数全部保留。
- 用明确的离线迁移工具更新batch契约，正式训练器的严格resume检查保持不变。迁移只允许batch=2→4和effective_batch=8→16，禁止夹带其他训练修改。
- CounterSampler的micro_seen从944换算为472，global_images_seen仍为3776；后续各rank重新分组，下一段全局图片/裁剪序列连续，不跳过或重复旧样本。
- 学习率峰值与日程保持原配置，未随batch翻倍。主学习率warmup 500 steps从已有472步继续，训练总计200K updates。
- 原始latent warmup仍为旧BT v1 50K raw codec，保留原C+D后端初始化和既有训练状态。

## 训练配方

训练图像512×512。每卡实际输入[4,3,512,512]，累积2次，两张V100S 32GB做DDP，global batch=16。

主损失：`4*MSE + LPIPS + DISTS + 0.1*base + beta(q)*rate`；`repaired_weight=0`，仅记录修复latent指标。codec完整训练；后端保留既有LoRA/条件模块/文本投影及AE decoder LoRA优化设置。FP16 AMP，梯度检查点。

完成总更新20000步后，从当前raw student复制teacher；从第20001次更新启用8→20 Self-Flow，2K ramp，教师EMA=0.9999。每卡各有一份teacher，SF继续向student前端/后端反传。混合/正常更新交替，混合更新随机25%编码位置取q=1，其余为当前q；teacher全图q=1。相同student视图同时做重建与对齐。SF接口梯度目标5%、上限10%，每500步校准。

Kodak24原图六档实际码率验证每2K进行；每2K保存，保留最近3份及最佳，每10K完整码流解码校验。

## 文件与入口

- `configs/joint.yaml`：冻结正式配置。
- `batch_migration.json`：断点SHA256、batch和数据计数换算记录。
- `joint/paused.pt`：离线迁移断点（正式运行后正常暂停会更新它）。
- `preflight/verification.json`：batch4真实双卡SF短测及显存结果，完成后生成。
- `pipeline.json`、`logs/joint.log`、`joint/train.jsonl`：正式状态与训练日志。
- `batch_changes.patch`：相对上一个batch2源码副本的变更。
- 原方法和工程细节：`/data2/ls/checkpoints/cd_sf_after20k_20261022/README.zh.md`。

工作目录：`/data2/ls/code/DiT-IC-self-flow-b4-20261022`。

```bash
/tmp/bt-self-flow-runtime-20261016/linux/venvs/ditic/bin/python scripts/launch_delayed_self_flow.py --gpus 0,1
```

启动器使用独占锁和代码/配置SHA256清单；自动从本目录最新有效断点恢复。安全停止应向pipeline.json中的supervisor_pid发送SIGTERM，等待trainer在更新边界保存，不直接杀torchrun或DataLoader。

验证完成：batch4真实双卡6步短测通过（2步RD＋4步SF，中途保存恢复），SF系数和梯度均有效，无OOM；PyTorch显存峰值27,843,873,280 bytes，SF运行时驱动显示约27.7 GiB/卡。初始AMP自动降低scale并重试3次，随后正常更新。迁移前后5200个实际断点张量全部一致，batch契约和全局样本进度核对通过。
