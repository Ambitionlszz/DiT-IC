# 低码率 C 的辅助损失对照

两卡先共同完成一次从零开始的 50K latent warmup，再共享同一个前端和
SANA/LoRA 初始化，各自端到端训练 200K。两个主实验都用 C，不使用 D 或 Self-Flow。

| 配置 | 共同 warmup | GPU0：C_aux | GPU1：C_no_latent |
|---|---|---|---|
| 更新数 | 50,000 | 200,000 | 200,000 |
| 每卡 batch | 32 | 2 | 2 |
| 梯度累积 | 1 | 4 | 4 |
| 有效总 batch | 64（两卡 DDP） | 8 | 8 |
| 量化步长 coarse/fine | 8 / 1 | 8 / 1 | 8 / 1 |
| 全局码率损失倍率 | 3 | 3 | 3 |
| 粗 latent 重建目标 | 主目标，保留 | D_base 权重 0.1 | D_base 权重 0 |
| 修复后 latent 辅助目标 | 不适用 | D_latent 权重 0.1 | D_latent 权重 0 |

目标是约 0.01–0.1 bpp，不将步长当成实际 bpp 的保证。固定旧前端的六张
Kodak 图像敏感性测试中，8/1 对应约 0.011–0.107 bpp；从零训练后，分析网络
可以改变特征幅度，所以码率项同时乘 3。实际范围以新训练的 Kodak 结果为准。
这个倍率在 warmup 和两组主实验中相同，并进入 checkpoint 的恢复训练协议。
旧配置默认倍率仍为 1。

验证使用完整 Kodak 24 张、六个质量点和包含全部头信息的真实压缩包。
每 2K 验证、保存 snapshot 并更新 best；保留最近三个 snapshot 和 best。
两组主实验的数据顺序、裁剪、质量采样、模型初始化和其他损失完全一致。
共同 warmup 的潜变量目标不会因为 GPU1 后续移除辅助损失而被移除。

DC-AE encoder 冻结，压缩前端和 Router 从零训练；后续 SANA/DC-AE 使用
原有预训练权重和 LoRA 协议。无额外的 5K 后端适配阶段。

```bash
python scripts/launch_bt_c_ablation.py \
  --root /data2/ls/checkpoints/bt_self_flow_lowrate_20261019 \
  --gpus 0,1
```

启动器固定配置和 Git commit，使用文件锁阻止重复启动；重复相同命令可恢复。
warmup 完成、校准通过后自动生成一次 backend_init，再自动启动两组主训练。
任一 C 失败时另一组继续。SIGTERM 只通知实际 trainer，让数据 worker 在
当前 optimizer update 和 paused checkpoint 保存完成前保持存活。

旧 C 在 36,944 step 暂停并保留；旧 D 在 38,403 step 停止并保留。
新实验采用独立目录，不覆盖旧 checkpoint。
