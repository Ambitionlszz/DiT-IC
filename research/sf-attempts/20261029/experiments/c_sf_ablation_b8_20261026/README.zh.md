# 有效 batch 8 的 C / C+SF 对照准备

状态：固定版本 ckpt 已完整下载、通过 SHA256 校验，双分支状态迁移已回读核对。旧实验已在 21589 step 保存退出，新 C+SF 已完成 10001 step 实际更新；随后在 12K 自动切到纯 C 对照。

代码版本：`2a59bac`。
GitHub 分支：https://github.com/Ambitionlszz/DiT-IC/tree/codex/c-sf-ablation-b8-20261026

| 配置 | 两条分支的共同设置 |
|---|---|
| 输入 | 512 × 512 |
| Batch | 双卡，每卡 4，累积 1，总 batch 8 |
| 六档 q | 0 / .2 / .4 / .6 / .8 / 1，每个 optimizer step 独立随机抽一档；两卡、两条分支使用相同抽档序列 |
| 图像损失 | 4 MSE + 1 LPIPS + 1 DISTS |
| Latent 损失 | 0.1 base NMSE；repaired NMSE 权重 0 |
| 码率权重 | 随 q 分别为 16 / 8 / 4 / 2 / 1 / .5 |
| 主学习率 | Codec、DiT、VAE 均 1e-4；SF head 同为 1e-4 |
| 辅助学习率 | 1e-3 |
| 优化器 | AdamW，betas (.9, .999)，weight decay 0，梯度裁剪 4 |
| LR 衰减 | 100K 乘 .5，160K 再乘 .2，190K 再乘 .1 |
| 精度 | 两条分支统一 FP16 + GradScaler，使用相同 FP32 数值修复 |
| 评测 | Kodak24，deployment EMA .999，包含 side information 的总 bpp |
| 保存 / 日志 / 验证 | 周期 10K / 100 / 2K；验证刷新 best 时也可能保存 |
| 总训练预算 | 累计 200K，包含已有 10K |

纯 C 使用原始 `BTCodec`，不添加 residual head。C+SF 使用已实现的量化残差机制：
student 第 8 层预测 teacher 第 20 层的精度残差，预测修正进入 student 第 20 层，
student 同时接受重建和码率目标。Teacher 对原精度与局部降精度 latent 作比较，
冻结梯度、EMA .999；student 的局部降精度保留其内部非对称性。两卡各有 teacher 副本。

接收到同一份纯 C 10K checkpoint 后，先保留该状态，再接入 SF，从第 10001 次更新起
用 2K 步 ramp，在 12K 达到全权重；每条样本被选择的概率最终为 .5，选中的样本局部
降精度比例为 .25。SF 系数在该 checkpoint 上重新校准，目标梯度比例 .05，
每档 4 个校准样本，不复用之前 batch 16 的 alpha。

完整展开配置为 `pure_c.resolved.json`、`c_sf.resolved.json`，准备状态在 `plan.json`。
展开的 `self_flow.high_fraction/calibrate_every/gradient_cap` 是继承的旧方法字段，
当前 `quantization_residual_v1` 不读取这些字段；当前机制使用 `cell_fraction` 和一次性校准。

源 checkpoint 元数据确认是双卡每卡 2、累积 2（有效 batch 8），并非先前描述的单卡每卡 8。
V100 不支持原生 BF16，因此历史 BF16 纯 C 曲线用于参考；两条新的 FP16 续训
才用于判断 SF 带来的收益。文件到位后，以 checkpoint 内嵌的纯 C 配置逐项核对，
保留 raw/EMA、Adam 和辅助优化器状态、scheduler、两卡 RNG 与数据全局计数。
micro_seen 从 20000 转为 10000，均代表已见 80000 张样本；两条分支使用相同的新 FP16 GradScaler。当前准备脚本只做配置展开和元数据检查，不会自动迁移或开训。

38 项相关测试已通过，包含 CPU 双进程 SF 启用、DDP 辅助优化器和启用后的精确恢复。
实际 checkpoint 的加载一致性、SF 校准及 GPU 续训检查，等待文件到位后执行。

最新核对：配置项与预训练资产一致；源训练图片列表按根目录重定位后，fingerprint 仍不匹配，待查明。尚未启动迁移或新实验。

源数据说明：用户确认只有目录/文件布局调整。新的 C 与 C+SF 使用同一份本机 89,026 张训练图片；路径指纹差异记录在 migration.json。

启动检查通过：10K 基线与源 CSV 六档 PSNR 最大差 0.00879 dB；新 SF alpha=0.641302792947235，首个更新 loss/梯度有限。即时 privileged 重建检查仍未改善，SF 的实际收益须由新对照的 12K 评测判断。
