本目录保存 2026-10-19 主机重启后的 C/D 续训核查与补充评测。

训练从 C 36,000、D 38,000 的完整 checkpoint 恢复，目标仍为各自 200,000 个 optimizer update，冻结配置没有变化。`resume_acceptance.json` 记录重启后确实新增的首个更新及运行进程；旧日志中未保存的更高步数不代表恢复后的实际步数。

启动时发现原文件枚举在冷缓存上逐个 stat 89,026 张图片，导致长时间等待。提交 `2845a1d595e346b2e3c273d95c743e22820fbb61` 改用 scandir 的目录项信息。完整路径清单、顺序和指纹完全相同，见 `inventory_acceptance.json`。`manifest.before_inventory_fix.json` 和 `resume_inventory_fix.json` 保存运行清单的变更前记录及原因；只更新代码提交标识，没有修改配置。

最终表来自 `C_36000` 与 `D_38000_gpu0`，都是 deployment EMA，在 Kodak24 原图上逐图进行真实 rANS 压缩包编码与解码，单次 SANA 预测。所有五项指标使用同一份浮点重建；MS-SSIM 使用 pyiqa 0.1.16 默认 Y 通道 / yiq 协议。bpp 包含包头、主码流、超先验、分配图和 D 的尺度图，分母为原图像素数；逐图求指标后再取算术均值。

`D_38000` 是为避免 GPU1 显存冲突而启动、随后停止的 CPU 补评测中间目录，不参与最终统计。C 完整评测结束后，D 改在 GPU0 以 19.5% 分配器配额评测，启动前检查额外 1,280 MiB 余量；训练均持续运行。内存核查保存在 `D_gpu0_evaluation_memory_review.json`。

生成表格：`python3 build_tables.py`。脚本要求每组 144 条记录、每档 24 张图、指标有限、包字节精确计费、两组数据指纹相同，并与原验证记录核对 PSNR / LPIPS / DISTS / bpp。最终输出 `metrics.csv`、`metrics.json`、`metrics.zh.md`。
