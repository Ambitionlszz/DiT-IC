# V100 与 RTX 5090 运行最近的 BT 修改

本分支包含 2026-10-19 的低码率共同 warmup、C_aux / C_no_latent 对照、
FP16 稳定性修复与训练恢复协议。代码和配置在 GitHub；训练数据、预训练权重、
实验 checkpoint 仍从各机器的本地存储读取。

## 分别安装环境

在新的 Python 3.12 虚拟环境安装，避免改变正在训练的环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# V100 二选一：
python -m pip install -r cloud/v100/requirements.txt
# RTX 5090 二选一：
python -m pip install -r cloud/rtx5090/requirements.txt
```

V100 使用当前训练的 PyTorch 2.12.0 / torchvision 0.27.0 CUDA 12.6。
5090 的候选环境为 PyTorch 2.8.0 / torchvision 0.23.0 CUDA 12.8，需要支持
该 GPU 和 CUDA runtime 的 NVIDIA 驱动。不要把 V100 的 cu126 环境直接复制
到 5090；训练代码会在 5090 选择原生 BF16，在 V100 选择 FP16 + GradScaler。
具体精度与内核是否可用，以目标机器验证报告为准。

官方依据：[Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/)、
[匹配的 PyTorch / torchvision 安装包](https://pytorch.org/get-started/previous-versions/)。

## 设置目标机器的路径

以下为 5090 的示例；先确认真实目录。SANA 目录必须包含 transformer/vae 的
config.json 与 diffusion_pytorch_model.safetensors，以及 scheduler/scheduler_config.json。
固定文本 embedding 使用与 V100 相同的文件。

```bash
export BT_DATA_ROOT=/data2/luosheng/data
export BT_SANA_PATH=/data2/luosheng/checkpoints/SANA
export BT_TEXT_PATH=/data2/luosheng/checkpoints/ditic-50k-warmups/pretrained/fixed_text_embed_gemma.pt
export BT_RUN_ROOT=/data2/luosheng/checkpoints/bt_lowrate_5090
export BT_CACHE_ROOT=/data2/luosheng/checkpoints/bt-cache
export TORCH_HOME="$BT_CACHE_ROOT/torch"
export HF_HOME="$BT_CACHE_ROOT/huggingface"
```

BT_DATA_ROOT 替换内置配置的 /data2/ls/data 前缀，保留其下面的 DIV2K、Flickr2K、
LSDIR、CLIC、Kodak 相对目录；自定义的其他数据路径保持原值。
路径在启动器固定 manifest 前解析，实际训练和预检使用同一份配置。
不设置覆盖时，原 V100 配置含义保持不变。启动脚本使用当前环境的 python3，
也可用 BT_PYTHON 显式指定；输出位置由 BT_RUN_ROOT 指定。

## 验证并启动

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_bt_gpu.py --expect 5090 --output /tmp/bt-5090-kernels.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests -q -p no:cacheprovider
# 完整 512px 模型验收需要空闲显存、所有数据和权重，会创建独立 smoke checkpoint：
python scripts/preflight_bt_self_flow.py --gpus 0,1 --smoke --output /tmp/bt-5090-smoke
# 两张卡执行最近的低码率 C 对照：
python scripts/launch_bt_c_ablation.py --root "$BT_RUN_ROOT" --gpus 0,1
```

V100 的小型内核检查把 --expect 改为 v100。check_bt_gpu 只验证卷积、注意力、
AMP 前向反向与架构，不能替代完整模型 smoke。完整 smoke 使用标准 C/D 工程
流程检查实际训练、暂停恢复和码流；低码率配方另由配置/训练协议回归覆盖。
两组实验启动器需要两张卡；单卡调试应使用 train_bt_self_flow.py 和明确的配置，
同时核对有效 batch。更换 GPU、精度或 PyTorch 版本不保证逐位一致。

已有 V100 checkpoint 可以在 5090 评估：设置 BT_SANA_PATH / BT_TEXT_PATH 后运行
eval_bt_self_flow.py，仍校验原预训练资产的 SHA-256。跨机器恢复训练仍受原有
数据指纹、路径、world size 与配置协议约束；不要绕过检查或直接搬移旧 manifest。

2026-10-19：V100S 实机的小型卷积/注意力 AMP 前向反向已通过，PyTorch
2.12.0+cu126 包含 sm_70。其架构列表不含 sm_120，不能作为 5090 的环境使用。
最近的 FP16 注意力稳定性、真实微型训练和码流 CUDA 回归 5 项通过。
路径覆盖、低码率配方、训练暂停恢复、分布式同步、独立评估与 GAN 协议
CPU 回归 31 项通过。报告保存在 V100 的 results/bt_gpu_portability_20261019。
5090 候选 requirements 的依赖解析已通过；这不代表 5090 实机训练通过。

本机没有 5090 SSH 私钥，当前未完成 5090 实机验收。仅当目标机器内核检查、
CUDA 回归与完整模型 smoke 均通过后，才能记录为已在 5090 验证。
