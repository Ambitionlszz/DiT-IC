# 为什么 DiT-IC 实验选 Gemma 而不是 InternVL

**SANA 中两者分工不同：**

| | InternVL | Gemma-2-2B |
|---|---|---|
| 角色 | VLM，给图像**生成** caption | Text Encoder，把 caption **编码**成 2304d embedding |
| 何时用 | 离线标注（训练前跑完） | 每次 forward 都跑 |
| 在 DiT 中的位置 | 不在模型内 | 直接输入 DiT 的 `encoder_hidden_states` |

**DiT-IC 要做的事：** 用 InternVL 预先给所有训练图打 caption，然后用 Gemma 把这些 caption 编码后拿来做对比学习（训练一个 latent prompter 去对齐 Gemma embedding），从而推理时不再需要 text encoder。

**所以选 Gemma 的原因：** DiT 实际"看到"的语义信号来自 Gemma 编码后的 embedding，实验的对比学习目标也是对齐 Gemma embedding。和 InternVL 无关——InternVL 只是生成 caption 的工具。
