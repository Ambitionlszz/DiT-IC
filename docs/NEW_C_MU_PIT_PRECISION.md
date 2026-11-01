# New C + B with additive precision AdaLN

This extends the mu-Q/K main model B from `f2853de`. It uses a separate v4
method, `precision_consistent_mu_qk_precision_adaln_pixel_adapter_v4`, and
`configs/bt_self_flow/precision_v3/mu_pit_b_precision_joint.yaml`. Original
mu-only B and control A retain their methods and complete checkpoint formats.

## Exact change

The receiver computes C's existing shared 128-channel precision embedding
from the actual decoded quantization step: `precision_embedding(log(delta))`.
It aligns this field to the PiT region grid using nearest-neighbor expansion,
retaining quantization-cell boundaries. Each PiT block owns a separate
`Linear(128,64)` with zero weight and bias, emitting four 16-channel terms:

```text
shift_p_attention, scale_p_attention, shift_p_mlp, scale_p_mlp
```

Each precision term is shared by the 16x16 pixels in that region. Mu still
produces distinct parameters for each pixel. For attention and MLP separately:

```text
normalized = LayerNorm(pixel)
modulated = normalized * (1 + scale_mu + scale_precision)
            + shift_mu + shift_precision
```

Mu alone supplies Q/K and the original two pixel residual gates. V comes from
the compact modulated pixel features. The precision branch adds no gate,
normalization, nonlinear projection, bits or SANA call. All original LayerNorm,
RoPE, packing, positions and zero final writeback remain in place. Shared
precision parameters receive gradients normally; nothing is detached/frozen.

The new heads are appended after initializing the complete original B, so all
old tensors keep exactly the same seeded initial values. The total new D has
17,393,968 parameters, an increase of 16,512. Even with an active final output
projection, zero precision heads reproduce original B exactly. With the final
projection also zero, the image path starts as native C.

## Initialization and training

Use the same native new-C shared backend initialization:

```text
/data1/ls/code/runs/newc_pit_gpu0_20261029/backend_init.pt
SHA256: 7286c89f56637c0c94e5d3cb8fb34801f849a44816a8977a5c579973b11efaa8
```

This comes from the completed new-C 50K warmup EMA. Formal joint training
starts at update zero with fresh joint optimizers, not from the old PiT's
trained decoder. The inherited joint recipe uses batch 8, accumulation 1,
effective batch 8, seed 903, 200K successful updates, six quality/rate pairs,
the original optimizer and milestone schedule, and 4 MSE + LPIPS + DISTS plus
the existing base-latent/rate terms. There is no SF term or additional freeze.

Explicit shared initialization starts v4 D fresh; ordinary load/resume requires
every precision-head tensor and the architecture-version marker. Old B
checkpoints are rejected as complete v4 states. New precision parameters use
the existing pixel-adapter optimizer group and deployment EMA.

The de13052 numerical repairs remain unchanged. Full AE-tail/perceptual
checkpointing and the existing BF16/recovery policies remain active.

## First-free GPU1/GPU2 queue

`scripts/queue_precision_mu_pit.py` monitors the user-authorized physical GPU1
(`GPU-ccb69b0d-acbb-96eb-9f08-b91776b96d65`) and GPU2
(`GPU-820974f0-0aed-b2b3-236a-04bacb2df8e8`) on this host. One supervisor
selects the first idle card and launches exactly one experiment. Preflight and
formal training stay on that same physical card. Use a dedicated pinned, clean
checkout; the run plan records its commit and file hashes.

Preparation is CPU-only and compares native C and v4's actual initialized
codec/LoRA/conditioner tensors, constructor RNG, pretrained assets, training
contract, data order, learning rates and new-head initialization:

```bash
source /data1/ls/activate_cvpr.sh
python scripts/queue_precision_mu_pit.py --prepare-only \
  --run-root /data1/ls/code/runs/newc_mu_pit_b_precision_gpu2_20261101 \
  --gpus 1 2 \
  --init-checkpoint /data1/ls/code/runs/newc_pit_gpu0_20261029/backend_init.pt
```

The existing run-root name is retained for continuity; it does not restrict
device selection. `--gpus 1` or `--gpus 2` limits a newly prepared plan to one
card. For `--watch`, the candidates come from the saved plan.

Start `--watch` from the same checkout in a detached process, with stdout and
stderr redirected to `supervisor.log`. It polls every five seconds and holds
no CUDA allocation while waiting. Two consecutive polls must report at most
256 MiB used and at most 5% utilization. It then runs three complete pretrained
batch-8 engineering updates, including original image losses, auxiliary
optimizer, EMA, initial C parity and actual packet decoding. Those engineering
updates are discarded.

Only after that report passes does the watcher initialize formal B from the
original shared checkpoint. Before releasing the first-update gate, it checks
the actual trainer PID, batch, accumulation, config contract, dataset
fingerprint, assets and source commit against the audited plan. It saves an
extra recovery checkpoint at update 10. It never selects A or the earlier B.

Status and logs are under the run root:

- `status.json`, `supervisor.pid`, `supervisor.log`: watcher state and identity.
- `selected_gpu.json`: the one physical card selected for this experiment.
- `plan.json`, `strict_ablation_audit.json`: source/init/config/data identity.
- `preflight/full_batch8.json` and `.log`: actual GPU preflight result.
- `joint/train.jsonl`, `train.stdout.log`: formal updates and trainer output.
- `joint/0000010.pt`: early complete recovery state when update 10 is reached.

Create a `STOP` file in the run root to cancel waiting, or request a clean
training pause through the watcher. Preflight/config/numerical failures are
recorded as failed and are not silently retried with another model, batch size
or device. The watcher also stays attached to the child to record its exit.

## Validation scope

Tests cover exact old-B initialization, additive affine/broadcast math, unchanged
mu Q/K, three-step gradient opening, FP32/BF16 checkpointed tuple conditions,
whole decoder and joint loss, six native packet qualities, a fresh-process
receiver, strict checkpoint/EMA boundaries and exact trainer continuation.
Queue tests simulate busy/idle transitions, either card becoming idle first,
retaining that card after preflight, rejecting unauthorized or duplicate cards,
failed preflight, wrong trainer contract/source/method, successful startup gate
and cancellation.

Production-width adapter results are recorded in
[the CPU report](validation/newc_mu_pit_precision_adapter_cpu_20261101.json).
The full pretrained batch-8 GPU preflight is deferred until GPU1 or GPU2 becomes free;
no RD gain is established by these engineering checks.

Completed on 2026-11-01: CPU regression **187 passed, 13 skipped**; CUDA
supplement **51 passed**; latest queue checks **12 passed**. An independent
subagent found no blocking issue and ran 12 targeted CPU checks. Validation
counts overlap and should not be added as distinct cases. Details are in
[the validation record](validation/newc_mu_pit_precision_validation_20261101.json).
