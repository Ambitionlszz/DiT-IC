# Continue the original C/D experiment with runtime optimizations

Base: `2845a1d595e346b2e3c273d95c743e22820fbb61`.
Ported runtime changes from `8644d932e8159c63627740b56da4120920403e02`
and `5f4df50a64677d300bdc0ec75ff81623cdf2853c`:

- Batch CUDA deployment EMA interpolation with foreach, retaining CPU behavior.
- Atomically alias an unchanged periodic snapshot on pause or joint completion;
  keep durable-copy fallback and fresh serialization after codec calibration.
- Record data wait and validation duration separately.

This checkout continues the original paired experiment: one GPU per C/D,
512 crops, batch 2, accumulation 4, effective batch 8, original losses,
quantization endpoints, full Kodak24 six-quality actual-rate validation,
and 200k joint budget. All model, loss, data, precision, validation and
training-contract modules remain identical to the base commit, apart from
the checkpoint runtime implementation. Model/optimizer/scaler/EMA/RNG/data
progress restore through the existing strict contract.

The source branch also contains a different low-rate recipe, Router warmup
validation policy and two-rank C launcher. Those changes are outside this
continuation. With both GPUs already running independent models, distributed
validation of one model cannot use the other GPU without coordination.

Acceptance evidence and same-checkpoint speed measurements live under
`/data2/ls/results/cd_runtime_speedup_20261020/`. Migration records are saved
under `/data2/ls/checkpoints/bt_self_flow_v1/runtime_speedup_20261020/`.
