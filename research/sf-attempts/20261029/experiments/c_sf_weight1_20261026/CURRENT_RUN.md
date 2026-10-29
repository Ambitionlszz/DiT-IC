# C+SF, fixed SF weight 1

Both GPUs continue C+SF only, with batch 4 per GPU and accumulation 1 (effective batch 8). Pure C is already trained on the user's other server and must not be started here.

The previous run was saved at update 13271. Starting at update 13272, the fixed SF coefficient is 1: L_main + 1 * (whole_batch_sf_loss / 0.5). The previous coefficient was 0.641302792947235. This is a continuation, so 10K–13271 used the previous coefficient.

Raw and deployment EMA tensors, teacher, both optimizers, scheduler, scaler, both rank RNG states, data counters and validation best were compared exactly across the migration. Only the explicit SF weight and output directory changed.

Active pipeline: /data2/ls/checkpoints/c_sf_weight1_20261026/pipeline.json
Training log: /data2/ls/checkpoints/c_sf_weight1_20261026/logs/joint.log
Migration audit: /data2/ls/checkpoints/c_sf_weight1_20261026/weight_change_verification.json

The 12K comparison belongs to the previous coefficient; no fixed-1 validation has completed yet.
