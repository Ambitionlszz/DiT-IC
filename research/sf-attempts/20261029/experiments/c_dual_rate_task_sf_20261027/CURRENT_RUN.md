新 C + dual-rate task SF 已完成双卡20步短跑、完整保存和从10020恢复后的真实更新，目前持续训练至200K。

代码：29dc767，codex/c-dual-rate-task-sf-20261027。

旧实验完整保存于 /data2/ls/checkpoints/c_sf_weight1_20261026/joint/paused.pt，step21663。

当前配置：两卡各4，积累1，有效8，512输入，主RD 4/1/1 +0.1 base，原生独立六档q，SF外权重在10K后2K步升至1。

日志与恢复：/data2/ls/checkpoints/c_dual_rate_task_sf_20261027。

预检确认梯度边界和teacher读出内容依赖，尚无新增步数Kodak24结果，不能宣称压缩收益。
