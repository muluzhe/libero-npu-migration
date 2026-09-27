# 验证结果目录

结果按模型分目录保存：

- `xvla/`：X-VLA 多 seed 结果和视频
- `openvla/`：OpenVLA 结果和视频
- `pi05/`：PI0.5 结果、历史归档和视频
- `common/`：跨模型汇总、协议和工具产物

> 当前证据说明：原始结果 JSON 与 NPU 日志是统计依据；视频目前仅保存 OpenVLA object 独立复验 10 段（5/10），不混入 object 全量统计。用户反馈历史 X-VLA/OpenVLA 视频整理丢失；目前没有保存或可恢复的删除证明。后续所有验证必须在同一次实际 NPU 闭环 rollout 中保存视频，才能作为视频证据。
>
> 论文参考值与本项目原始统计分开记录；未核实的来源不作为已确认事实。

每个模型目录建议使用以下结构：

```text
<model>/
├── <suite>_<run>_results.json
├── logs/
└── videos/<suite>/taskXX_epYY_successZ.mp4
```

视频必须来自模型实际运行在 Ascend NPU 上的闭环 rollout。JSON、日志和视频文件名应能互相定位。根目录下的 `openvla_full/` 是当前/历史全量验证目录，任务运行期间不得移动；任务结束后可归档到 `openvla/`，并保留兼容入口或索引说明。
