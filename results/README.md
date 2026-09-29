# 验证结果目录

公开仓库仅保存无个人路径的统计汇总和历史已发布结果；本机原始 JSON、推理日志及视频留在被 `.gitignore` 排除的运行目录。新 run 的布局见 `docs/EVAL_PROTOCOL.md` 第三节：

- `xvla/`：X-VLA 多 seed 结果和视频
- `openvla_full/`：OpenVLA 四套件全量结果（spatial 77/100、goal 79/100、object 75/100、libero_10 57/100）
- `pi05/summary_20260927.json`：PI0.5 四套件公开汇总，386/400（96.5%）；本机 `pi05/full_libero/20260927_214309_3556587/` 保存原始记录与 400 段解码核验视频。
- `g05/summary_20260928.json`：G0.5 四套件公开汇总，1977/2000（98.85%）；本机 `g05/full_libero/20260928_122503_1236126/` 保存原始记录与 2000 段视频。

> 当前证据说明：原始结果 JSON 与 NPU 日志是统计依据。PI0.5 四套件全量（`pi05/full_libero/20260927_214309_3556587/`）的 400 段视频已完成解码核验；G0.5 独立只读核验确认四个 suite 各有 500 个非空 MP4（2000/2000）；`ffprobe` 视频流与正时长全部合格，`ffmpeg` 每段首帧实际解码均成功，各 suite 抽首/中/尾三段多帧解码共 12/12 成功。MP4 success/failure 尾缀与原始 JSON 按 suite 统计完全一致：spatial 496/4、object 500/0、goal 490/10、libero_10 491/9，合计 1977/2000（98.85%）。未对每部视频完整逐帧解码。历史 X-VLA/OpenVLA 视频证据情况见各专项文档。所有验证必须在同一次实际 NPU 闭环 rollout 中保存视频，才能作为视频证据。
>
> 论文参考值与本项目原始统计分开记录；未核实的来源不作为已确认事实。统一评测协议与目录结构见 `docs/EVAL_PROTOCOL.md`。

新 run 的模型目录建议使用以下结构：

```text
<model>/
├── <suite>_<run>_results.json
├── logs/
└── videos/<suite>/taskXX_epYY_successZ.mp4
```

视频必须来自模型实际运行在 Ascend NPU 上的闭环 rollout。JSON、日志和视频文件名应能互相定位。`openvla_full/` 是 OpenVLA 的历史结果目录；新 run 可按统一目标布局使用 `results/<model>/full_libero/<run>/`，不据此推断历史结果均采用该布局。
