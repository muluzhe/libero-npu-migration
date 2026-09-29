# 新模型适配指南

项目把模型专用内容放在 `scripts/<model>/`、`docs/<model>/` 和 `results/<model>/`。模型脚本和文档只维护在对应子目录中，命令与链接均应使用子目录实际路径。

## 适配前检查

0. 阅读 `docs/EVAL_PROTOCOL.md`（统一流程与参数规范）：冒烟 -> 全量两阶段，默认参数以 `scripts/eval_defaults.sh` 为准。
1. 明确 checkpoint、训练框架、输入图像 dtype/range、state schema、姿态四元数顺序。
2. 读取 checkpoint 的 norm_stats，逐维检查 LIBERO 输入是否落在训练分布内。
3. 明确动作是 absolute 还是 delta，动作单位、旋转表示、夹爪开合语义和 chunk 长度。
4. 先用单 task 单 episode 验证，再扩展到 suite；每个 rollout 必须保存视频。

## 推荐文件布局

```text
models/<model>/server_v2.py       # 推理 server（外部官方代码模式则放 README.md，见 models/README.md）
scripts/<model>/eval_<model>.py  # NPU 闭环评估，保存 JSON、日志、mp4
scripts/<model>/run_<model>_full_libero.sh  # 四 suite 全量编排（统一命名）
results/<model>/full_libero/<run>/<suite>/  # 全量结果、日志、videos/
results/<model>/smoke_<date>/    # 冒烟结果
docs/<model>/                     # 最多两个专项记录
```

通用环境变量 `source scripts/common_env.sh`；**统一评测默认参数 `source scripts/eval_defaults.sh`**（seed 42、50 trials/task 项目推荐/对照官方的默认全量、horizon 220/280/300/520、强制视频，详见 `docs/EVAL_PROTOCOL.md`）。PI0.5/OpenVLA 现有结果中的 10 trials/task 属于缩减协议。server 适配骨架见 `scripts/model_server_template.py`。

## 视频要求

评估脚本默认保存 agentview mp4，文件名应包含 suite/task/episode/success。视频必须在模型实际运行于 Ascend NPU 的同一次闭环 rollout 中采集，不能用静态帧或事后生成视频替代。推荐参数：

```bash
python scripts/<model>/eval_<model>.py \
  --video_dir results/<model>/videos/<suite>
```

## 结果对比

结果 JSON 必须记录：模型、checkpoint、NPU device、dtype、seed、suite、trials/task、horizon、success rate、论文/官方基准和结果视频目录。论文基准要注明来源、评估协议差异（项目推荐/对照官方的默认全量 50 vs 缩减 10 trials、精度、后端）、样本数和单 seed/多 seed 情况，不能直接把不同协议的数字当成同一指标。目录结构与产物要求见 `docs/EVAL_PROTOCOL.md` 第三节。
