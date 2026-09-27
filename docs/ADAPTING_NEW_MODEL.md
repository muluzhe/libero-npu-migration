# 新模型适配指南

项目把模型专用内容放在 `scripts/<model>/`、`docs/<model>/` 和 `results/<model>/`。模型脚本和文档只维护在对应子目录中，命令与链接均应使用子目录实际路径。

## 适配前检查

1. 明确 checkpoint、训练框架、输入图像 dtype/range、state schema、姿态四元数顺序。
2. 读取 checkpoint 的 norm_stats，逐维检查 LIBERO 输入是否落在训练分布内。
3. 明确动作是 absolute 还是 delta，动作单位、旋转表示、夹爪开合语义和 chunk 长度。
4. 先用单 task 单 episode 验证，再扩展到 suite；每个 rollout 必须保存视频。

## 推荐文件布局

```text
models/<model>/server_v2.py       # 推理 server
scripts/<model>/eval_<model>.py  # NPU 闭环评估，保存 JSON、日志、mp4
scripts/<model>/run_<model>.sh   # 环境与 server/eval 编排
results/<model>/                  # 结果、日志、videos/
docs/<model>/                     # 最多两个专项记录
```

通用环境变量可直接使用 `source scripts/common_env.sh`。server 适配骨架见 `scripts/model_server_template.py`。

## 视频要求

评估脚本默认保存 agentview mp4，文件名应包含 suite/task/episode/success。视频必须在模型实际运行于 Ascend NPU 的同一次闭环 rollout 中采集，不能用静态帧或事后生成视频替代。推荐参数：

```bash
python scripts/<model>/eval_<model>.py \
  --video_dir results/<model>/videos/<suite>
```

## 结果对比

结果 JSON 必须记录：模型、checkpoint、NPU device、dtype、seed、suite、episodes/task、horizon、success rate、论文/官方基准和结果视频目录。论文基准要注明来源、评估协议差异、样本数和单 seed/多 seed 情况，不能直接把不同协议的数字当成同一指标。
