# LIBERO NPU 验证统一流程与参数规范

所有模型（X-VLA / OpenVLA / PI0.5 / G0.5 / 后续新模型）的 LIBERO 闭环验证遵循本规范。默认参数定义在 [`scripts/eval_defaults.sh`](../scripts/eval_defaults.sh)（编排脚本 source 它），环境变量定义在 [`scripts/common_env.sh`](../scripts/common_env.sh)。

## 一、统一评测协议参数

| 参数 | 默认值 | 依据 |
|---|---|---|
| 随机种子 | 42 | 项目统一，保证可复现 |
| trials/task | **50**（项目推荐/对照官方的默认全量）/ 10（缩减） | LIBERO 官方 benchmark 每 task 50 个初始状态；PI0.5/OpenVLA 现有结果使用过 10，缩减验证必须在 manifest 与结果 JSON 中标注 |
| horizon 上限 | spatial 220 / object 280 / goal 300 / libero_10 520 | G0.5 官方 evaluator 固定值，脚本默认固定，**勿改** |
| 相机分辨率 | 256x256 | 官方评测默认 |
| 初始静置步 | 20 | G0.5 官方 `--num_steps_wait 20` |
| 并行环境数 | 10（资源受限可降） | G0.5 官方默认；影响吞吐不影响协议 |
| 视频输出 | **强制开启** | 项目硬性要求：视频必须来自模型实际 NPU 闭环 rollout |
| 成功判定 | LIBERO `env.check_success()` | 官方定义 |

**模型固有参数不统一**（各模型论文/训练分布决定，在 manifest 中记录）：动作类型（X-VLA=abs，OpenVLA/PI0.5=rel，G0.5=ActionCodec 20D）、action chunk 步数（PI0.5/G0.5=10，OpenVLA/X-VLA=1）、推理精度（OpenVLA=bf16，PI0.5/G0.5=fp32）、server 协议（HTTP+json_numpy 或 WebSocket+msgpack）。

## 二、统一验证流程（两阶段）

### 阶段 1：冒烟（每个模型/每次适配变更后必做）

1. 启动该模型 NPU server（单卡，`ASCEND_RT_VISIBLE_DEVICES` 指定空闲卡）。
2. 单 suite（建议 libero_spatial）单 task × 1~5 trials，固定 seed。
3. 检查：闭环能跑、动作语义正确（成功率合理）、**视频生成且解码核验通过**（帧数/尺寸/非静态）。
4. 产物存 `results/<model>/smoke_<date>/`。冒烟不通过不得进入全量。

### 阶段 2：全量（四 suite 串行）

1. 编排脚本 `scripts/<model>/run_<model>_full_libero.sh`：server 全 suite 复用（或每 suite 独立，模型有 suite 专属 checkpoint 时），四 suite 串行评估。
2. 参数：`eval_defaults.sh` 默认值（50 trials/task，项目推荐/对照官方的默认全量；缩减值须标注）、seed 42、强制视频。
3. 中途失败按评估器实际能力处理；G0.5 官方 evaluator 不支持 `--resume`，需按 suite 重跑，不得伪造数据。
4. 完成后生成 summary（四 suite 成功率 + 总平均）。

## 三、统一目录与产物结构

```text
scripts/<model>/run_<model>_full_libero.sh   # 编排器（统一命名）
scripts/<model>/eval_*.py                    # 评估器
models/<model>/server*.py 或 README.md       # NPU 推理服务（见 models/README.md）
results/<model>/full_libero/<YYYYmmdd_HHMMSS_PID>/
├── manifest.json          # 完整配置：模型、checkpoint、设备、精度、seed、trials、horizon、视频开关
├── server.log             # NPU 推理日志（含 torch_npu/Ascend 运行证据）
├── server.pid / server.port
└── <suite>/
    ├── client.log（或 eval.log）
    ├── <suite>_results.json  # 原始统计：per-task successes/total、success_rate
    └── videos/*.mp4          # 每 trial 一段，文件名含 task/episode/success 标记
results/<model>/smoke_<date>/                # 冒烟产物（同上子集）
```

## 四、结果记录与对比规范

1. **NPU 原始统计**与**论文/官方参考值**分开记录，不混写；参考值注明来源与协议。
2. 协议差异（缩减 trials、精度、后端替换）如实标注；不作"统计等价""2σ 内达标""逐字节一致"等未经验证的断言。
3. 视频证据：G0.5 官方 JSON 为 per-task 汇总，不逐条记录 video_path 或核验状态；按视频文件名与 suite 成功/失败总数交叉核验。视频核验采用全量视频首帧解码，并对 12 段视频做多帧抽样，不声称全片逐帧核验。
4. 每模型专项文档（`docs/<model>/`）记录适配过程与结果；总过程记录 `docs/PROJECT_TRACKING.md`。

## 五、当前各模型执行状态对照

| 模型 | 冒烟 | 全量 | 协议 | 结果位置 |
|---|---|---|---|---|
| OpenVLA | 通过 | 完成 | 10 ep/task（缩减，历史结果协议） | `results/openvla_full/`（历史结果目录） |
| PI0.5 | 通过 | 完成 | 10 ep/task（缩减）、seed42、220/280/300/520 | `results/pi05/full_libero/20260927_214309_3556587/` |
| G0.5 | 通过 | 完成 | **50 trials/task（项目推荐/对照官方的默认全量）**、seed42、fp32、action_steps10/chunk10、220/280/300/520 | `results/g05/full_libero/20260928_122503_1236126/` |
| X-VLA | 历史 | 历史 | 多 seed，见专项文档 | `results/xvla/` |
