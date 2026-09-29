# LIBERO NPU 迁移与 VLA 闭环评估

本项目将 LIBERO 仿真评估迁移到 Ascend NPU，使用 OSMesa 软件渲染，在无 GPU 渲染设备的条件下运行 VLA 模型闭环验证。项目同时保留模型适配补丁、推理服务、评估脚本、原始结果索引和迁移过程记录。

## 当前四模型状态

以下数字是本项目当前保存的 NPU 原始统计，模型顺序与 suite 顺序均明确列出；它们与论文或官方 CUDA/bf16 参考值分开记录，不作协议完全一致或统计等价断言。

| 模型 | suite 结果 | 当前状态 |
|---|---|---|
| X-VLA | spatial 90/100、goal 99/100、object 100/100、libero_10 94/100；平均 95.75% | 已有历史四套件记录；多 seed 结果需结合相同 init_seed 限制解读 |
| OpenVLA | spatial 77/100、goal 79/100、object 75/100、libero_10 57/100 | 四套件原始 JSON 已保存；视频证据按实际运行产物记录 |
| PI0.5 | spatial 96/100、object 100/100、goal 97/100、libero_10 93/100；合计 386/400（96.5%） | 四套件缩减协议已完成；同一 checkpoint、seed42、10 episodes/task |
| G0.5 | spatial 496/500、object 500/500、goal 490/500、libero_10 491/500；合计 1977/2000（98.85%，一位小数 98.9%） | 官方 50 trials/task 协议已完成；四套件各 500 个非空视频，首帧 2000/2000 可解码，抽样多帧 12/12 通过 |

G0.5 的视频核验覆盖视频流、正时长、每段首帧实际解码，以及每套件首/中/尾三段的多帧抽样；没有对每部视频完整逐帧解码。PI0.5 的 400 段视频已按评估脚本逐段解码核验。公开仓库只提供脱敏汇总，原始 JSON、日志和视频留在本机 `results/` 下。

## 从哪里开始

- 评估协议：[`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)
- 新模型适配：[`docs/ADAPTING_NEW_MODEL.md`](docs/ADAPTING_NEW_MODEL.md)
- 模型入口索引：[`models/README.md`](models/README.md)
- 项目总结、技术决策和路径边界：[`docs/LIBERO_NPU_MIGRATION.md`](docs/LIBERO_NPU_MIGRATION.md)
- PI0.5 过程与问题记录：[`docs/pi05/PI05_TRACKING.md`](docs/pi05/PI05_TRACKING.md)、[`docs/pi05/PI05_RECORD.md`](docs/pi05/PI05_RECORD.md)
- G0.5 资料核对与迁移记录：[`docs/g05/G05_TRACKING.md`](docs/g05/G05_TRACKING.md)
- G0.5 NPU 补丁与启动说明：[`models/g05/README.md`](models/g05/README.md)、[`models/g05/g05_npu.patch`](models/g05/g05_npu.patch)
- 开源发布前检查：[`docs/OPEN_SOURCE_CHECKLIST.md`](docs/OPEN_SOURCE_CHECKLIST.md)
- 变更记录：[`CHANGELOG.md`](CHANGELOG.md)

## 项目目录

```text
libero-npu-migration/
├── README.md
├── CHANGELOG.md
├── docs/
│   ├── EVAL_PROTOCOL.md              # 评估参数、命名和证据边界
│   ├── ADAPTING_NEW_MODEL.md         # 新模型适配步骤
│   ├── LIBERO_NPU_MIGRATION.md      # 项目总结与技术路线
│   ├── OPEN_SOURCE_CHECKLIST.md      # 发布前检查
│   ├── PROJECT_TRACKING.md           # 全项目历史过程
│   ├── xvla/                         # X-VLA 专项记录
│   ├── openvla/                      # OpenVLA 专项记录
│   ├── pi05/                         # PI0.5 过程与问题记录
│   └── g05/                          # G0.5 资料与迁移记录
├── models/
│   ├── README.md                     # 模型服务器索引
│   ├── xvla/server.py
│   ├── openvla/server_v2.py
│   ├── pi0/server_v2.py              # PI0.5
│   ├── g05/                          # G0.5 补丁和说明
│   ├── smolvla/server.py             # 骨架
│   ├── act/server.py                 # 骨架
│   └── diffusion_policy/server.py    # 骨架
├── patches/                          # OSMesa 与 MuJoCo 适配补丁
├── scripts/
│   ├── setup_env.sh
│   ├── apply_patches.py
│   ├── eval_defaults.sh
│   ├── run_eval.sh                  # X-VLA 通用入口
│   ├── openvla/                      # OpenVLA 四套件入口
│   ├── pi05/                         # PI0.5 入口
│   └── g05/run_g05_full_libero.sh    # G0.5 四套件入口
└── results/                          # 原始结果、日志和视频索引
```

项目公开文档不依赖个人机器的绝对路径。运行时请用 `$PROJECT_ROOT`、`$MODEL_PATH`、`$X_VLA_ROOT`、`$OPENVLA_CKPTS`、`$G05_ROOT` 等环境变量替换本机路径；模型权重、外部源码和视频按 `.gitignore` 规则管理。GitHub 仅发布 [`PI0.5` 汇总](results/pi05/summary_20260927.json)和 [`G0.5` 汇总](results/g05/summary_20260928.json)，原始结果、日志与视频仍保留在本机对应运行目录。

## 环境与运行

前置条件通常包括 Ascend 910B/910B4、匹配的 CANN 与 torch_npu、Python/conda 环境、LIBERO assets，以及 OSMesa 软件渲染库。先在项目根目录设置环境：

```bash
export PROJECT_ROOT=/path/to/libero-npu-migration
cd "$PROJECT_ROOT"
bash scripts/setup_env.sh
```

手动运行或排查渲染时，需要在 Python 启动前设置 OSMesa 和 NPU 环境：

```bash
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export LIBGL_DRIVERS_PATH="$HOME/render_libs/dri"
export LD_LIBRARY_PATH="$HOME/render_libs:/usr/lib64:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0
```

各模型使用专用入口，动作语义和 checkpoint 不能混用：

```bash
# X-VLA：快速检查或四 suite 运行
bash scripts/run_eval.sh xvla "$MODEL_PATH" "$PROJECT_ROOT/results/xvla_local" 1
bash scripts/run_eval.sh xvla "$MODEL_PATH" "$PROJECT_ROOT/results/xvla_local" 10

# OpenVLA：使用 suite 专用编排器和各 suite checkpoint
bash scripts/openvla/run_openvla_full_validation.sh 0 8011 libero_spatial libero_goal

# PI0.5：使用独立环境和专用编排器
bash scripts/pi05/run_pi05_full_libero.sh

# G0.5：在独立 g05 环境中按官方 50 trials/task 协议运行
bash scripts/g05/run_g05_full_libero.sh
```

运行前应先阅读对应模型文档，确认 checkpoint、端口、NPU 卡、seed、horizon 和输出目录；完整协议约定见 [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)。查看结果时直接读取对应 `results/<model>/` 下的原始 JSON 和 manifest，不用 README 中的示例数字替代实际产物。

## 技术边界

OSMesa 适配位于渲染层，包含 GL 坐标处理；客户端和模型服务通过 HTTP 或 WebSocket 对接，动作格式由模型决定。PI0.5 使用 lerobot 的输入处理契约；G0.5 使用官方 WebSocket/msgpack 服务和 ActionCodec，并通过补丁将图像 patch 变换拆为 Ascend 可接受的张量维度，同时使用纯 PyTorch/SDPA 回退。G0.5 的 NPU 结果使用 fp32；官方 CUDA/bf16 参考值只作为公开参考。

本项目只说明已保存证据支持的结论。历史过程文档中的旧结果、旧根因假设和旧路径均保留用于复盘，并在文档顶部或相应章节标注为历史快照。

## License

Apache-2.0；第三方模型、数据集和上游代码仍以各自许可证及使用条件为准。
