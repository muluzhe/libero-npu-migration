# LIBERO 仿真验证 NPU 迁移方案

将 LIBERO 机器人仿真验证迁移到 Ascend NPU，无需 GPU 即可跑通 VLA 模型的闭环仿真评估。本方案以 **OSMesa 软件渲染** 为核心，支持多种 VLA 模型（X-VLA / OpenVLA / PI0.5 / SmolVLA / ACT / Diffusion Policy）。当前有 NPU 日志证据的结果为 X-VLA、OpenVLA 与 PI0.5；评估支持按 episode 保存视频，视频范围以各次运行实际产物为准。

**文档结构**（每模型 ≤2 文档 + 总过程记录 + 总结文档，详见 [docs/LIBERO_NPU_MIGRATION.md](docs/LIBERO_NPU_MIGRATION.md) §0）：
复现看本 README → 全过程看 [PROJECT_TRACKING.md](docs/PROJECT_TRACKING.md) → 总结看 [LIBERO_NPU_MIGRATION.md](docs/LIBERO_NPU_MIGRATION.md) → 各模型细节看 [RENDER_DIFF_DIAGNOSIS.md](docs/xvla/RENDER_DIFF_DIAGNOSIS.md)（X-VLA）/ [OPENVLA_HANDOVER.md](docs/openvla/OPENVLA_HANDOVER.md)（OpenVLA）/ [PI05_TRACKING.md](docs/pi05/PI05_TRACKING.md) + [PI05_RECORD.md](docs/pi05/PI05_RECORD.md)（PI0.5）。

## 目录结构

```
libero-npu-migration/
├── README.md                          # 本文档（复现指南）
├── CHANGELOG.md                       # 变更日志
├── patches/                           # NPU 迁移补丁（唯一修改点）
│   ├── robosuite_osmesa_render.py     # OSMesa 渲染适配（核心，含 read_pixels GL 坐标修复）
│   └── robosuite_mj_fullM.py          # mujoco 3.10 API 兼容
├── scripts/                           # 自动化脚本（按模型分类）
│   ├── setup_env.sh                   # 环境搭建（依赖+渲染库+assets+patch）
│   ├── apply_patches.py               # 自动应用 patch（幂等，可重复运行）
│   ├── run_eval.sh                    # X-VLA 一键验证（openvla/pi0 见专用脚本）
│   ├── openvla/run_openvla_full_validation.sh # OpenVLA 4 suite 全量验证编排器
│   ├── openvla/eval_openvla_suite.py      # OpenVLA 任意 suite 验证脚本
│   ├── pi05/eval_pi05_spatial.py + run_pi05_spatial.sh  # PI0.5 spatial 验证
│   ├── gpu_infer_compare.py           # PI0.5 NPU/GPU 推理输出方向对比（双端同脚本）
│   └── ...                            # 诊断/验证脚本（头部有弃用标注）
├── models/                            # 各 VLA 模型 NPU 推理服务器
│   ├── xvla/server.py                 # X-VLA（四套件原始统计平均 95.75%）
│   ├── openvla/server_v2.py           # OpenVLA（已验证 spatial 76.0%；server.py 为弃用 v1）
│   ├── pi0/server_v2.py               # PI0.5（推理+闭环跑通，0% 根因诊断中；server.py 为弃用骨架）
│   ├── smolvla/server.py              # SmolVLA（骨架）
│   ├── act/server.py                  # ACT（骨架）
│   └── diffusion_policy/server.py     # Diffusion Policy（骨架）
├── docs/                              # 总结、总过程和各模型专项文档
│   ├── openvla/                       # OpenVLA 专项文档
│   ├── pi05/                          # PI0.5 专项文档
│   ├── xvla/                          # X-VLA 专项文档
│   ├── ADAPTING_NEW_MODEL.md          # 新模型适配指南
│   └── ...                            # 根目录公共文档
└── results/                           # 验证结果（results/README.md）
    ├── xvla/                          # X-VLA 多 seed 结果与视频
    ├── openvla/                       # OpenVLA 结果与视频
    ├── pi05/                          # PI0.5 结果、历史归档与视频
    ├── openvla_full/                  # 当前/历史全量验证目录，任务运行时不移动
    └── common/                        # 跨模型汇总和公共产物
```

## 复现步骤（3 步，以 X-VLA 为例）

### 前置条件
- 硬件：Ascend 910B（snt9b1）NPU 服务器
- 镜像：`pytorch_ascend:pytorch_2.7.1-cann_8.2.rc1-py_3.12-euler_2.10.11-aarch64`（HCE 2.0, glibc 2.34）
- 已下载 X-VLA 模型权重到 `$MODEL_PATH`
- 已 clone X-VLA 代码到 `$X_VLA_ROOT`

### 第 1 步：环境搭建

```bash
cd $PROJECT_ROOT
bash scripts/setup_env.sh
```

该脚本自动完成（约 15 分钟）：
1. 安装 Python 依赖（mujoco/robosuite/libero/transformers 等）
2. 从 HCE 2.0 官方仓库下载 OSMesa 渲染库（`libOSMesa` + `swrast_dri` + `libLLVM-12`，无需 root）
3. 下载 LIBERO mujoco assets（从 Hugging Face 官方仓库）
4. 自动应用 patch（见 `scripts/apply_patches.py`）

### 第 2 步：运行仿真验证

```bash
# 完整 4 suite × 10 episodes（约 4-5 小时）
bash scripts/run_eval.sh xvla $MODEL_PATH ./libero_eval_results 10

# 快速验证（1 suite × 1 episode，约 6 分钟）
bash scripts/run_eval.sh xvla $MODEL_PATH ./libero_quick_test 1
```

### 第 3 步：查看结果

```bash
# 成功率
cat ./libero_eval_results/*/results.json

# 视频文件
ls ./libero_eval_results/*/*.mp4
```

**已记录结果**（X-VLA，init_seed=42，4 套件各 10 ep）：spatial 90% / goal 99% / object 100% / long 94%，平均 **95.75%**。这是原始统计结果；论文参考值及协议来源另列，尚待逐项核实。详见 [RENDER_DIFF_DIAGNOSIS.md](docs/xvla/RENDER_DIFF_DIAGNOSIS.md)。

**OpenVLA 验证（专用路径）**

OpenVLA 是 delta action 模型，不能走 `run_eval.sh`（动作语义不同），用专用编排器：

```bash
# 4 suite 全量验证（自动起停 server，双卡可并行两条流，默认断点续跑）
bash scripts/openvla/run_openvla_full_validation.sh 0 8011 libero_spatial libero_goal
bash scripts/openvla/run_openvla_full_validation.sh 1 8021 libero_object libero_10
# checkpoint 放 $OPENVLA_CKPTS/libero-{spatial,object,goal,10}/（官方微调版）
```

**已记录结果**：spatial 77/100（77.0%）、goal 79/100（79.0%）、object 75/100（75.0%）、libero_10 57/100（57.0%），均为 seed42、每 task 10 ep 的 NPU 结果。另有 object 独立视频复验 10 段，成功 5/10；该样本不混入 object 全量统计。论文/官方参考值与原始统计分开记录，来源待核实。详见 [OPENVLA_HANDOVER.md](docs/openvla/OPENVLA_HANDOVER.md) §8。

## PI0.5 验证（专用路径）

PI0.5 是 lerobot 框架的 delta chunk 模型，独立 conda env + 专用脚本（P4.29 修复版，输入构造契约见 [LIBERO_NPU_MIGRATION.md](docs/LIBERO_NPU_MIGRATION.md) §4.4）：

```bash
bash scripts/pi05/run_pi05_spatial.sh   # 自动起 fp32 server + 跑 spatial 10 task × 10 ep，默认保存 results/pi05/videos/
```

## 关键环境变量（运行前必须设置）

```bash
export NUMBA_DISABLE_JIT=1                              # 禁用 numba JIT（避免 segfault）
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1         # OSMesa 软件渲染
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri         # swrast_dri.so 位置
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0
```

`run_eval.sh` / `run_openvla_full_validation.sh` 已自动设置这些变量，手动运行时需先 export。**后台（nohup/setsid）跑 NPU 任务时，启动脚本必须先 source Ascend 环境**（`/usr/local/Ascend/ascend-toolkit/set_env.sh` + 驱动库路径），否则 `torch.npu.is_available()=False`（详见 [PI05_RECORD.md](docs/pi05/PI05_RECORD.md) 教训#33）。

## 视频倒置问题（必读）

根因是 GPU 与 OSMesa 的渲染坐标系差异（GL 原点左下 vs 左上），修复是在 `read_pixels` 做 `np.flip(img, 0)` 模拟 GL 坐标——`libero_client.py` 保持官方原样零改动。完整根因分析、三方案对比与验证方法见 [LIBERO_NPU_MIGRATION.md](docs/LIBERO_NPU_MIGRATION.md) §6。

## 支持的 VLA 模型

| 模型 | 服务器代码 | 动作格式 | 验证状态 |
|---|---|---|---|
| **X-VLA** | `models/xvla/server.py` | `[pos3,rot6d,grip1]` 绝对 | 已记录 4 套件结果：90% / 99% / 100% / 94%，平均 95.75%；历史多 seed 统计的独立性需谨慎解释 |
| **OpenVLA** | `models/openvla/server_v2.py` | `[delta_pos3,aa3,grip1]` | 已记录：spatial 77%、goal 79%、object 75%、libero_10 57%；另有 object 10 段视频复验 5/10 |
| **PI0.5** | `models/pi0/server_v2.py` | `[delta_pos3,aa3,grip1]` chunk | 旧 run 已记录 spatial 96.0%（96/100）；新 run 正在 NPU0 fp32 进行，当前 4/100（task0 已完成 4/10，4/4 成功） |
| **SmolVLA** | `models/smolvla/server.py` | 离散 token | 待验证（骨架） |
| **ACT** | `models/act/server.py` | action chunk | 待验证（骨架） |
| **Diffusion Policy** | `models/diffusion_policy/server.py` | DDPM 采样 | 待验证（骨架） |

新模型迁移指南（含各模型差异点与一键运行支持边界）见 [LIBERO_NPU_MIGRATION.md](docs/LIBERO_NPU_MIGRATION.md) §7；评估协议约定（episodes/seed/步数上限/success 判定）见同文档 §5。

## 技术路线

- **推理**：torch_npu 在线推理（X-VLA fp32+eager；OpenVLA bf16+sdpa 最优，fa2/INT8 NPU 不支持）
- **渲染**：OSMesa 软件渲染（swrast_dri），无需 GPU 设备
- **仿真**：LIBERO benchmark（robosuite + mujoco 3.10）
- **硬件**：Ascend 910B4 ×2，CANN 8.5.2

## License

Apache 2.0
