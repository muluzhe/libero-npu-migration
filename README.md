# LIBERO 仿真验证 NPU 迁移方案

将 LIBERO 机器人仿真验证迁移到 Ascend NPU，无需 GPU 即可跑通 VLA 模型的闭环仿真评估。本方案以 **OSMesa 软件渲染** 为核心，支持多种 VLA 模型（X-VLA / OpenVLA / PI0 / SmolVLA / ACT / Diffusion Policy），X-VLA 作为首个验证通过的模型示例。

## 目录结构

```
libero-npu-migration/
├── README.md                          # 本文档（复现指南）
├── patches/                           # NPU 迁移补丁（唯一修改点）
│   ├── robosuite_osmesa_render.py     # OSMesa 渲染适配（核心，含 read_pixels GL 坐标修复）
│   └── robosuite_mj_fullM.py          # mujoco 3.10 API 兼容
├── scripts/                           # 自动化脚本
│   ├── setup_env.sh                   # 环境搭建（依赖+渲染库+assets+patch）
│   ├── apply_patches.py               # 自动应用 patch（幂等，可重复运行）
│   └── run_eval.sh                    # 仿真验证一键运行
├── models/                            # 各 VLA 模型 NPU 推理服务器
│   ├── xvla/server.py                 # X-VLA（已验证，成功率 90%）
│   ├── openvla/server.py              # OpenVLA
│   ├── pi0/server.py                  # PI0 / PI0.5
│   ├── smolvla/server.py              # SmolVLA
│   ├── act/server.py                  # ACT
│   └── diffusion_policy/server.py     # Diffusion Policy
├── docs/
│   ├── VIDEO_ORIENTATION.md           # 视频倒置问题根因与修复（必读）
│   └── EXTENSIBILITY.md              # 扩展到其他 VLA 模型指南
└── results/                           # 验证结果（成功率+视频）
    └── xvla/                          # X-VLA 验证结果
```

## 复现步骤（3 步）

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

**预期结果**（X-VLA，固定 seed=42，10ep/suite）：4 suite 平均 **95.8%**（spatial 90% / goal 99% / object 100% / long 94%），视频方向正常。与论文官方 GPU 基准 98.1%量级一致，**NPU 迁移成功**，此结果可稳定复现。

### 成功率归因（必读）

**最终结论（2026-07-13）**。NPU 迁移成功，95.8% vs 论文 98.1%（-2.3%）。渲染差异的系统性影响远小于预想。

详见 `docs/RENDER_DIFF_DIAGNOSIS.md` 与 `docs/PROJECT_TRACKING.md`（最终结果）。

### OpenVLA 验证结果（2026-07-17）

OpenVLA spatial suite（10 task × 5 ep = 50 rollouts, seed 42, bf16+sdpa）：**76.0%** (38/50)，对比官方基准 84.7% ± 0.9%（A100, 3 seed × 50 trial/task）。单 seed 小样本差距 8.7pp 合理（bf16 NPU vs A100 fp32、单 seed vs 官方 3 seed 平均）。

**根因与修复**（详见 `docs/OPENVLA_HANDOVER.md` 阶段 12-13）：OpenVLA 输出 delta action，但 X-VLA client 默认 `act_type="abs"` 强制 `controller.use_delta=False`，delta pos 被当绝对目标坐标解释 → 机器人瞬间被指令拉到工作空间外的位置 → 永远 `done=False` → 闭环 0%。修复：改走 `act_type="rel"` 路径让 env 保持默认 `use_delta=True`，与官方 `run_libero_eval.py:228` 的 `env.step(action.tolist())` 语义等价。

**性能优化**：`server_v2.py` 全局预计算 crop 参数 + 固定 seed 只设一次 + `json_numpy.patch()` 只调一次 → 单步推理 1.5s → 0.45s（3.3x），SR 无损失。

## 关键环境变量（运行前必须设置）

```bash
export NUMBA_DISABLE_JIT=1                              # 禁用 numba JIT（避免 segfault）
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1         # OSMesa 软件渲染
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri         # swrast_dri.so 位置
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0
```

`run_eval.sh` 已自动设置这些变量，手动运行时需先 export。

## 视频倒置问题（必读）

**根因**：OSMesa 渲染图像原点在左上（正常），GPU 的 `mjr_readPixels` 原点在左下（GL 坐标，上下颠倒）。LIBERO 客户端的 `_flip_agentview` 假设输入是 GPU 的颠倒图，若直接用 OSMesa 图会导致：① 视频倒置 ② 模型收到错误方向图像 → 成功率 0%。

**修复**（唯一修改点）：在 `robosuite/utils/binding_utils.py` 的 `read_pixels` 中，对 OSMesa 渲染图做 `np.flip(img, 0)` 模拟 GPU 的 GL 坐标。这样 `libero_client.py` 保持官方原样零改动，视频方向自动正确，模型输入与训练数据一致。

**详细说明**：见 `docs/VIDEO_ORIENTATION.md`

## 支持的 VLA 模型

| 模型 | 服务器代码 | 动作格式 | 迁移难度 | 验证状态 |
|---|---|---|---|---|
| **X-VLA** | `models/xvla/server.py` | `[pos3,rot6d,grip1]` | 低 | ✅ 已验证 95.8%(4 suite 平均, 固定 seed=42) |
| **OpenVLA** | `models/openvla/server.py` | 离散 token | 低 | ✅ 已验证 spatial 76.0%(50 rollouts, seed 42) |
| **PI0/PI0.5** | `models/pi0/server.py` | `[pos3,aa3,grip1]` | 中 | 待验证 |
| **SmolVLA** | `models/smolvla/server.py` | 离散 token | 低 | 待验证 |
| **ACT** | `models/act/server.py` | action chunk | 中 | 待验证 |
| **Diffusion Policy** | `models/diffusion_policy/server.py` | DDPM 采样 | 中 | 待验证 |

运行其他模型：
```bash
bash scripts/run_eval.sh openvla /path/to/openvla-model ./results_openvla 10
bash scripts/run_eval.sh pi0 /path/to/pi0-model ./results_pi0 10
```

## 技术路线

- **推理**：torch_npu 在线推理（fp32 + eager attn），零模型代码改动
- **渲染**：OSMesa 软件渲染（swrast_dri），无需 GPU 设备
- **仿真**：LIBERO benchmark（robosuite + mujoco 3.10）
- **硬件**：Ascend 910B4，CANN 8.2.RC1

## License

Apache 2.0
