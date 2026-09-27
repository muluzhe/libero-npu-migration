# LIBERO NPU 迁移项目说明文档（总结文档）

> **项目目标**: 将 VLA（Vision-Language-Action）模型在 Ascend 910B4 NPU 上完成 LIBERO 仿真闭环验证
> **启动时间**: 2026-07-02 ｜ **最后更新**: 2026-09-26（下午 P4.29）
> **硬件**: Ascend 910B4 NPU ×2（bf16 原生推理，OSMesa 软件渲染，无 GPU）
> **最新纠正（2026-09-27）**：本文当前摘要以原始 JSON 和日志为准。X-VLA 四套件为 90% / 99% / 100% / 94%，平均 95.75%；OpenVLA 四套件为 spatial/goal/object/libero_10 = 77/79/75/57；PI0.5 旧 spatial run 为 96/100，当前新 spatial run 正在 NPU0 fp32 上进行。论文参考值待核实，不把结果表述为协议完全对齐或统计等价。视频按实际 episode 运行产物记录，X/OpenVLA 视频暂不展开。
>
> **状态**: X-VLA 与 OpenVLA 已有上述原始统计；PI0.5 旧 spatial run 有 JSON/log 证据，新 spatial run 正在进行并按 episode 保存视频。

---

## 0. 文档结构（2026-09-26 整合后）

| 文档 | 类别 | 内容 |
|---|---|---|
| `README.md` | 入口 | 复现指南（3 步 + 环境变量 + 预期结果） |
| `docs/PROJECT_TRACKING.md` | **总过程记录** | 全项目逐阶段追踪（X-VLA/OpenVLA/PI0.5 + 项目级问题表） |
| `docs/LIBERO_NPU_MIGRATION.md` | **总结文档** | 本文档（成果 + 架构 + 协议约定 + 扩展指南 + 关键修复） |
| `docs/xvla/RENDER_DIFF_DIAGNOSIS.md` | X-VLA 文档① | OSMesa vs GPU 渲染差异归因（95.8% vs 98.1% 的证据链） |
| `docs/openvla/OPENVLA_HANDOVER.md` | OpenVLA 文档① | OpenVLA 验证记录（根因定位过程 + 避坑 + 全量验证） |
| `docs/pi05/PI05_TRACKING.md` | PI0.5 文档①（过程追踪） | PI0.5 逐阶段追踪（§六.1-§六.20） |
| `docs/pi05/PI05_RECORD.md` | PI0.5 文档②（问题记录） | PI0.5 问题-根因-解决全记录（P4.1-P4.29 + 教训） |
| `docs/OPEN_SOURCE_CHECKLIST.md` | 开源事务 | 发布操作手册 + 发布前检查记录 |

每模型 ≤2 个文档 + 1 个总过程记录 + 1 个总结文档。历史文档（EXTENSIBILITY / VIDEO_ORIENTATION / LIBERO_NPU_RECORD / FINAL_REVIEW / ATOMCODE_HANDOVER / GPU_INFER_README）已合并入上述结构。

---

## 1. 项目概览

| 模型 | 动作类型 | NPU 推理 | 闭环验证 | 官方基准对比 | 状态 |
|---|---|---|---|---|---|
| **X-VLA** | 绝对 action（EEF-6D） | ✅ fp32+eager | 已记录四套件原始统计：90% / 99% / 100% / 94%，平均95.75% | 论文参考待核实 | 已记录 |
| **OpenVLA** | delta action（EEF-7D） | ✅ bf16+sdpa | 已记录 spatial/goal/object/libero_10：77/79/75/57（各100 rollouts） | 论文/官方参考值待核实 | 已有四 suite JSON；视频证据仅 object 独立复验 |
| **PI0.5** | delta action chunk（EEF-7D × chunk） | ✅ fp32 | 旧 run 已记录 spatial 96/100；新 run 正在 NPU0 fp32 进行，当前 4/100（task0 4/10，4/4 成功） | 论文/官方参考值待核实 | 旧 run 有 spatial JSON/log；新 run 进行中 |
| SmolVLA / ACT / Diffusion Policy | 各异 | 骨架 server | 未实测 | — | ⏳ 待验证 |

---

## 2. 核心成果

### 2.1 X-VLA NPU 验证（历史多次运行；当前摘要采用四套件原始统计）

| Suite | X-VLA NPU 原始统计（init_seed=42） | 论文参考（来源待核实） |
|---|---|---|
| libero_spatial | 0.90 | 98.2% |
| libero_goal | 0.99 | 97.8% |
| libero_object | 1.00 | 98.6% |
| libero_10 | 0.94 | 97.6% |
| **平均** | **95.75%** | 待核实 |

**说明**: 当前摘要保留四套件原始统计。历史记录中的多 seed 结果使用相同 `init_seed=42`，不能据此把各次结果当作独立环境样本，也不能据此断言推理随机性已被完全吸收或差距完全归因于 OSMesa；渲染诊断仍需结合独立视频和可复核实验。

### 2.2 OpenVLA NPU 验证（2026-07-17 完成 spatial；2026-09-26 起 4 suite 全量补跑）

| 指标 | 值 |
|---|---|
| spatial SR（首次全量，10ep×10task=100 rollouts, seed42, 官方协议） | **77.0%** (77/100) |
| 外部参考（来源待核实） | 84.7% ± 0.9%（记录中的参考值；不作为已核实结论） |
| 推理配置 | bf16 + sdpa + npu:0，单步 ~0.45s（优化后 3.3x） |
| 4 suite 全量验证记录（10ep×10task×4suite=400 rollouts） | 已有 spatial/goal/object/libero_10 JSON，结果分别为 77/79/75/57；视频按实际运行产物记录，X/OpenVLA 视频暂不展开 |

**当前记录的运行配置**：4 suite 各用对应 checkpoint，记录为 seed42、10 ep/task，步数上限为 spatial 220 / object 280 / goal 300 / long 520。论文参考协议、checkpoint 来源与逐项对齐关系尚待核实，不能写成“完全对齐”或“统计等价”。逐 suite 原始 JSON 见 `results/openvla_full/`；X/OpenVLA 视频暂不展开。

### 2.3 PI0.5 验证（2026-09-26 P4.29 修复后 spatial 原始统计）

| 指标 | 值 |
|---|---|
| spatial SR（10 task × 10 ep = 100 rollouts，seed42，horizon 220） | **96.0%**（96/100） |
| 外部参考（来源待核实） | openpi spatial 98.8% / openpi 30k 均值 96.85% / lerobot repro 97.5% |
| 逐 task | task0-4/6/7/8 = 100%，task5=70%，task9=90% |
| 推理配置 | fp32 + npu:1；总耗时 10359s = 172.65min = 2.8775h，约 103.59s/ep（含推理+仿真） |
| 修复前 | 闭环 0%（100 ep 全跑满 220 步，三个月 11 阶段诊断未破） |

**结论**: P4.29 修复后记录到 NPU spatial 96/100。该数字与外部参考值的协议和来源仍需单独核实；本文不据此宣称统计等价或“2σ 内达标”。

- **闭环 0% 真根因（P4.29）**：输入构造五重缺陷——**主因 F**：图像以 uint8 直传，模型内 float32 转换后 0-255 被 `resize_with_pad_torch` 的 `clamp(-1,1)` 摧毁成近二值图（官方管线在 observation_processor 先 /255）；A：`robot0_eef_quat` 按 wxyz 误解释（实为 xyzw）→ state 姿态 ~10σ OOD；A2：client.proprio 冻结首帧（server 以为机器人从未移动）；B：wrist 视图缺 H+W 翻转；C：1-grip 反转错配（OpenVLA 约定误移植）
- **诊断方法论**：训练分布对照法（server 输入逐维对照 ckpt norm_stats）+ 敏感性测试（四输入扰动均影响输出 → 排除模型/NPU 损坏）——半天收口 3 个月未破的 0%
- **重要修正**：P4.16"模型推理输出错方向"归因错误（实为输入摧毁）；P4.28 NPU 基准作废；GPU 对比（D/G）不再必要（若照旧脚本跑会两端一致误判"模型坏"）
- 旧 run 结果文件：`results/pi05/spatial_100ep_2026-09-26/spatial_results.json`；新 run 目录：`results/pi05/20260927_104705_120708_2242249/`（当前进行中）；历史 0% 证据归档 `results/pi05/spatial_0pct_archive_2026-07/`
- 完整诊断链见 `docs/pi05/PI05_TRACKING.md` §六.20 / `docs/pi05/PI05_RECORD.md` P4.29
- 新模型适配见 `docs/ADAPTING_NEW_MODEL.md`；OpenVLA 实际 NPU 视频样本见 `results/openvla/videos/libero_object/`

---

## 3. 技术架构

### 3.1 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│  LIBERO 仿真环境（OSMesa 软件渲染 + robosuite）              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  OffScreenRenderEnv (256×256, agentview + wrist)      │  │
│  │  env.step(action) ← 7维 [delta_pos3, delta_aa3, grip1]│  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↑ HTTP /act
┌─────────────────────────────────────────────────────────────┐
│  X-VLA Client (libero_client.py，官方原样零改动)             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  ClientModel.step() → action_plan 队列缓存            │  │
│  │  LiberoAbsActionProcessor: rot6d↔aa 同源转换           │  │
│  │  act_type="rel" → env 保持 use_delta=True             │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ HTTP POST
┌─────────────────────────────────────────────────────────────┐
│  VLA 推理服务器（server_v2.py，FastAPI + uvicorn）          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  /act 端点: 图像预处理 → predict_action → 动作后处理  │  │
│  │  设备: npu:0/1 | dtype: bf16/fp32 | attn: sdpa/eager  │  │
│  │  全局预计算 crop 参数 + 固定 seed（性能优化）          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          ↓ NPU 推理
┌─────────────────────────────────────────────────────────────┐
│  Ascend 910B4 NPU（torch_npu，双卡可并行两条验证流）         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  OpenVLA 7B (bf16, ~14GB HBM, sdpa, ~0.45s/步)       │  │
│  │  X-VLA (fp32, eager) / PI0.5 4B (fp32, flow matching) │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 关键文件清单

| 文件 | 说明 | 状态 |
|---|---|---|
| `models/xvla/server.py` | X-VLA NPU 推理服务器（官方 model.run 薄封装） | ✅ |
| `models/openvla/server_v2.py` | OpenVLA NPU 推理服务器（bf16+sdpa，全局预计算优化） | ✅ |
| `models/pi0/server_v2.py` | PI0.5 NPU 推理服务器（适配 lerobot 框架） | ✅ 推理跑通 |
| `scripts/run_eval.sh` | X-VLA 一键验证（openvla/pi0 见专用脚本） | ✅ |
| `scripts/openvla/run_openvla_full_validation.sh` | OpenVLA 4 suite 全量验证编排器（自动起停 server） | ✅ 2026-09-26 |
| `scripts/openvla/eval_openvla_suite.py` | OpenVLA 任意 suite 验证脚本（原 eval_spatial_full.py 泛化） | ✅ |
| `scripts/pi05/eval_pi05_spatial.py` + `scripts/pi05/run_pi05_spatial.sh` | PI0.5 spatial 验证 | ✅ |
| `scripts/pi05/gpu_infer_compare.py` | PI0.5 NPU/GPU 推理输出方向对比（双端同脚本，P4.28 schema 对齐） | ✅ 2026-09-26 |
| `scripts/diag_stage1_official.py` | OpenVLA 官方 action 推理对比诊断 | ✅ |
| `patches/robosuite_osmesa_render.py` | OSMesa 渲染 + GL 坐标修复（唯一修改点） | ✅ |
| `results/xvla_npu_seed*/` | X-VLA 5 seed 结果 | ✅ |
| `results/spatial_full_results.json` | OpenVLA spatial 首次完整结果（76.0%） | ✅ |
| `results/openvla_full/` | OpenVLA 4 suite 全量验证结果 | ✅ |
| `results/pi05/spatial_100ep_2026-09-26/` | PI0.5 旧 spatial 完整结果 | ✅ |

---

## 4. 关键技术决策

### 4.1 NPU 推理配置

| 配置 | X-VLA | OpenVLA | PI0.5 |
|---|---|---|---|
| device | npu:0 | npu:0/1 | npu:0 |
| dtype | fp32 | bf16 | fp32（F4，bf16 精度损方向） |
| attn | eager | **sdpa**（eager 慢 3x，flash_2 NPU 不支持） | eager/sdpa |
| 量化 | 无 | 无（INT8/4bit 是 CUDA 专用） | 无 |

### 4.2 delta action 参考系修复（OpenVLA 0% → 76% 的核心根因）

**根因**: OpenVLA 输出 delta action（`openvla.py:47` docstring 明示 "end-effector deltas"），但 X-VLA client 默认 `act_type="abs"`，强制 `robot.controller.use_delta = False`，把 delta pos 当**绝对目标坐标**解释 → 机器人瞬间被拉到工作空间外 → 永远 `done=False`，闭环 0%。

**修复**: client 走 `act_type="rel"` 路径，env 保持默认 `use_delta=True`，与官方 `run_libero_eval.py:228` 的 `env.step(action.tolist())` 语义完全一致。

### 4.3 性能优化（OpenVLA，3.3x）

单步推理 ~1.5s → ~0.45s：①全局预计算 crop 参数 ②固定 seed 只设一次 ③`json_numpy.patch()` 只调一次 ④图像预处理合并。

### 4.4 PI0.5 的输入构造契约（P4.29 后定稿）

此前 11 阶段诊断误把"server 侧绕过 client 设计"当根因方向（P4.5v2 末步 proprio 回传、P4.8 ori6d perm+signs 均为无效绕过，已全部移除）。P4.29 真根因是**输入构造与训练管线不一致**，修复后确立的 /act 接口契约：

| 契约项 | 约定 | 违反后果（实证） |
|---|---|---|
| 图像 dtype | server 必传 float [0,1]（/255） | uint8 → 模型内 clamp 摧毁成二值图（主因，闭环 0%） |
| 图像方向 | 双视图均 H+W 双翻转（client 翻 agentview，server 补翻 wrist） | wrist 缺翻 → 空间理解错乱 |
| state 姿态 | `robot0_eef_quat` 按 **xyzw** → aa（lerobot env_processor 同源） | wxyz 误解释 → ~10σ OOD |
| proprio 新鲜度 | eval 每步 `client.proprio=None` 强制 fresh | 冻结首帧 → 模型以为机器人未动 |
| 夹爪语义 | π0.5 grip 直传（-1=开/+0.92=合，即 env 语义） | 1-grip 反转 → 永不能抓取 |
| chunk 长度 | 官方 n_action_steps=10 | F1=2 是错误输入时代的止血 |

**架构教训**：①"链路自洽"≠"输入正确"，最终判据是**训练分布对照**（ckpt norm_stats 逐维检查）；②action 语义（abs/delta）+ proprio 回传策略 + 图像 dtype/方向 + 姿态约定都应成为 /act 接口契约的一部分（详见 `PI05_RECORD.md` P4.29 教训 #36/#37）。

---

## 5. 评估协议约定表（2026-09-26 立此存照）

| 约定项 | X-VLA | OpenVLA | PI0.5 | 说明 |
|---|---|---|---|---|
| episodes/task | 10 | 10（全量验证；首次 spatial 为 5） | 10 | 官方协议 50 trial/task，时间预算取 10 |
| env seed（init_seed） | 42 | 42 | 42 | 控制环境初始状态序列 |
| 推理 seed | 42（XVLA_SEED） | 42（OPENVLA_SEED） | 42（PI05_SEED） | 控制 torch.randn 噪声，与 env seed 是两套概念 |
| 步数上限 | client 默认 | 220/280/300/520（官方 per-suite） | 220 | 官方 run_libero_eval.py:173-182 |
| 前 10 步 dummy action | client 自带 | 是（num_steps_wait=10） | 是 | 等物体稳定 |
| act_type | abs | **rel** | rel | delta 模型必须走 rel |
| success 判定 | done flag | done flag | done 后 `env.check_success()` 复核 | PI0.5 更严谨，能排除假阳性 |
| server 端口 | 8010 | 8011（流A）/8021（流B） | 8012 | 双卡并行时用不同端口 |
| 结果目录 | `results/xvla_npu_seed*` | `results/openvla_full/` | `results/pi05_*` | 统一进项目 results/ |

---

## 6. 仿真层关键修复：渲染坐标（原 VIDEO_ORIENTATION.md，2026-09-26 并入）

### 6.1 问题现象
- 仿真视频上下颠倒（桌面在上方）+ 成功率 0%（模型收到错误方向图像）

### 6.2 根因：GPU 与 OSMesa 的坐标系差异

| 渲染方式 | 图像原点 | 图像方向 | 桌面位置 |
|---|---|---|---|
| GPU `mjr_readPixels` | 左下（GL 标准） | 上下颠倒 | 顶部（亮） |
| OSMesa `Renderer.render()` | 左上（正常） | 正常方向 | 底部（亮） |

LIBERO 客户端数据流：`robosuite env → obs['agentview_image'] → _flip_agentview(双翻转) → 发给模型 + 存视频`。`_flip_agentview` 假设输入是 GPU 的颠倒图；OSMesa 图已是正常方向，再翻转 → 倒置 + 模型输入错 → 0%。

### 6.3 修复（唯一修改点）

`robosuite/utils/binding_utils.py` 的 `read_pixels()` 返回前做 `np.flip(img, 0)` 模拟 GPU 的 GL 坐标。效果：`libero_client.py` 保持官方原样零改动，视频方向自动正确，模型输入与训练数据一致。

### 6.4 为什么只改这一处

| 方案 | 改动点 | 风险 |
|---|---|---|
| **本方案** | 仅 `read_pixels` 1 处 | 低，client 零改动 |
| 错误方案A | 改 `read_pixels` + 改 `_rollout` | 依赖两处一致，易出错 |
| 错误方案B | 仅改 `_rollout` 不翻转 | 模型输入仍错误，0% |

**关键原则**: `libero_client.py` 必须保持官方原样，所有适配在渲染层完成——客户端可随官方更新，无需维护本地改动。

### 6.5 验证方法
```python
img = obs['agentview_image']
print('顶部均值:', img[0].mean(), '底部均值:', img[-1].mean())
# 修复后: 顶部亮(>150) 底部暗(<110) = 模拟 GPU 成功
```

---

## 7. 扩展到其他 VLA 模型（原 EXTENSIBILITY.md，2026-09-26 并入）

仿真层（OSMesa 渲染 + robosuite patch + LIBERO 客户端）完全可复用，只需实现 `models/<model>/server.py` 的模型加载部分。

### 7.1 通用迁移步骤

1. **编写模型服务器**（约 50 行）：实现 `/act` 端点，返回 `[T, action_dim]` 动作数组
2. **NPU 加载**（通用模式）：
```python
import torch_npu
device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
model = Model.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").to(device)
```
3. **运行验证**：X-VLA 用 `run_eval.sh`；其他模型按 §7.3 差异走专用路径

### 7.2 各模型差异点

| 模型 | 动作格式 | 需额外处理 | NPU 算子风险 | 验证状态 |
|---|---|---|---|---|
| X-VLA | `[pos3, rot6d, grip1]` 绝对 | rot6d→axis-angle | 低 | ✅ 95.8%（4 suite, 5 seed） |
| OpenVLA | `[delta_pos3, aa3, grip1]` | act_type=rel + steps=1 + 同源转换链 | 低 | 已记录四 suite：77/79/75/57 |
| PI0.5 | `[delta_pos3, aa3, grip1]` chunk | lerobot 框架 + 输入构造契约（§4.4：图像/255、xyzw quat、fresh proprio、grip 直传） | 中（flow matching 采样） | ✅ P4.29 修复后闭环跑通，spatial 全量中 |
| SmolVLA | 离散 token | 同 OpenVLA | 低 | ⏳ 骨架未实测 |
| ACT | action chunk | CVAE 编码 | 中（卷积算子） | ⏳ 骨架未实测 |
| Diffusion Policy | DDPM 采样 | 50步去噪 | 中（需 eager attn） | ⏳ 骨架未实测 |

### 7.3 一键运行的支持边界（重要）

- **X-VLA**：`run_eval.sh xvla ...` 一键可跑（唯一全协议支持的模型）
- **OpenVLA**：`scripts/openvla/run_openvla_full_validation.sh <npu_device> <port> <suite>...`（需 per-suite ckpt + server_v2 + act_type=rel + steps=1 monkey-patch）
- **PI0.5**：`scripts/pi05/run_pi05_spatial.sh`（需独立 conda env `lerobot-pi05`）
- **新模型**：先读 §5 协议约定表 + §4.4 架构教训，动作语义与 proprio 回传是最大的坑

### 7.4 新模型适配文件布局与通用入口

新增模型统一放入：

```text
models/<model>/server_v2.py
scripts/<model>/eval_<model>.py
scripts/<model>/run_<model>.sh
docs/<model>/                 # 最多两个专项文档
results/<model>/videos/<suite>/
```

通用环境脚本：`scripts/common_env.sh`；NPU server 适配骨架：`scripts/model_server_template.py`；完整适配说明：`docs/ADAPTING_NEW_MODEL.md`；结果命名和视频约定：`results/README.md`。模型专用脚本和文档只维护在对应子目录中，命令使用子目录实际路径。

### 7.5 仿真层复用清单（所有模型通用）

- `patches/robosuite_osmesa_render.py` + `patches/robosuite_mj_fullM.py`（`apply_patches.py` 幂等应用）
- `scripts/setup_env.sh`（环境搭建）+ 渲染库 `~/render_libs/` + LIBERO assets
- LIBERO 客户端 `libero_client.py`（官方原样）+ 环境变量（`MUJOCO_GL=osmesa` 等）

---

## 8. 避坑指南（速查；完整问题链见各模型文档）

| 坑 | 解法 | 详见 |
|---|---|---|
| openvla 官方 eval import segfault | prismatic→dlimp→tensorflow 链 segfault，stub 注入绕过 | OPENVLA_HANDOVER |
| bf16 模型 + fp32 inputs NPU 报错 | inputs 必须用 infer_dtype | OPENVLA_HANDOVER |
| flash_attention_2 NPU 不支持 | sdpa（最佳）或 eager | OPENVLA_HANDOVER |
| bitsandbytes INT8 是 CUDA 专用 | bf16/fp32 原生推理 | OPENVLA_HANDOVER |
| X-VLA client 每 10 步推理 1 次 | delta/每步模型 monkey-patch steps=1 | OPENVLA_HANDOVER |
| OpenVLA delta vs X-VLA abs 路径 | act_type="rel" → use_delta=True | OPENVLA_HANDOVER §4.2 |
| LD_LIBRARY_PATH 进程内 setdefault 无效 | 必须 python 启动前 export（动态链接器只读一次） | PI05_RECORD P1.5 |
| conda env/渲染库被外部清理 | 开工先 `conda env list` + `ls ~/render_libs` 验证 | PROJECT_TRACKING 问题表 |
| ModelArts 出站代理拒所有外网 IP | checkpoint 走 modelscope 镜像；GPU 对比走人工交接 | PI05_RECORD P4.26/27 |
| 后台跑 NPU 任务丢 CANN 驱动路径 | 启动脚本先 `source ~/.bashrc`（含 Ascend set_env）再用 nohup setsid | 2026-09-26 新增 |
| 图像 uint8 直传视觉模型 | 官方管线 /255 → [0,1]；模型内 resize clamp(-1,1) 会静默摧毁 uint8 图 | PI05_RECORD P4.29 |
| 新模型闭环 0% 先查什么 | 输入逐维对照 ckpt norm_stats（训练分布对照）+ 四输入敏感性测试，再谈模型/算子 | PI05_RECORD P4.29 #36/#37 |
| 验证流被环境重启杀掉 | eval --resume 断点续跑；续跑配置需按实际结果和视频证据复核 | PROJECT_TRACKING 阶段 18 |

---

## 9. 项目里程碑

| 日期 | 里程碑 | 状态 |
|---|---|---|
| 2026-07-02 | 项目启动，OSMesa 渲染环境搭建 | ✅ |
| 2026-07-12 | 推理 seed 确定化修复（±15-20% 波动消除） | ✅ |
| 2026-07-13 | X-VLA seed42 基准 95.8%（4 suite） | ✅ |
| 2026-07-15 | 历史 X-VLA 多次运行记录完成（相同 init_seed，独立性需谨慎解释） | 已记录 |
| 2026-07-16 | OpenVLA server_v2 开发 + bf16/sdpa 测速 | ✅ |
| 2026-07-17 | OpenVLA delta 根因修复 → spatial 76.0% + 3.3x 优化 | ✅ |
| 2026-07-18 | 开源准备（隐私清理 + 规范文件） | ✅ |
| 2026-07-20 | PI0.5 验证启动（env/server/闭环跑通） | ✅ |
| 2026-07-21~23 | PI0.5 0% 根因诊断（P4.5/P4.7/P4.8 + F1/F4，8 方向推翻） | ✅ |
| 2026-07-24 | F6 use_peft 兼容 + 官方便样 ckpt 同源铁证 | ✅ |
| 2026-07-25 | GPU 对比两路阻断 → 人工交接打包 | ✅ |
| 2026-09-26 | 项目整理：P0 代码修复（脚本损伤/凭证脱敏/schema 漂移）+ 文档结构整合（14→8）+ OpenVLA 4 suite 全量验证启动 + PI0.5 NPU 基准重生成（P4.28 schema 对齐） | ✅ |
| 2026-09-26 下午 | **PI0.5 闭环 0% 终结（P4.29）**：训练分布对照法定位输入构造缺陷，修复后 spatial 原始统计 **96.0%（96/100）**；OpenVLA 四 suite JSON 已保存；外部论文/官方参考值待核实 | 已记录 |

---

## 10. 总结


**已验证**：
- **X-VLA**: 四套件原始统计为 90% / 99% / 100% / 94%，平均 95.75%；历史多 seed 记录需结合相同 init_seed 的限制解读
- **OpenVLA**: spatial/goal/object/libero_10 原始统计分别为 77/79/75/57（各100 rollouts）；视频证据仅有 object 独立复验 5/10
- **PI0.5**: 旧 spatial run 原始统计 96/100；新 NPU0 fp32 spatial run 当前完成 4/100（task0 4/10，4/4 成功），仍在进行；输入构造缺陷及修复过程见 P4.29，外部参考值和协议来源待核实

**核心技术资产**：
1. OSMesa 软件渲染方案（无 GPU 跑通 LIBERO 闭环，含 GL 坐标修复）
2. NPU 推理配置经验（bf16+sdpa 最优、fa2/INT8 不可用、dtype 匹配规则）
3. delta action 参考系根因方法论（act_type 语义对齐官方 eval）
4. 同源转换链原则（server 与 client 用同一转换函数，两端 diff=0）
5. **训练分布对照法（P4.29 新增）**：server 输入逐维对照 ckpt norm_stats + 敏感性测试，是"输入构造正确性"的最终判据——比链路内部一致性对比/GPU 双端对比都更根本（PI0.5 案例中 GPU 对比本会给出错误定论）
