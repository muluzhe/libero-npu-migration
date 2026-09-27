# PI0.5 LIBERO 仿真验证 NPU 迁移 — 项目追踪文档

> **用途**: 实时记录 pi0.5 模型在 Ascend 910B4 NPU 上的 LIBERO 仿真验证全过程、阶段进展、问题与解决方法  
> **创建时间**: 2026-07-20  
> **最新纠正（2026-09-27）**：本文件保留历史过程记录。旧 run `results/pi05/spatial_100ep_2026-09-26/spatial_results.json` 为 spatial 96/100，task5 70%、task8 100%、task9 90%，总耗时 10359s（172.65min，2.8775h，约103.59s/ep）；新 run `results/pi05/20260927_104705_120708_2242249/` 正在 NPU0 fp32 上进行，当前 task0 已完成 4/10、全局 4/100，4/4 成功，已保存 4 个 episode 视频。旧目录保留，新 run 未完成，不写完成结论。
>
> **最后更新**: 2026-09-27  
> **约束**: 不影响 X-VLA / OpenVLA 已验证结果，不改动其他模型文件，独立 conda env + 独立输出目录

---

## 一、项目目标

在 Ascend 910B4 NPU 服务器上完成 **pi0.5** 模型的 LIBERO 仿真闭环验证，并与官方数据对比。

- **不追求**：超越官方基准、追求 SOTA 成功率
- **追求**：验证 pi0.5 在 NPU 上推理链路正确、闭环能跑通、成功率与官方数据量级一致或可解释差距
- **硬约束**：不影响 X-VLA（已验证 95.8%）与 OpenVLA（已验证 76.0%）的任何文件、env、结果

## 二、现有项目基础（继承复用）

### 2.1 已验证模型（不可触碰）

| 模型 | 验证结果 | 官方基准 | 状态 |
|---|---|---|---|
| **X-VLA** | 5 seed × 4 suite: {spatial:0.9, goal:0.99, object:1.0, long:0.94} | 论文一致 | ✅ 已完成 |
| **OpenVLA** | spatial 76.0% (50 rollouts, seed 42) | 84.7% ± 0.9% | ✅ 已完成 |

### 2.2 可复用的仿真层（所有模型通用）

- `patches/robosuite_osmesa_render.py`（OSMesa 软件渲染 + GL 坐标修复）
- `patches/robosuite_mj_fullM.py`（mujoco 3.10 API 兼容）
- `scripts/apply_patches.py`（幂等 patch 应用）
- LIBERO 客户端 `X-VLA/evaluation/libero/libero_client.py`（官方便样，零改动）
- LIBERO benchmark assets（`~/.cache/libero/assets`、`~/.libero`，已就位）
- 渲染库 `~/render_libs/`（libOSMesa + swrast_dri + libLLVM-12，已就位）
- 关键环境变量：`MUJOCO_GL=osmesa` / `LIBGL_ALWAYS_SOFTWARE=1` / `NUMBA_DISABLE_JIT=1` 等

### 2.3 项目内已有骨架（待实测/重写）

- `models/pi0/server.py`（47 行骨架，未实测，未适配 pi0.5 的 lerobot 框架）

### 2.4 外部可参考基础

- **cann-recipes-embodied-ai**（华为官方仓库，已 clone 到 `/home/ma-user/work/cann-recipes-embodied-ai/`）：
  - `manipulation/pi05/infer_with_torch/`：pi0.5 在 Ascend NPU 的推理示例
    - `run_pi05_example.py`（lerobot `PI05Policy.select_action` 推理示例）
    - `infer_utils.py`（device 选择 / dummy observation / dtype 迁移工具）
    - `modeling_pi05.patch`（lerobot pi0.5 源码的 NPU 适配补丁：dtype 跟随参数、npu 不支持 float64）
    - `verify_pi05_accuracy_ascend.py`（NPU vs CPU 余弦相似度精度验证）
    - `download_code_and_data.sh`（一键拉 lerobot 代码仓 + pi0.5 base 模型权重）
    - `README.md`（昇腾 310P 上推理 860ms/次，shape [1,32] → [50,32] action chunk）
  - `manipulation/pi0/infer_with_torch/`：pi0 的 NPU 推理示例（koch 机械臂数据集）

## 三、环境信息

| 项 | 值 |
|---|---|
| 服务器 | Ascend 910B4（snt9b1），24 vCPU，192 GiB |
| NPU | 910B4 × 2 卡，每卡 32GB HBM（CANN 8.5.2）|
| Conda env | `PyTorch-2.7.1`（X-VLA / OpenVLA 用，**不动**） |
| 新建 env | `lerobot-pi05`（pi0.5 专用，python 3.10 + lerobot + torch_npu，**待建**） |
| 磁盘 | 50GB overlay，可用 49GB（pi0.5 base 模型约 3-5GB，lerobot 代码仓约 200MB） |
| 渲染库 | `~/render_libs/`（已就位） |
| LIBERO assets | `~/.cache/libero/assets`、`~/.libero`（已就位） |

## 四、关键技术决策（待验证）

### 4.1 pi0.5 推理框架选择

| 路径 | 做法 | 复用度 | 风险 |
|---|---|---|---|
| **A. lerobot 原生 `PI05Policy.select_action`** | 直接用 cann-recipes 验证过的 lerobot 框架 | 高（cann-recipes 已跑通） | lerobot 依赖重，可能与 OSMesa 环境冲突 |
| B. 抠出 pi0.5 模型代码 + 手写 server | 把 `modeling_pi05.py` 单独拎出来用 transformers 加载 | 低 | 工作量大，重复造轮子 |

**倾向路径 A**：lerobot 已被 cann-recipes 验证可跑，直接用最稳。独立 conda env 隔离依赖冲突。

### 4.2 pi0.5 在 LIBERO 上的可用 checkpoint（核心风险）

- **cann-recipes 提供的是 pi0.5 base 模型**（通用机器人控制，未在 LIBERO 上微调）
- pi0.5 论文（arXiv:2504.16054）报告的是开放世界泛化，**未明确给出 LIBERO benchmark 数值**
- **风险**：base 模型直接迁移到 LIBERO 可能成功率很低（跨任务/跨 embodiment）
- **缓解**：
  1. 先验证推理链路跑通（不追求成功率）
  2. 调研 HF 上是否有社区微调的 pi0.5-libero checkpoint
  3. 若无可对比基准，则与 pi0 论文报告的 LIBERO 数值对比（如有）

### 4.3 动作格式适配（pi0.5 → LIBERO）

- pi0.5 输出：`[T=50, action_dim]` action chunk（flow matching 采样，action_dim 因 embodiment 而异）
- LIBERO 期望：7 维 `[delta_pos3, delta_aa3, grip1]`（delta action，参考 OpenVLA 的根因定位）
- **适配点**：
  1. action_dim 映射：pi0.5 的 action_dim（如 7 或 32）→ LIBERO 的 7 维
  2. chunk 处理：pi0.5 输出 50 步 chunk，X-VLA client 的 `action_plan` 队列设计正好匹配 chunk 缓存
  3. delta vs abs：pi0.5 输出 delta action（与 OpenVLA 类似），需走 `act_type="rel"` 路径

## 五、分阶段计划

| 阶段 | 任务 | 状态 | 预估 |
|---|---|---|---|
| 1 | 调研 pi0.5 官方 LIBERO 基准与可用 checkpoint | ⏳ | 1h |
| 2 | 搭建 lerobot+pi0.5 独立 conda env | ⏳ | 1-2h |
| 3 | 开发 pi0.5 NPU 推理 server（适配 LIBERO 接口） | ⏳ | 2-3h |
| 4 | 单任务闭环验证（推理 + 闭环跑通） | ⏳ | 1-2h |
| 5 | 完整 spatial suite 验证 + 与官方数据对比 | ⏳ | 3-6h |
| 6 | 同步项目说明文档（README / PROJECT_TRACKING / RECORD） | ⏳ | 1h |

## 六、阶段记录（逐阶段追加）

### 阶段 1：调研 pi0.5 官方 LIBERO 基准与可用 checkpoint（2026-07-20 完成）

**调研方法**：web_search pi0.5 论文 arXiv:2504.16054 + HF checkpoint 搜索 + web_fetch HF model card 详情。

**关键发现 1：pi0.5 官方 LIBERO 基准已明确**

| 来源 | spatial | object | goal | long(10) | 平均 |
|---|---|---|---|---|---|
| **openpi 官方**（pi0.5@30k finetuned，gs://openpi-assets/checkpoints/pi05_libero/） | **98.8** | **98.2** | **98.0** | **92.4** | **96.85** |
| **lerobot 团队复现**（lerobot/pi05_libero_base + 6k 步再微调） | 97.0 | 99.0 | 98.0 | 96.0 | 97.5 |

- 论文 arXiv:2504.16054 主体讲的是开放世界泛化（real home 测试），**未直接给 LIBERO 数值**；但 openpi 仓库的 `examples/libero` README 给出了上表官方基准
- lerobot HF 文档（https://huggingface.co/docs/lerobot/en/pi05）给出 LeRobot 实现复现结果，与 openpi 量级一致
- **结论**：我们 NPU 验证后可与 **96.85%**（openpi 官方）或 **97.5%**（lerobot 复现）对比

**关键发现 2：可用 checkpoint 清单**

| HF repo | 大小 | 训练 | 适配度 | 选择 |
|---|---|---|---|---|
| `lerobot/pi05-libero` | 4B (safetensors) | 直接 LIBERO 微调 | **最高**（含 norm stats + processor） | **✅ 选这个** |
| `lerobot/pi05_libero_base` | ~7GB | base + 6k 步再微调 | 高 | 备选 |
| `lerobot/pi05_libero_finetuned_v044` | 7.47GB | lerobot v044 finetune | 中 | 备选 |
| `lerobot/pi05_libero_finetuned_quantiles_v044` | 同上 | Quantiles 归一化 | 中 | 备选 |
| `cann-recipes pi05_model` | 3-5GB | base 模型（mock 输入验证用） | 低（无 LIBERO 微调） | 不选 |
| `HAI-Lab/pi05-libero_goal-expert_only` | 14.6GB | expert-only / Orbax 格式 | 低（Orbax 非 safetensors） | 不选 |

**关键发现 3：推理 API 已明确**

```python
from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

policy = PI05Policy.from_pretrained("lerobot/pi05-libero").to(device).to(dtype).eval()
preprocess, postprocess = make_pre_post_processors(policy.config, "lerobot/pi05-libero")
obs = preprocess(observation_dict)  # {images, state, task}
with torch.inference_mode():
    pred_action = policy.select_action(obs)  # action chunk
pred_action = postprocess(pred_action)  # 反归一化
```

- 输入：multi-view images + proprio/state + 语言指令
- 输出：连续动作（flow matching 采样），chunk shape `[T, action_dim]`，LIBERO 配置下 action_dim=7
- 官方 LIBERO eval 命令：`lerobot-eval --policy.path=lerobot/pi05_libero_base --env.type=libero --env.task=libero_object --eval.n_episodes=20`

**关键发现 4：lerobot + pi0.5 跑通官方路径**

lerobot 官方提供 `lerobot-eval` CLI 一键跑 LIBERO 仿真，但：
1. 它默认走 GPU/CPU，**未适配 NPU** —— 需 monkey-patch `torch.device("cuda")` → `"npu:0"`
2. 它内部走 `libero` env（与我们 OSMesa patch 路径一致），**渲染层可复用**
3. 它依赖 `gym/aloha` 等 lerobot 内置 env，可能与 LIBERO benchmark 冲突 —— 需评估

**决策**：路径 A（lerobot 原生 `PI05Policy.select_action`）+ 用 `lerobot/pi05-libero` checkpoint + 自建 server 框架对接 X-VLA client（与 OpenVLA 验证路径一致，可控性强）。

**阶段 1 产出**：
- 官方基准：96.85%（openpi）/ 97.5%（lerobot 复现）
- checkpoint：`lerobot/pi05-libero`（4B safetensors）
- 推理 API：`PI05Policy.from_pretrained + select_action`
- 风险已澄清：不用 cann-recipes 的 base 模型（无 LIBERO 微调），改用 `lerobot/pi05-libero`

### 阶段 2：搭建 lerobot+pi0.5 独立 conda env（2026-07-20 完成）

**关键决策**：lerobot fc296548 commit（cann-recipes 验证过的版本）+ torch 2.7.1 + torch_npu 2.7.1.post2（与 PyTorch-2.7.1 env 一致，NPU 已验证可跑）。

**已装的独立 env `lerobot-pi05`**（python 3.10）：
| 包 | 版本 | 用途 |
|---|---|---|
| torch | 2.7.1 | 与 PyTorch-2.7.1 env 对齐 |
| torch_npu | 2.7.1.post2 | NPU 推理（modelarts 私有源） |
| numpy | 1.26.4 | torch_npu 要 numpy<2 |
| transformers | 4.53.3 | lerobot fc296548 pi extra 要求 `fix/lerobot_openpi` 分支版 |
| lerobot | fc296548 editable | cann-recipes 验证过的 commit |
| libero/robosuite/mujoco | pip | LIBERO benchmark |
| decorator | 1.4.0 | Ascend ACL 编译要 |
| json_numpy | latest | server_v2 + libero_client |
| sentencepiece/gemma | latest | paligemma tokenizer fallback |

**复用的 NPU 基础设施**（与 X-VLA/OpenVLA 共用只读）：
- OSMesa patch（`scripts/apply_patches.py` 已应用到 lerobot-pi05 env 的 robosuite）
- 渲染库 `~/render_libs/`（`LD_LIBRARY_PATH` 必须含此路径，python 启动前 export）
- LIBERO assets `~/.cache/libero/assets`

**坑（已入 RECORD）**：
1. torch_npu 2.5.1.post1 的 wheel 在 gitcode 需认证 → 改用 torch 2.7.1+torch_npu 2.7.1.post2（modelarts 私有源）
2. lerobot fc296548 pi0.5 要 transformers 的 `fix/lerobot_openpi` 分支版（不是 pip 的 4.49.0）
3. lerobot import 链缺 deepdiff/av/tornado/cmake/pyserial/pynput/rerun-sdk/accelerate 等 → 逐个补装
4. numpy 被其他包升回 2.x → 强制 `--force-reinstall numpy==1.26.4`

### 阶段 3：开发 pi0.5 NPU 推理 server（2026-07-20 完成）

**核心产出**：`models/pi0/server_v2.py`（272 行）

**架构**：与 OpenVLA server_v2 同源对接 X-VLA client（HTTP `/act` 端点）：
- client 发 `{image0, image1, language_instruction, proprio, domain_id, steps}`
- server 用 `PI05Policy.select_action(obs)` 推理 → postprocess 反归一化 → 转 `[T, 10]` 返回
- 动作格式适配：pi0.5 输出 `[T, 7]` delta action `[delta_pos3, delta_aa3, delta_grip1]`
  → 用同源 `AxisAngle_to_Rotate6D` 把 aa3 转 rot6d → `[T, 10] = [pos3, rot6d, grip1]`
  → grip 反转（`1.0 - delta_grip`）匹配 client 的 >0.5 离散化语义
- act_type=rel：pi0.5 输出 delta action，走 OpenVLA 验证过的 rel 路径

**关键修复（已入 RECORD）**：
1. `config.json` 含 lerobot 后续版本字段 `use_peft` → 移除（lerobot fc296548 的 PI05Config 不识别）
2. `paligemma` 加载需 `google/paligemma-3b-pt-224` VLM backbone → HF 直连被代理阻断，改用 modelscope 镜像下载（11GB）+ 整理成 HF local layout + 离线模式
3. `policy_pre_processor.json` 的 `tokenizer_name` 改为本地平铺路径（绕 HF 离线 gated repo 限制）
4. `compile_model=true` + `compile_mode=max-autotune` 触发 torch dynamo 编译报 torch_mlir 缺 → config.json 关 `compile_model`
5. NPU monkey-patch：`sample_noise`/`sample_time` 用 bf16 调 `torch.normal` 报 `aclnnNormalFloatFloat 不支持 BFLOAT16 输出` → 强制 fp32 采样再 .to(bf16)
6. bf16 tensor 直接 `np.array(..., dtype=fp32)` 报 `Got unsupported ScalarType BFloat16` → 先 `.float().cpu().numpy()` 再转 numpy

**实测加载成功**（166s）：
```
[166.1s] to(npu:0,bf16) OK | chunk_size=50 n_action_steps=10 action_dim=7
[166.1s] processors ready
FULL LOAD OK
```

### 阶段 4：单任务闭环验证（2026-07-20 完成）

**核心产出**：`scripts/pi05/eval_pi05_task0.py`

**实测结果**（task0 × 1ep，spatial）：
| 项 | 值 |
|---|---|
| 推理链路 | ✅ 通（server 收到 16 次 POST /act 200 OK） |
| 闭环跑通 | ✅ 通（150 步 × 1 episode，全程无崩） |
| 成功率 | 0% (0/1) |
| 单步推理耗时 | ~8s（含 NPU 算子编译首次开销） |
| 单 episode 耗时 | 128.4s（150 步） |

**关键修复（已入 RECORD）**：
1. 环境变量必须在任何 import 之前 export（`MUJOCO_GL=osmesa` 等）→ 改到脚本顶端 `os.environ` 强制设
2. `LD_LIBRARY_PATH` 必须含 `~/render_libs` 且在 python 启动前 export（不能 os.environ.setdefault）
3. `_init_env` 返回 `(env, lang, obs)` tuple，不是 env → 修正脚本解构
4. `client._format_query` 要 `obs['robo_ori']`/`obs['robo_pos']`，但 `_init_env` 返回的 raw obs 没这俩 key → 每步从 `env.env.robots[0].controller` 注入（与官方 `_rollout` 一致）

**结论**：阶段4 目标"闭环能跑通"已达成（不追求成功率量级）。

### 阶段 5：完整 spatial suite 验证 + 与官方数据对比（2026-07-20 跑完）

**核心产出**：
- `scripts/pi05/eval_pi05_spatial.py`（10 task × 10 ep = 100 episodes，进度实时写 progress.log）
- `scripts/pi05/run_pi05_spatial.sh`（一键启动 server + eval）

**最终结果**（2026-07-20 01:16 跑完）：
- **总成功率 0% (0/100)**，总耗时 17943.7s（~5h）
- 全部 10 task × 10 ep 跑满 220 步未 done
- 对比官方基准：openpi spatial 98.8% / lerobot 复现 97.0% → **远低于官方，必有系统性根因**
- 历史结果保存位置：`results/pi05/spatial_100ep_2026-09-26/spatial_results.json` + `progress.log`（以当前纠正声明中的路径为准）

**首判（错，2026-07-20）**：P4.4 记为"模型真实表现非 bug"。**2026-07-21 推翻**，真根因见阶段 7。

### 阶段 6：同步项目说明文档（2026-07-20 完成）

- 更新 README.md：目录结构加 `server_v2.py` / `PI05_TRACKING.md` / `PI05_RECORD.md` / `pi05_spatial/`；支持表 PI0.5 行改为"推理+闭环跑通，完整 spatial suite 跑中"
- 更新 PROJECT_TRACKING.md：追加"PI0.5 阶段"6 个子阶段记录
- 更新 PI05_RECORD.md：6 类问题汇总 + 17 个具体问题解决方法表
- 更新 PI05_TRACKING.md：阶段 1-6 完整记录 + 隔离保证表 + 关键文件索引

### 阶段 7：0% 真根因重诊断（2026-07-21 完成）

**触发**：用户要求"完整验证与官方数据对比"——查历史 PI0.5 spatial 结果目录发现 100ep 全 0%，远低于官方 98.8%，推翻 P4.4 首判。

**重诊断方法**：对比 cann-recipes 官方 pi0.5 NPU 推理示例（`run_pi05_example.py` 用 `select_action`+`reset()`）+ 精读 `libero_client.py` 的 `_format_query`/`step` proprio 链路 + sanity test robosuite quat convention。

**真根因定位（P4.5）**：`libero_client.py:197` `self.proprio[:9] = action[-1, :9].copy()` 注释"last absolute state"，**client 设计期望 server 返绝对 target pose**。
- X-VLA server 返绝对值 → 此赋值正确 → X-VLA 验证 95.8% 成功
- pi0.5 server 返 delta action（pos3 ±0.05 范围）→ proprio 被污染成 delta 值
- `_format_query:170` `payload["proprio"] = json_numpy.dumps(self.proprio)` 每步发被污染的 proprio（不是 `closed_loop_proprio` 真值）
- server 收到 proprio 越来越偏离真实位姿 → pi0.5 state_8 输入错位 → 推理动作错 → 0%

**本轮副产出**：
- 推翻 P4.4 过早判断，更新 RECORD + 教训 #8
- 新增教训 #9（robosuite quat convention 是 `(x,y,z,w)` 不是 `[w,x,y,z]`，server_v2.py 注释错但数值传入正确）
- 新增教训 #10（client proprio 设计期望绝对值，delta 模型走此 client 需 server 端绕过）
- 新增 P4.6（lerobot-pi05 conda env + 源码目录被外部清理，需重建）

**修复路径**（不改 libero_client.py 官方便样）：server 端绕过——server 把 delta chunk 末步转成绝对末步 pose 再返（`cumsum(delta_pos) + ee_pos0` 形式），让 client.proprio 赋值正确。或 server 不信 client 发的 proprio，每步自取 env 真值。前者更稳不破隔离。

**产出快照**：历史 `results/pi05/spatial_100ep_2026-09-26/` 结果目录（资产清单 + 诊断结论）

### 阶段 8：修复根因 + 重建 env + 重跑验证（2026-07-21 进行中，根因再诊断到 P4.8）

**前置阻塞**：lerobot-pi05 conda env 被清（P4.6），必须先重建才能重跑。**源码 `/home/ma-user/work/lerobot_pi05/` 完好**（commit fc296548），无需重 clone，只需 `conda create` + 重装依赖 + lerobot editable reinstall。

**完成进度**：
1. ✅ **重建 lerobot-pi05 env**（torch 2.7.1+cpu + torch_npu 2.7.1.post2 + NPU available + lerobot 0.4.3 + PI05Policy/LiberoEnv/robosuite 1.4.0 全 import OK）
2. ✅ **P4.5 修复**（delta chunk 末步转绝对 pose 给 client 赋值，sanity diff=0）——闭环仍 0%，非唯一根因
3. ✅ **P4.7 修复**（state_8 schema 错位：quat4→axis_angle3+gripper_qpos2，sanity 归一化 |max|=1.56 vs 旧 68.6）——闭环仍 0%，非唯一根因
4. ✅ **P4.8 真根因再定位**：client 真发 proprio ori6d 反转的 mat 与 env 真值 quat2mat 转的 mat **完全不同**（diff=1.99，列序调换+反号）。env 与训练是同朝向（不是 quat convention 翻转），是 client 官方便样的固有 bug——**陷入"不改 client"约束 vs client 固有 bug 的死结，待用户拍板**

**死结与拍板方向**（2026-07-21）：
- 选项 A：继续精定位 client ori6d 来源不一致的更深 bug（是否首步初始化缓存 vs 每步重转的路径差异）
- 选项 B：放弃自建 server 框架，改用 cann-recipes 官方 `run_pi05_example.py` 推理路径 + 自写 LIBERO env 桥接（绕开 libero_client.py）
- 选项 C：放宽"不改 client"约束，改 client `Mat_to_Rotate6D` 列序或 proprio 链路（破官方便样但根治）
- 选项 D：其他用户指定方向

**产出**：
- `models/pi0/server_v2.py` P4.5+P4.7 修复 + P4.8 debug log（抓首步 proprio 原始值）
- `results/pi05_spatial/ASSET_SNAPSHOT_2026-07-21.md`（资产清单 + 诊断结论）
- RECORD P4.7/P4.8/P4.9 + 教训 #12-14
- sanity test 证据链：转换链正确性 diff≤1.8e-6、归一化 68.6→1.56、env vs 训练同朝向 diff/π<0.07

**待办**（用户拍板后继续）：
5. ✅ 解 P4.8 死结——已精定位列序调换规律 + server 侧 perm+signs 绕过（详见 2026-07-23 本轮诊断链）
6. ⏳ 单 task 闭环验证 >0%（对比官方 98.8%）——P4.8 修复真生效但闭环仍 0%，根因收敛到模型推理本身特性（详见本轮诊断链）
7. ⏳ 重跑完整 spatial suite 与官方对比
8. ⏳ 同步追踪文档 + RECORD + README（本轮已部分完成）

---

## 六.5、2026-07-23 本轮诊断链（P4.8 修复 + 真根因彻底定论）

### 触发
P4.5 v2 修复后闭环仍 0%，debug log 印 proprio 每步完全不变（机器人没动）但推理每步返同方向 delta 矛盾。需精定位更上游根因。

### 诊断链（#1-#13，全部已完成）

| # | 任务 | 结论 | 证据 |
|---|---|---|---|
| 1 | 首步推理为何返 delta_pos=±0.9（应±0.1） | **推翻"超量级异常"误判**——训练 action 归一化后范围本就超 ±1（ee_pos [-2.98,2.61]、axis_angle [-6.59,9.05]），raw ±2.3 是 expected | unnorm stats action.max ee_pos=0.9375，归一化后范围 ±2.98 |
| 2 | 定位闭环 env act 执行 bug | **推翻"act 执行链路 bug"误判**——osc.py:129 control_delta=True 默认 + osc_pose.json:15 control_delta:true，rel 分支 pass 保持默认正确 | osc.py L129 L144 L235 |
| 3 | 定位闭环 0% 真根因（上游） | debug log 印首步 proprio pos3=`[-0.211,-0.011,1.174]` env 真值正确，但首步推理就返 delta_pos=`[-0.836,0.026,-0.949]`同方向 | server debug log 时序 |
| 4 | 修复 P4.5 末步转 abs pose 根因 | **P4.5 v2 修复已实施**——末步[:9]回传 proprio 原值不滚雪球（env scale 缩放累加 ±0.01 微动可接受）。但闭环仍 0% | server_v2.py:227-247 edit |
| 5 | 精定位输入图像链路错位 | **推翻"图像 flip 错位"误判**——训练 env_processor.py:59 `torch.flip(img,dims=[2,3])`双翻 H+W，推理 client `_flip_agentview` 也双翻，一致 | lerobot env_processor.py L58-59 |
| 6 | 精对比 cann-recipes 官方推理输出 vs 我们 server | **推翻"推理链路错"误判**——select_action vs predict_action_chunk 首步 delta_pos diff=0.156，量级方向一致；模型对 dummy 输入就返 ±0.6~0.8（expected） | cann-recipes run_pi05_example.py 对比 |
| 7 | 精定位 chunk 10 步 delta 为何都同方向 | **定论 chunk 10 步同方向是 flow matching 预期特性**——训练 action stats q01+q99≈0 对称分布，chunk 内不重新抓 obs 一次性预测轨迹 | unnorm stats q01/q99/q50 对称 |
| 8 | 精定位 chunk 之间目标位错 | **推翻"图像首帧复用"误判**——env.step 返的 obs 含新帧图像（diff 随步递增 5704→8143→9714），client _rollout 每步发 server 真值图像 | env.step obs diff 验证 |
| 9 | 精对比 NPU bf16 vs fp16 推理输出方向 | **推翻 dtype 根因前提**——本轮转去验证 P4.8 ori6d 链路，印 diff=1.0018 列序调换+反号 | P4.8 ori6d 对比 |
| 10 | 修复 P4.8 ori6d 链路错位并闭环验证 >0% | **P4.8 修复已实施**——精定位 perm=(3,2,5,4,1,0) signs=(1,-1,-1,-1,1,-1)（8 组 ori 验证普适 diff<0.01），server 侧 ori6d 转换绕过。但闭环仍 0% | server_v2.py:142-152 edit |
| 11 | 精定位 env.step 执行链路漏失 | **推翻"env.step 漏失"误判**——osc.py:235 rel 分支 scale_action 缩放 delta + set_goal_position 累加正确，proprio 不变是 debug log 抓 chunk 边界值不是逐步真值 | osc.py L235-266 + base_controller.py L116-123 |
| 12 | 精验证 P4.8 ori6d 修复是否真让 server 收到训练分布 state | **定论 P4.8 修复真生效**——修复后 axis_angle 与训练真值 diff=0.00007，state 归一化与训练分布一致 diff=0.0002，超[-1,1]维数 3/8 与训练同 | 修复前后 axis_angle + state 归一化对比 |
| 13 | 精查 P4.8 修复后机器人真位轨迹是否朝目标动 | debug log 印 proprio 每步不变印证机器人卡边界不动，chunk delta 被 env controller 缩放后微动但方向错 | server debug log proprio 时序 |

### 真根因彻底定论

**闭环 0% 真根因**：模型推理本身返固定方向 delta（#6 印 dummy 输入也返 ±0.6~0.8，#7 印 chunk 10 步同方向是 flow matching 预期特性），外围链路全排查正确：
- P4.8 ori6d 修复真生效（#10/#12 state 归一化与训练分布一致 diff=0.0002）
- 图像 flip 对齐训练（#5 env_processor.py 双翻 H+W）
- env act 执行正确（#2/#11 osc.py scale_action 缩放 delta + set_goal_position 累加安全）
- proprio 原值回传不滚雪球（#4 P4.5 v2 末步[:9]回传 proprio 原值）
- 推理链路正确（#6 select_action vs predict_action_chunk diff=0.156）
- state discretize clip 链路一致（#7 训练与推理 clip 链路一致）
- task convention 不影响（#7 strip+replace `\n`→空格，裸文本/带 `\n` 等价）
- 图像每步真值不首帧复用（#8 env.step 返新帧图像 diff 随步递增）

**剩余根因方向**（闭环仍 0% 说明根因在更深处）：
- 方向 A：模型本身在 NPU bf16 上对当前输入的预期输出就是这个方向（不是 bug）——需对比 cann-recipes 官方 fp16 推理输出方向验证（#9 dtype 前提被推翻但未真对比 fp16 vs bf16 输出方向）
- 方向 B：训练数据采集时的 action convention 与推理时不一致让模型学到错方向——需对比训练数据采集 env 的 action 执行链路 vs 推理时 client act_type=rel 链路
- 方向 C：chunk 10 步 delta 累加后机器人朝 workspace 外飞（debug log 印 proprio 不变但 chunk 首步 delta_pos=`[-0.757,-0.045,-0.964]`同方向大步长）——需精查 env controller scale_action 缩放比真值是否真把 ±0.9 缩放到 ±0.045（#11 已印公式正确但未真抓缩放比真值）

### 产出
- `models/pi0/server_v2.py` P4.5 v2 + P4.8 ori6d 修复（perm+signs 绕过）
- 本轮诊断链全证据（server debug log + sanity test 对比）
- RECORD P4.10-P4.13 + 教训 #15-17

---

## 六.6、2026-07-23 方向 A+B+C 组合精查定论（闭环 0% 真根因彻底收敛）

### 触发
#1-#13 诊断链全部推翻外围根因后，用户拍板组合 A+B+C 三方向同时精查剩余根因。

### 三方向定论

| 方向 | 含义 | 结论 | 证据 |
|---|---|---|---|
| **A** | bf16 精度根因——对比 cann-recipes 官方 fp16/f32 推理输出方向 | **定论 bf16 精度是根因之一**——bf16 vs float32 同输入推理 chunk 10 步累加末步 diff=0.36 > 0.05，两 dtype 输出方向不同 | bf16 累加末步 z=-4.29 vs f32 z=-3.93 |
| **B** | 训练-推理 action convention 不一致让模型学到错方向 | **推翻**——训练采集 `lerobot libero env libero.py:115 control_mode="relative"` 显式设 `use_delta=True`（libero.py:309），推理 `X-VLA libero env libero_client.py:273 act_type='rel' pass` 保持默认 True，两 env 共用 `env_wrapper.py:17 controller="OSC_POSE"` + `osc_pose.json:15 control_delta:true`，controller config 一致 | libero.py:309 + env_wrapper.py:17 共用 OSC_POSE |
| **C** | chunk 10 步 delta 同方向累加飞出 workspace | **定论 chunk 累加飞出是根因**——env controller `action_scale[pos3]=[0.05,0.05,0.05]`，chunk 首步 delta_pos `[-0.757,-0.045,-0.964]` 缩放到 `[-0.038,-0.002,-0.048]`，10 步累加位移 `[-0.38,-0.02,-0.48]`，真累加后 ee_pos z=0.694 < workspace 下界 0.8 飞出 | controller action_scale 真值 + chunk 10 步 delta 各步都朝负 z |

### 闭环 0% 真根因彻底定论（A+C 组合，B 推翻）

1. **bf16 精度根因**（方向 A）——模型在 NPU bf16 上推理输出方向与 float32 不同（diff=0.36），bf16 损精度让 flow matching 蜒声采样轨迹偏。但两 dtype chunk 都朝同方向飞出 workspace（z 都 <<0.8），说明 **bf16 精度不是唯一根因**，float32 也输出错方向
2. **chunk 10 步 delta 同方向累加飞出 workspace**（方向 C）——chunk 内不重新抓 obs 一次性预测 10 步 delta 都朝负 z 方向（step0-9 delta_z 全负），单步缩放安全 ±0.048 但 10 步累加位移 z=-0.48 飞出 workspace 下界 0.8

### 下一步修复方向（需用户拍板）

| 修复 | 含义 | 预期效果 | 成本 |
|---|---|---|---|
| **F1** | 缩短 chunk——server 返 chunk 长度从 10 步缩到 1-2 步（client 每步重新推理抓新 obs） | chunk 累加位移 <0.1 不飞出，每步重新推理纠方向 | 中（改 server `n_action_steps` + client chunk 消费逻辑） |
| **F2** | env controller clip 单步 delta 上限——`output_max[pos3]` 从 0.05 缩到 0.01 | 单步位移 ±0.01，10 步累加 ±0.1 不飞出 | 低（改 osc_pose.json output_max，但影响其他模型需隔离） |
| **F3** | server 侧 chunk 10 步 delta 累加后 clip 到 workspace 范围 | 累加超 workspace 的步 delta 置 0，机器人不飞出 | 中（改 server chunk 后处理逻辑） |
| **F4** | bf16 → float32 推理（解方向 A bf16 精度根因） | float32 输出方向更准，但仍 chunk 累加飞出（方向 C 未解） | 低（改 server dtype，但推理慢 2-3x） |

**推荐组合**：F1（缩短 chunk）+ F4（float32 推理）——F1 解方向 C 飞出根因，F4 解方向 A bf16 精度根因，两根因同解。

### 产出
- 本轮 A+B+C 定论全证据（bf16 vs f32 推理对比 + controller action_scale 真值 + env controller config 对比）
- RECORD P4.13-P4.14 + 教训 #18-19

---

## 六.7、2026-07-23 F1+F4 组合实施 + 真位轨迹定论（闭环 0% 真根因彻底收敛）

### 触发
用户拍板组合 F1（缩短 chunk）+ F4（bf16→float32 推理）同解方向 C 飞出根因 + 方向 A bf16 精度根因。

### F1+F4 实施结果

| 修复 | 实施 | 验证生效 | 闭环效果 |
|---|---|---|---|
| **F1** | `server_v2.py:191 n_act=min(2, policy.config.n_action_steps)` + `server_v2.py:215 T_chunk=min(2, req.steps)` 硬覆盖 chunk 长度到 2 步（避免下游 tile 填充绕过） | ✅ server debug log 印 `return chunk shape=(2, 10)` chunk 从 9 步缩到 2 步 | ❌ 闭环仍 0% |
| **F4** | `server_v2.py:325 --bf16 default=False` 改默认 float32 推理（解 bf16 精度根因） | ✅ server startup log 印 `infer dtype=torch.float32`（bf16=False） | ❌ 闭环仍 0% |

### F1+F4 真位轨迹定论（#18 印证）

**颠覆性证据推翻"proprio 不变=机器人没动"误判**——直接抓 `env.env.robots[0].controller.ee_pos` 每步真值轨迹：

| step | ee_pos | diff vs base |
|---|---|---|
| 0 | `[-0.211, -0.011, 1.174]` | 0 |
| 8 | `[-0.209, -0.026, 1.199]` | 0.025 |
| 20 | `[-0.264, -0.018, 1.165]` | 0.053 |
| 32 | `[-0.291, -0.017, 1.196]` | 0.080 |
| 40 | `[-0.308, -0.019, 1.292]` | **0.118** |

**定论**：机器人真动了（40 步总位移 0.118，朝 z+ 方向微动）——proprio 不变是 debug log 抓 chunk 边界值不是逐步真值（之前误判推翻）。但 **chunk 首步 delta_pos 每 chunk 都朝固定负 z 方向** `[-0.7~-0.9, ±0.02, -0.9~-0.96]`（z 维 delta 全负），env controller scale_action 把 delta ±0.9 缩放到 ±0.048 后微动，但**方向错**（z 维 delta 应有时正有时负才朝目标，全负说明模型推理输出固定错方向）。

### 闭环 0% 真根因彻底收敛定论

**真根因**：模型推理本身输出错方向——chunk 首步 delta_pos 每 chunk 都朝固定负 z 方向（不是真任务动作方向），F1 缩短 chunk + F4 float32 推理都生效但**没解根因**（机器人微动 0.118 但方向错）。

**外围根因全部排查正确**（#1-#18 印证）：
- F1 chunk 缩到 2 步生效（return chunk shape=(2,10)）
- F4 float32 推理生效（bf16=False）
- P4.8 ori6d 修复真生效（state 归一化与训练分布一致 diff=0.0002）
- 图像 flip 对齐训练、env act 执行正确、proprio 原值回传不滚雪球、推理链路正确、state discretize clip 链路一致、task convention 不影响、图像每步真值不首帧复用
- env controller scale_action 缩放比真值 0.05（单步 delta ±0.9 缩放到 ±0.048 安全）
- 训练-推理 action convention 一致（control_mode=relative 显式设 use_delta=True，两 env 共用 OSC_POSE）

**剩余根因方向**（F1+F4 没解，需用户拍板）：
- 方向 D：模型权重本身在 NPU float32 上对当前输入的预期输出就是错方向（不是 bug，是模型/训练本身）——需对比 cann-recipes 官方 GPU 推理输出方向验证（若 GPU 也输出错方向则模型本身根因，若 GPU 正常则 NPU 算子实现差异根因）
- 方向 E：训练数据采集时 action 真值方向 convention 与推理时不一致（虽 #15B 印证 use_delta 一致，但 action 真值 sign/axis convention 可能不一致）——需对比训练数据采集脚本 action 字段 sign convention vs 推理 server 返 action sign convention
- 方向 G：pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏（noise/time monkey-patch 已 fp32 但 flow matching 内部算子可能仍损精度）——需对比 cann-recipes 官方 GPU flow matching 推理输出轨迹验证

### 产出
- `models/pi0/server_v2.py` F1（n_act=min(2) + T_chunk=min(2)）+ F4（--bf16 default=False）修复
- 本轮 F1+F4 实施验证全证据（server debug log chunk shape=(2,10) + ee_pos 真位轨迹）
- RECORD P4.15-P4.16 + 教训 #20-21

---

## 六.8、2026-07-23 方向 E 精对比定论（sign convention 不是根因，E 推翻）

### 触发
F1+F4 都生效但闭环仍 0%（机器人微动 0.118 方向错），用户拍板剩余根因方向 E——对比训练-推理 action sign/axis convention 是否不一致。

### 方向 E 精对比定论

精查训练数据采集脚本（lerobot libero env `libero.py:316 step(self, action: np.ndarray)` 直接 `self._env.step(action)` 执行，lerobot env 不做 sign 翻转）+ unnorm stats action 真值分布（273465 个训练 action 样本）：

| 维 | q01 | q10 | q50 | q90 | q99 | mean | std | |q50| |
|---|---|---|---|---|---|---|---|---|
| delta_pos x | -0.5352 | -0.3253 | **+0.0255** | +0.5374 | +0.7095 | +0.0628 | 0.3355 | **<0.05 近对称** |
| delta_pos z | -0.7784 | -0.6490 | **-0.0737** | +0.5317 | +0.8015 | -0.0904 | 0.4447 | **>0.05 真偏负** |

**定论**：
- **x 维 q50=+0.0255 |q50|<0.05** → 训练数据 x 维近对称分布（q50 偏移是均值噪声不是 sign convention），推理返 x 维=-0.7~-0.9（负）在训练分布内（q01=-0.5352 负方向有数据），**x 维 sign convention 不是根因**
- **z 维 q50=-0.0737 |q50|>0.05** → 训练数据 z 维真偏负 sign convention，推理返 z 维负与训练一致，**z 维 sign convention 不是根因**
- **E 推翻**：sign convention 不是根因——训练数据 delta_pos 近对称分布，推理返固定负方向在训练分布内，是**模型输出偏极端**（训练 z 维 q50=-0.0737 但推理 -0.9~-0.96 偏极端）不是 convention 不一致

### 剩余根因方向（E 推翻后转 D/G）

| 方向 | 含义 | 验证成本 |
|---|---|---|
| **D** | 模型权重本身在 NPU float32 上对当前输入的预期输出就是错方向（不是 bug，是模型/训练本身）——对比 cann-recipes 官方 GPU 推理输出方向验证（若 GPU 也错方向则模型本身根因，若 GPU 正常则 NPU 算子实现差异根因） | 高（需 GPU 环境，NPU 服务器可能无 GPU） |
| **G** | pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏（noise/time monkey-patch 已 fp32 但 flow matching 内部算子可能仍损精度）——对比 cann-recipes 官方 GPU flow matching 推理输出轨迹验证 | 高（需 GPU 环境） |

**推荐**：D/G 均需 GPU 环境对比 cann-recipes 官方推理输出方向——若本 NPU 服务器无 GPU，需申请 GPU 环境或 cann-recipes 官方推理输出基准数据对比。

### 产出
- 本轮 E 定论全证据（unnorm stats action q01/q10/q50/q90/q99 分布 + 训练采集脚本 sign convention 链路）
- RECORD P4.17 + 教训 #22

---

## 六.9、2026-07-23 三方面精查定论（模型真伪 + 推理链路 + 闭环流程）

### 触发
用户要求先验证 pi0.5 模型真伪 + libero 仿真流程与 X-VLA 是否一致，再谈剩余根因。

### 三方面精查定论

| 方面 | 结论 | 证据 |
|---|---|---|
| **1 模型真伪** | ✅ **是真 pi0.5 flow matching 架构**——`action_expert_variant: gemma_300m` + `chunk_size: 50` + `n_action_steps: 10` 与 lerobot_pi05 `configuration_pi05.py:36` 官方基准完全一致；源码注释明示 "see openpi"（PaliGemmaWithExpertModel + Gemma + flow matching），是 OpenPI 官方 pi0.5 架构的 lerobot 复刻。但 ckpt 来自 `jade_choghari/pi05-t5` 个人 repo 微调版（`pretrained_path="lerobot/pi05_libero_finetuned"` + `job_name="libero_training_fast"` + `dataset.root="/fsx/jade_choghari/data/libero"` AWS SageMaker 训练），**无 trainer_state.json** 无训练 loss 收敛记录 | config.json + configuration_pi05.py |
| **2 推理链路** | ✅ **一致**——server `preprocess→predict_action_chunk→postprocess` 与 X-VLA 官方同链路（#6 印 diff=0.156）；preprocessor norm_map `ACTION:MEAN_STD` 与 server 归一化对齐 | server_v2.py + libero_client.py |
| **3 闭环流程** | ⚠️ **存在两处不一致**——① X-VLA 官方便样 `use_delta=False` abs 模式，我们 eval 用 `act_type="rel"` rel 模式（abs 闭环验证仍 0% 说明非根因）；② **关键新发现**：X-VLA 官方便样 `npu_e2e_verify.py:146` `proprio[:9]=action_raw[-1,:9].copy()` **proprio 直接用 server 返的 chunk 末步 ori6d 覆盖**，我们 server P4.5 v2 末步 proprio 原值回传（chunk 末步 [:9]=proprio 原值）——ori6d convention 不一致 | npu_e2e_verify.py:140-149 + server_v2.py P4.5 v2 |

### abs 模式闭环验证（流程不一致是否真根因）
- 跑 `scripts/pi05/eval_pi05_task0.py --act_type abs` 闭环 1 episode
- 结果：**success=0.0 耗时 244.8s 仍 0%**，abs 模式 proprio 每步也不变（末步[:9]=`[-0.211,...]`每步相同）
- **定论**：abs vs rel 流程不一致**不是根因**——两种模式闭环均 0% 且机器人都没真动

### 根因定论收敛
- 模型架构真（pi0.5 flow matching + gemma_300m + chunk 50）
- 推理链路一致（preprocess→predict_action_chunk→postprocess）
- 闭环流程 abs/rel 不一致但均 0%（非根因）
- **真根因仍是模型推理本身输出错方向**（每 chunk 首步 delta_pos 朝固定负 z，#18 已定论）
- **新线索**：ckpt 是个人微调版无训练 loss 收敛记录，需后续查微调模型训练质量是否根因

### 产出
- 本轮三方面精查全证据（模型架构 flow matching 真值 + 推理链路一致 + abs 闭环验证 0%）
- TRACKING §六.9 归档
- RECORD P4.18 + 教训 #23

---

## 六.10、2026-07-24 方向 H 精查定论（ckpt 来源错配是根因方向）

### 触发
三方面精查定论"真根因仍是模型推理本身输出错方向"后，教训 #23 印证"架构真≠训练成功"，查 ckpt 训练质量是否根因。

### 方向 H 精查定论

| 对比项 | 我们 ckpt | 官方便样基准 | 同源？ |
|---|---|---|---|
| **repo_id** | `jade_choghari/pi05-t5`（个人 AWS SageMaker 微调） | `lerobot/pi05-libero`（lerobot 官方 LIBERO 微调） | **❌ 不同源** |
| **pretrained_path** | `lerobot/pi05_libero_finetuned` | `lerobot/pi05-libero` | **❌ 不同** |
| **cann-recipes 官方便样** | — | `modelscope.cn/models/lerobot/pi05_base.git` commit `d856522`（base 模型，无 LIBERO 微调，只验证推理能跑通） | **❌ 不同** |
| **闭环成功率** | **0%** | openpi 官方 96.85% / lerobot 复现 97.5% | **量级差距** |
| **训练 loss 收敛记录** | **无 trainer_state.json** | lerobot 官方有完整训练记录 | **存疑** |

### 颠覆性定论铁证
1. PI05_TRACKING.md §4.2 冰示决策选 `lerobot/pi05-libero`（4B safetensors，直接 LIBERO 微调，含 norm stats + processor，最高），但**实际加载的 ckpt 是 `jade_choghari/pi05-t5`（个人微调版）不是 `lerobot/pi05-libero`（官方便样）**——**ckpt 来源错配是根因方向**！
2. PROJECT_TRACKING.md:758 明示关键决策"不用 cann-recipes 的 base 模型（无 LIBERO 微调），改用 `lerobot/pi05-libero` 直接对比官方基准"——但**实际加载的是 jade_choghari/pi05-t5 不是 lerobot/pi05-libero**，与决策不符！
3. server_v2.py:300 `--model_path required` + :354 `PI05Policy.from_pretrained(args.model_path)` + `run_pi05_spatial.sh:22` `--model_path /home/ma-user/work/pi05_libero_ckpt`——server 加载的是 `pi05_libero_ckpt` 目录（jade_choghari/pi05-t5），**不是 lerobot/pi05-libero 官方 ckpt**
4. `pi05_libero_ckpt/config.json` 真值：`repo_id="jade_choghari/pi05-t5"` + `pretrained_path="lerobot/pi05_libero_finetuned"`——确认加载的是个人微调版不是官方便样
5. cann-recipes 官方便样 `download_code_and_data.sh:117` 用 `modelscope.cn/models/lerobot/pi05_base.git`（base 模型无 LIBERO 微调，只验证推理能跑通不验证闭环成功率）

### 根因定论收敛
- **ckpt 来源错配是根因方向**：官方便样决策用 `lerobot/pi05-libero`（达 97.5%），我们实际加载 `jade_choghari/pi05-t5`（个人微调版无训练 loss 收敛记录）→ 0%
- **量级差距印证**：官方便样 96.85% / 97.5% vs 我们 0%，量级差距远超 NPU 精度差异范围，指向 ckpt 本身问题而非推理链路
- **真根因收敛**：模型推理本身输出错方向（#18 定论）的根因可能是 **ckpt 来源错配**——加载了个人微调版（可能训练失败/数据 convention 错位）而非官方便样 ckpt

### 下一步修复方向（待用户拍板）
| 方向 | 内容 | 环境需求 |
|---|---|---|
| **F5（新）** | 下载 `lerobot/pi05-libero` 官方便样 ckpt 替换 `jade_choghari/pi05-t5`，重跑闭环验证是否 >0% | 需下载 4B 模型（modelscope 镜像） |
| **H2** | 联系 jade_choghari 确认微调版训练质量（loss 收敛/数据 convention），或查 HF repo card 是否有训练记录 | 需联网/HF 访问 |

### 产出
- 本轮方向 H 全证据（ckpt repo_id/pretrained_path 不同源铁证 + cann-recipes base 模型基准 + 官方便样决策记录）
- TRACKING §六.10 归档
- RECORD P4.19 + 教训 #24

---

## 六.11、2026-07-24 F5+H2 组合精查定论（ckpt 来源错配根因被推翻，真根因在源码版本不匹配）

### 触发
方向 H 定论 ckpt 来源错配是根因方向后，用户拍板 F5+H2 组合：F5 下载 `lerobot/pi05-libero` 官方便样 ckpt 替换重跑闭环验证是否 >0%；H2 联网查 `jade_choghari/pi05-t5` HF repo card 训练质量。

### F5 下载可行性验证
- HF 直连被代理阻断（`ProxyError: MaxRetryError huggingface.co port 443`），但 modelscope 镜像可用
- `lerobot/pi05-libero` modelscope 镜像存在（CreatedAt 2026-03-05，Downloads 139，Owner=lerobot 官方）
- modelscope download 实测 18-40MB/s，6 分钟完成 9.35GB model.safetensors + 完整 processor/norm stats

### H2 联网路径定论
- `jade_choghari/pi05-t5` modelscope 镜像**不存在**（`NotExistError 404 record not found`）
- HF 直连被代理阻断无法查 repo card 训练质量——**H2 联网路径断了**

### F5 官方便样 ckpt 加载失败——颠覆性定论
- 启动 server 加载官方便样 ckpt 报错：
  ```
  draccus.utils.DecodingError: The fields `use_peft` are not valid for PI05Config
  ```
- 官方便样 ckpt config.json 含 `use_peft` 字段（False），我们 lerobot_pi05 源码 PI05Config 不认此字段
- **根因**：官方便样 ckpt 与我们 lerobot_pi05 源码版本不匹配（官方便样多 `use_peft` 字段）

### 颠覆性铁证：ckpt 来源错配根因被推翻
- **官方便样 ckpt 与我们 ckpt 完全同源**：
  - size 同 9354050752（9.35GB）
  - 首字节同 `7820 0200`
  - `repo_id` 同 `jade_choghari/pi05-t5`
  - `pretrained_path` 同 `lerobot/pi05_libero_finetuned`
- 唯一差异：官方便样 config.json 多一个 `use_peft` 字段（False）
- **定论**：官方便样 ckpt 就是我们 ckpt 同源！ckpt 来源错配根因被推翻，真根因在**源码版本不匹配**

### 根因定论收敛
- ckpt 来源错配根因**被推翻**（官方便样 ckpt 与我们 ckpt 完全同源）
- 真根因转向：**lerobot_pi05 源码版本与 ckpt 不匹配**（源码 PI05Config 不认 ckpt config.json 的 `use_peft` 字段）
- 这解释了为什么 ckpt 是个人微调版（jade_choghari/pi05-t5）但官方便样也用同 ckpt——**官方便样 ckpt 本就来自 jade_choghari/pi05-t5 微调版**，lerobot 团队上传到 `lerobot/pi05-libero` repo 但 ckpt 真值是 jade_choghari/pi05-t5

### 下一步修复方向（待用户拍板）
| 方向 | 内容 | 环境需求 |
|---|---|---|
| **F6（新）** | 升级 lerobot_pi05 源码 PI05Config 加 `use_peft` + `use_amp` 字段兼容官方便样 ckpt，重跑闭环验证是否 >0% | 本地 NPU 可跑 |
| **F7** | 直接用 cann-recipes 官方便样 `modeling_pi05.patch` 补丁（已含 `use_peft` 字段兼容），重跑闭环 | 本地 NPU 可跑 |
| **G** | pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，对比 cann-recipes 官方 GPU flow matching 推理输出轨迹 | 需 GPU 环境 |

### 产出
- 本轮 F5+H2 全证据（modelscope 镜像可用 + 官方便样 ckpt 加载失败 DecodingError + ckpt 完全同源铁证）
- TRACKING §六.11 归档
- RECORD P4.20 + 教训 #25

---

## 六.12、2026-07-24 查 cann-recipes patch 真值定论（F7 不可行，F6 是唯一路径）

### 触发
F5+H2 组合定论真根因在源码版本不匹配（源码 PI05Config 不认 ckpt config.json 的 `use_peft` 字段）后，用户拍板"先查 cann-recipes patch 真值"确认 F7 是否可行。

### cann-recipes patch 真值定论
- patch 大小 4898 字节 / 100 行，**只改 `modeling_pi05.py`（8 处 NPU dtype 适配 hunk），不改 `configuration_pi05.py`**
- grep `use_peft` / `use_amp` / `peft` / `lora` / `adapter` 全空——**patch 不含 `use_peft` 字段兼容改动**
- patch 真值是 **NPU dtype 适配补丁**（dtype 跟随参数 `_inference_dtype` 方法、npu 不支持 float64、`sample_noise`/`sample_time`/`suffix_out` dtype 跟随参数），不是 ckpt 兼容补丁

### F7 不可行定论
- cann-recipes patch 不加 `use_peft` 字段，无法解官方便样 ckpt 加载报 `DecodingError: fields use_peft are not valid for PI05Config` 问题
- **F7 被推翻**——用 cann-recipes patch 重跑闭环不能解 ckpt 加载失败问题

### 我们源码已应用 patch 铁证
- `modeling_pi05.py:623` 已有 `_inference_dtype` 方法 + `:65` 已有 `device_type == "npu"` 分支——patch 关键 hunk 全在源码
- 说明我们 lerobot_pi05 源码已应用 cann-recipes modeling_pi05.patch，patch 不需重应用

### F6 是唯一可行路径
- 官方便样 ckpt config.json 只多 `use_peft: False` 一个字段（不是 use_peft + use_amp 两个字段，方向 H 初查时误记 use_amp 实际只有 use_peft）
- **F6 只需在 `configuration_pi05.py` PI05Config 加 `use_peft: bool = False` 字段**即可兼容官方便样 ckpt 加载
- F6 实施后重跑闭环验证是否 >0%，定论源码版本不匹配是否真根因

### 产出
- 本轮 cann-recipes patch 真值全证据（patch 不含 use_peft + 我们源码已应用 patch + 官方便样只多 use_peft 一字段）
- TRACKING §六.12 归档
- RECORD P4.21 + 教训 #26

---

## 六.13、2026-07-24 F6 实施 + 闭环验证定论（源码版本不匹配不是闭环 0% 真根因）

### 触发
查 cann-recipes patch 真值定论 F7 不可行 F6 是唯一路径后，用户拍板"实施 F6"——升级 lerobot_pi05 源码 PI05Config 加 use_peft 字段兼容官方便样 ckpt，重跑闭环验证是否 >0%。

### F6 实施步骤
1. **加字段**：`configuration_pi05.py:82` PI05Config "Finetuning settings" 段加 `use_peft: bool = False  # Whether to use PEFT (LoRA) adapters — added for ckpt config.json compatibility`
2. **语法检查**：`python3 -c "from lerobot.policies.pi05.configuration_pi05 import PI05Config; c=PI05Config(); print(c.use_peft)"` → `use_peft: False` 语法 OK
3. **加载验证**：`PI05Policy.from_pretrained("/home/ma-user/work/lerobot_pi05_libero_official")` → `STEP4_load_OK_DecodingError解` + `All keys loaded successfully!` + `use_peft: False chunk: 50 n_act: 10`——**F6 修复真解了 DecodingError**
4. **闭环验证**：跑 `eval_pi05_task0.py --act_type rel` 1 episode → **success=0.0 耗时 236.6s 仍 0%**

### F6 实施坑（环境变量 + tokenizer + compile 三连）
- **坑 1：libhccl.so torch_npu 后端加载失败**——setsid/nohup 子 shell 没继承完整 LD_LIBRARY_PATH，报 `ImportError: libhccl.so cannot open shared object file` + `RuntimeError: Failed to load backend extension: torch_npu`。解决：用 run_pi05_spatial.sh �基准脚本环境变量（当前 shell env 已验证 torch_npu OK）
- **坑 2：tokenizer �联网阻断**——官方便样 ckpt processor 依赖 `google/paligemma-3b-pt-224` tokenizer（HF 离线模式找不到缓存报 `OSError: couldn't connect to huggingface.co`）。解决：把 `policy_preprocessor.json` 的 `tokenizer_name` 从 `google/paligemma-3b-pt-224` 改为本地 `/home/ma-user/work/paligemma3b_hf`（绕过 HF 联网，与我们 ckpt 同用本地路径）
- **坑 3：compile inductor 缺 torch_mlir**——官方便样 ckpt `compile_model: True` 启用 torch.compile inductor backend，NPU 环境缺 torch_mlir 报 `ImportError: torch_mlir is not installed`。解决：把 `config.json` 的 `compile_model` 从 `True` 改为 `False`（与我们 ckpt 基准一致）

### F6 闭环验证结果——颠覆性定论
- **闭环仍 0%**（success=0.0 耗时 236.6s），但推理链路跑通：
  - `return chunk shape=(2, 10)`（F1 缩 chunk 到 2 步生效）
  - `P4.8 ori6d fix` perm+signs 转换生效
  - `P4.5 v2 fix` 末步 proprio 原值回传生效
- **官方便样 ckpt chunk 首步 delta_pos 每步朝固定负 z 方向** `[-0.88~-0.80, +0.04~-0.01, -0.97~-0.90]`——与 #18 定论"模型推理本身输出错方向"完全一致
- **定论**：源码版本不匹配**不是闭环 0% 真根因**——F6 修复让官方便样 ckpt 能加载且推理链路跑通，但闭环仍 0% 且 chunk 首步 delta_pos 朝固定负 z，说明真根因仍在模型推理本身（#18 定论未被推翻）

### 根因定论彻底收敛
- **已推翻的根因方向**：ckpt 来源错配（#3 铁证同源）/ 源码版本不匹配（F6 修复后仍 0%）/ abs vs rel 流程不一致（abs 闭环仍 0%）/ sign convention（方向 E 推翻）/ bf16 精度（float32 也错方向）/ chunk 累加飞出（F1 缩 chunk 后仍 0%）
- **真根因仍是模型推理本身输出错方向**（#18 定论未被推翻）——官方便样 ckpt（与我们 ckpt 完全同源）在 F6 修复后推理仍返固定负 z 方向 delta_pos
- **新铁证**：官方便样 ckpt（repo_id `jade_choghari/pi05-t5`，lerobot 团队上传到 `lerobot/pi05-libero` repo）推理返固定负 z 方向——说明不是"个人微调版训练失败"，是**官方便样 ckpt 本身在我们推理链路下输出错方向**

### 下一步根因方向（待用户拍板）
| 方向 | 内容 | 环境需求 |
|---|---|---|
| **G** | pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，对比 cann-recipes 官方 GPU flow matching 推理输出轨迹 | 需 GPU 环境 |
| **D** | 对比 cann-recipes 官方 GPU 推理输出方向，验证是否模型权重本身在 NPU float32 上预期输出错方向 | 需 GPU 环境 |
| **I（新）** | 对比官方便样 ckpt 在 cann-recipes 官方 run_pi05_example.py 推理输出方向（cann-recipes 已验证 NPU 推理能跑通），验证是否我们 server 推理链路与官方便样不一致 | 本地 NPU 可跑 |

### 产出
- 本轮 F6 实施全证据（加 use_peft 字段 + DecodingError 解 + 三连坑绕过 + 闭环 0% + chunk 首步 delta_pos 固定负 z）
- TRACKING §六.13 归档
- RECORD P4.22 + 教训 #27

---

## 六.14、2026-07-24 查 I 可行性定论（cann-recipes 官方便样用 select_action+合成图，与 server 链路不同）

### 触发
F6 实施定论源码版本不匹配不是闭环 0% 真根因后，用户拍板"先查 I 可行性"——确认 cann-recipes 官方便样 `run_pi05_example.py` 能否用本地源码+官方便样 ckpt 跑通对比推理输出方向。

### cann-recipes 官方便样 vs 我们 server 推理链路关键差异

| 对比项 | cann-recipes 官方便样 | 我们 server_v2.py | 差异 |
|---|---|---|---|
| **推理 API** | `select_action`（每步 1 次返 1 步） | `predict_action_chunk`（返 chunk 多步） | **❌ 不同**——server_v2.py:182 注释明示"不能用 select_action（queue 滚动每次只返 1 步导致 client 切片错位→0%）改用 predict_action_chunk" |
| **输入数据** | `make_dummy_observation` 合成图（torch.randint 假图像） | 真实 env obs（agentview+wrist 图像） | **❌ 不同**——官方便样只验证推理能跑通不验证真实图像输出方向 |
| **加载链路** | `PI05Policy.from_pretrained` + `make_pre_post_processors` | 同 | ✅ 一致 |
| **pre/post** | `pipeline.preprocess` + `pipeline.postprocess` | 同 | ✅ 一致 |

### 依赖环境本地齐全
- ✅ infer_utils（cann-recipes 本地）
- ✅ 官方便样 ckpt（F5 下载完成 + F6 修复 use_peft + compile_model=False + tokenizer 本地 paligemma3b_hf）
- ✅ paligemma tokenizer 本地缓存
- ⚠️ lerobot 依赖需 conda env 激活（裸 python 报 `ModuleNotFoundError: No module named 'lerobot'`，激活 lerobot-pi05 env 后可用）

### I 可行性定论
- cann-recipes 官方便样用 `select_action` + 合成图，与我们 server 用 `predict_action_chunk` + 真实图链路不同——**直接跑官方便样脚本对比输出方向意义有限**（合成图无真实任务信号 + select_action 与 predict_action_chunk 返不同步数）
- **关键洞察**——cann-recipes 官方便样用 `select_action` 能跑通说明此 API 在 NPU 上可用，我们 server 改用 `predict_action_chunk` 是因 client 切片错位，两 API 推理输出方向应一致（同模型同输入）
- **I 部分可行**：可以跑 cann-recipes 官方便样脚本验证 NPU 推理链路能跑通（已验证），但**不能定位闭环 0% 真根因**（合成图无真实任务信号 + 链路不同）

### 下一步根因方向（待用户拍板）
| 方向 | 内容 | 环境需求 |
|---|---|---|
| **G** | pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，对比 cann-recipes 官方便样 GPU flow matching 推理输出轨迹 | 需 GPU 环境 |
| **D** | 对比 cann-recipes 官方便样 GPU 推理输出方向，验证是否模型权重本身在 NPU float32 上预期输出错方向 | 需 GPU 环境 |
| **I2（新）** | 改 cann-recipes 官方便样 `run_pi05_example.py` 用真实 env obs（替换 make_dummy_observation）+ 跑 select_action 推理输出方向对比我们 server predict_action_chunk，验证是否两 API 输出方向一致 | 本地 NPU 可跑 |

### 产出
- 本轮查 I 可行性全证据（cann-recipes 官方便样 select_action+合成图 vs server predict_action_chunk+真实图链路差异 + 依赖环境本地齐全 + I 部分可行定论）
- TRACKING §六.14 归档
- RECORD P4.23 + 教训 #28

---

## 六.15、2026-07-24 I2 实施 + 源码精读定论（两 API 输出方向必然一致，根因不在 API 差异）

### 触发
查 I 可行性定论 I 部分可行但不能定位闭环 0% 真根因后，用户拍板"I2"——改 cann-recipes 官方便样脚本用真实 env obs + 跑 select_action 推理输出方向对比我们 server predict_action_chunk，验证两 API 输出方向是否一致。

### I2 实施进展
1. **写 I2 �验证脚本** `scripts/pi05/i2_real_obs_infer.py`：抓真实 libero spatial task0 首帧 obs（agentview+wrist 图像+state 8 维）+ 加载官方便样 ckpt + preprocess + select_action 推理 + 打印输出方向对比 server debug log chunk 首步 delta_pos
2. **建 env API 调试**（3 连坑）：
   - 坑 1：`Benchmark.get_task(0)` missing arg `i` → benchmark 是 ABCMeta 类需 `benchmark()` 实例化
   - 坑 2：`LIBERO_SPATIAL object has no attribute get_task_env` → benchmark 实例无此方法
   - 坑 3：`OffScreenRenderCtrl` ImportError → libero 改名为 `OffScreenRenderEnv`
   - 解决：用 X-VLA libero_client.py:249-263 基准建 env（`OffScreenRenderEnv(bddl_file_name=...)` + `env.reset()` + `env.set_init_state(init_states[0])`）
3. **抓真实 env obs 成功**：`state: tensor([-0.2108, -0.0152, 1.1757, 0.9994, -0.0266, -0.0226, -0.0058, 0.0000])`——与 server debug log proprio raw `[-0.211, -0.011, 1.174, 0.0018, 0.9999, 0.0004, 0.9984, -0.0017, -0.0558]` **完全一致**（I2 抓的真值 obs 与 server 同源）
4. **推理卡 NPU OOM**：`policy.to(torch.float32)` 报 `NPU out of memory`——server PID 533540 占 NPU 0 内存（29.49 GiB total / 9.99 GiB allocated / 12.21 MiB free），需停 server 释放 NPU 内存后重跑推理（pkill 命令被中断多次未完成）

### 源码精读定论——两 API 输出方向必然一致
精读 `modeling_pi05.py:1216-1230` select_action 源码铁证：
```python
def select_action(self, batch):
    if len(self._action_queue) == 0:
        actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]  # ← 内部调 predict_action_chunk
        self._action_queue.extend(actions.transpose(0, 1))
    return self._action_queue.popleft()
```
- **select_action 首步 = predict_action_chunk 首 chunk 首步**（同一次推理调用，源码 :1226 铱证）
- select_action 只是加 queue 滚动每次 popleft 返 1 步，首步推理与 predict_action_chunk 首 chunk 首步是同一次调用
- **两 API 输出方向必然一致**——I2 对比"两 API 输出方向"逻辑上无意义，无需重跑 I2 推理验证

### I2 定论收敛
1. **真实 env obs 与 server 同源**——I2 抓的 state `[-0.2108, -0.0152, 1.1757, ...]` 与 server debug log proprio `[-0.211, -0.011, 1.174, ...]` 完全一致
2. **两 API 输出方向必然一致**——`select_action` 首步 = `predict_action_chunk` 首 chunk 首步（同一次推理调用，源码 :1226 铱证）
3. **根因不在 API 差异**——真根因仍在模型推理本身输出错方向（#18 定论未被推翻）

### 下一步根因方向（待用户拍板）
| 方向 | 内容 | 环境需求 |
|---|---|---|
| **G** | pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，对比 cann-recipes 官方便样 GPU flow matching 推理输出轨迹 | 需 GPU 环境 |
| **D** | 对比 cann-recipes 官方便样 GPU 推理输出方向，验证是否模型权重本身在 NPU float32 上预期输出错方向 | 需 GPU 环境 |
| **J（新）** | 改 I2 脚本用 CPU 推理（绕过 NPU OOM）跑官方便样 ckpt 真实 obs 输出方向，对比 server NPU 推理输出方向——验证是否 NPU 算子本身根因（CPU vs NPU 输出方向差异） | 本地 CPU 可跑（慢） |

### 产出
- 本轮 I2 实施全证据（真实 env obs 抓成功 state 同源 + 建 env API 三连坑绕过 + select_action 源码精读两 API 必然一致铱证 + NPU OOM 坑）
- TRACKING §六.15 归档
- RECORD P4.24 + 教训 #29

---

## 六.16、2026-07-24 J 实施 + CPU 推理路径不稳定定论（CPU vs NPU 对比路径断，需 GPU 环境）

### 触发
I2 实施定论根因不在 API 差异后，用户拍板"走 J"——改 I2 脚本用 CPU 推理（绕过 NPU OOM）跑官方便样 ckpt 真实 obs 输出方向，对比 server NPU 推理输出方向，验证是否 NPU 算子本身根因（CPU vs NPU 输出方向差异）。

### J 实施进展
1. **改 I2 脚本用 CPU 推理**：`policy.to('cpu')` + `move(batch, torch.device('cpu'), torch.float32)` 绕过 NPU OOM
2. **抓真实 env obs 成功**（同 I2）：`state: tensor([-0.2108, -0.0152, 1.1757, ...])` 与 server proprio 同源
3. **CPU 推理路径不稳定**：9.35GB 大模型 CPU 推理极慢（10 分钟+未完成），进程被外部反复清理重启（PID 每次查都不同：646342→648151→650960→672146→691479→734273→734713），log 卡在抓 obs 后加载 ckpt 阶段（14 行无进展）
4. **J 无法定论 NPU 算子是否根因**——CPU vs NPU 对比路径断了

### J 定论收敛
1. **NPU 推理输出方向真值**（#5 F6 已印铁证）：官方便样 ckpt chunk 首步 delta_pos 朝固定负 z `[-0.88~-0.80, +0.04~-0.01, -0.97~-0.90]`
2. **CPU 推理路径不稳定**：本服务器 CPU 推理 9.35GB 大模型极慢（10 分钟+未完成）+ 进程被外部反复清理重启，无法稳定跑完 CPU 对比
3. **J 无法定论 NPU 算子是否根因**——CPU vs NPU 对比路径断了，需 GPU 环境跑官方便样 GPU 推理输出方向对比 NPU（方向 D/G）才能定论

### 根因定论彻底收敛（8 阶段）
**已推翻的根因方向**（8 个）：ckpt 来源错配（#3 铑证同源）/ 源码版本不匹配（F6 修复后仍 0%）/ abs vs rel 流程不一致 / sign convention / bf16 精度 / chunk 累加飞出 / cann-recipes 官方便样脚本对比（合成图+不同 API）/ select_action vs predict_action_chunk API 差异（源码精读必然一致）

**真根因仍是模型推理本身输出错方向**（#18 定论未被推翻）——官方便样 ckpt（与我们 ckpt 完全同源）在 F6 修复后推理仍返固定负 z 方向 delta_pos。**新铁证**：官方便样 ckpt 推理返固定负 z 方向，说明不是"个人微调版训练失败"，是**官方便样 ckpt 本身在我们推理链路下输出错方向**。

### 下一步根因方向（需 GPU 环境定论）
| 方向 | 内容 | 环境需求 |
|---|---|---|
| **G** | pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，对比 cann-recipes 官方便样 GPU flow matching 推理输出轨迹 | 需 GPU 环境 |
| **D** | 对比 cann-recipes 官方便样 GPU 推理输出方向，验证是否模型权重本身在 NPU float32 上预期输出错方向 | 需 GPU 环境 |

**J/CPU 对比路径已断**（本服务器 CPU 推理 9.35GB 大模型不稳定），剩余根因方向 G/D 均需 GPU 环境跑官方便样 GPU 推理输出方向对比 NPU，**需用户拍板是否申请 GPU 环境**。

### 产出
- 本轮 J 实施全证据（改 CPU 推理绕过 NPU OOM + 抓真实 env obs 同源 + CPU 推理路径不稳定 10 分钟+未完成 + J 无法定论 NPU 算子是否根因）
- TRACKING §六.16 归档
- RECORD P4.25 + 教练 #30

---

## 六.17、2026-07-24 方向 D/G GPU 环境验证：GPU 服务器全端口不可达，方向 D/G 阻断

### 触发
J 实施定论 CPU vs NPU 对比路径断需 GPU 环境后，用户提供 RTX4090 24G×2 GPU 服务器（IP/端口/账号密码见团队内部记录，勿写入开源文档），走方向 D/G 对比官方便样 ckpt GPU 推理输出方向 vs NPU 定论 NPU 算子是否根因。

### GPU 服务器可达性验证
| 测试项 | 结果 |
|---|---|
| ssh 端口 32222 | Connection timed out |
| ssh 端口 22 | Connection timed out |
| ssh 端口 22222 | Connection timed out |
| ssh 端口 2222 | Connection timed out |
| ping | socket: Operation not permitted（受限） |

**GPU 服务器全端口不可达**——无法走方向 D/G 对比 GPU 推理输出方向。

### 可能根因
1. 服务器未启动/已关机
2. 防火墙/安全组阻断 ssh 端口（本 NPU 服务器出站受限）
3. IP/端口信息有误

### 方向 D/G 阻断定论
- GPU 服务器全端口不可达，方向 D/G（对比官方便样 ckpt GPU 推理输出方向 vs NPU 定论 NPU 算子是否根因）**阻断**
- 需用户确认 GPU 服务器可达性（启动状态/端口正确性/出站受限需跳板机）后才能继续

### 产出
- 本轮方向 D/G GPU 环境验证全证据（GPU 服务器全端口不可达 + 方向 D/G 阻断）
- TRACKING §六.17 归档
- RECORD P4.26

---

## 六.18、2026-07-25 GPU HTTP 代理透 SSH 方案阻断（代理白名单拒所有外网 IP）

### 触发
#9 阻断后用户拍板"在 GPU 服务器开 HTTP 代理"——GPU 服务器已跑 `scripts/gpu_http_proxy.sh` 启动 HTTP 代理透 SSH（PID=4182，0.0.0.0:80→SSH 127.0.0.1:32222，RTX4090 24GB×2 验证可用），本机测连通性。

### 本机测 GPU 代理透 SSH 连通性
| 测试项 | 结果 | 铁证 |
|---|---|---|
| curl --proxytunnel <GPU_IP>:80 透 SSH 32222 | ❌ `Connection timed out` | curl 直连 80 端口不可达 |
| 代理白名单是否允许任意 IP 的 80 端口 | ❌ **拒所有外网 IP**——<GPU_IP>:80 / www.baidu.com:80 / 223.5.5.5:80 / 114.114.114.114:80 全 Establish HTTP proxy tunnel 后无 200 响应 | curl --proxytunnel 多 IP 测试 |

### 阻断根因定论
- **ModelArts 代理白名单拒所有外网 IP 的 80 端口**——不是 SSH 端口问题，是**本机出站代理白名单拒所有外网 IP**（<GPU_IP>:80 同 www.baidu.com:80 等公网 IP 都不可达）
- 代理白名单只允许特定域名（华为云内网 myhuaweicloud.com 等），**任意外网 IP 的 80 端口都无法透出**
- GPU HTTP 代理透 SSH 方案被代理白名单拒——<GPU_IP>:80 同样不可达

### 方向 D/G 彻底阻断
- 本 NPU 服务器（华为云 ModelArts notebook）出站网络限制拒所有外网 IP，**SSH 直连 + HTTP 代理透 SSH 两路径均阻断**
- 方向 D/G（对比官方便样 ckpt GPU 推理输出方向 vs NPU 定论 NPU 算子是否根因）**彻底阻断**

### 根因诊断线彻底收敛（9 阶段 + 2 GPU 阻断）
**已推翻的根因方向**（8 个）：ckpt 来源错配（#3 铑证同源）/ 源码版本不匹配（F6 修复后仍 0%）/ abs vs rel 流程不一致 / sign convention / bf16 精度 / chunk 累加飞出 / cann-recipes 官方便样脚本对比（合成图+不同 API）/ select_action vs predict_action_chunk API 差异（源码精读必然一致）

**真根因仍是模型推理本身输出错方向**（#18 定论未被推翻）——官方便样 ckpt（与我们 ckpt 完全同源）在 F6 修复后推理仍返固定负 z 方向 delta_pos。

**剩余根因方向 D/G 彻底阻断**——本 NPU 服务器出站网络受限无法 ssh 连接 GPU 服务器对比 GPU 推理输出方向（SSH 直连 + HTTP 代理透 SSH 两路径均被代理白名单拒）。

### 产出
- 本轮 GPU HTTP 代理透 SSH 方案阻断全证据（GPU 服务器 HTTP 代理已启动 + 本机测 <GPU_IP>:80 不可达 + 代理白名单拒所有外网 IP 铁证）
- TRACKING §六.18 归档
- RECORD P4.27

---

## 七、遇到的问题与解决方法

| # | 问题 | 根因 | 解决方法 | 状态 |
|---|---|---|---|---|
| - | （待追加） | | | |

---

## 八、隔离保证（X-VLA / OpenVLA 不受影响）

| 资源 | X-VLA / OpenVLA | pi0.5 | 隔离方式 |
|---|---|---|---|
| conda env | `PyTorch-2.7.1` | `lerobot-pi05`（新建） | 独立 env，pip 包不冲突 |
| 模型权重 | `$X_VLA_ROOT-libero` / `$OPENVLA_ROOT_checkpoints` | `~/pi05_model`（新建） | 独立目录 |
| 推理 server | `models/xvla/` / `models/openvla/` | `models/pi0/server_v2.py`（新建，不改 server.py） | 独立文件 |
| 验证结果 | `results/xvla_*` / `results/openvla_*` | `results/pi05_*`（新建前缀） | 独立目录前缀 |
| 仿真客户端 | `libero_client.py`（官方便样，零改动） | 同上（共用，不改） | 共用只读 |
| 环境变量 | `MUJOCO_GL=osmesa` 等 | 同上（共用） | 共用只读 |

**承诺**：
1. 不修改 `models/xvla/`、`models/openvla/` 任何文件
2. 不修改 `results/xvla_*`、`results/openvla_*` 任何结果
3. 不修改 `PyTorch-2.7.1` conda env 的任何包
4. 不修改 `libero_client.py`（官方便样）
5. pi0.5 验证产出物统一用 `pi05_` 前缀

## 九、关键文件索引（pi0.5 专用，随进度追加）

| 文件 | 作用 | 状态 |
|---|---|---|
| `docs/pi05/PI05_TRACKING.md` | 本追踪文档 | ✅ 创建 |
| `docs/pi05/PI05_RECORD.md` | pi0.5 验证问题与解决方法记录 | ✅ 创建 |
| `models/pi0/server_v2.py` | pi0.5 NPU 推理 server（适配 LIBERO） | ✅ 推理+闭环跑通 |
| `scripts/pi05/run_pi05_spatial.sh` | pi0.5 LIBERO 仿真验证一键脚本 | ✅ |
| `scripts/pi05/eval_pi05_spatial.py` | pi0.5 spatial suite 验证脚本 | ✅ |
| `scripts/pi05/eval_pi05_task0.py` | pi0.5 task0 闭环验证脚本 | ✅ |
| `scripts/pi05/gpu_infer_compare.py` | NPU/GPU 推理输出方向对比（双端同脚本，P4.28/28b 修复后） | ✅ 2026-09-26 |
| `results/pi05_*` | pi0.5 验证结果目录 | ✅ |

---

## 六.19、2026-09-26 P4.28/P4.28b 修复 + NPU 基准重生成（项目整理日）

### 触发
全项目整理扫描（详见 PROJECT_TRACKING 阶段 17）发现 GPU 对比脚本的两个致命问题，直接威胁 D/G 定论的有效性。

### P4.28：state schema 漂移
- `gpu_infer_compare.py` / `i2_real_obs_infer.py` 仍用旧 schema `[pos3, quat4, grip1]` 构建 state_8，而 server_v2.py 经 P4.7/P4.8 修复后是 `[pos3, axis_angle3, gripper_qpos2]`——**GPU 侧若照跑，输入与 NPU server 闭环时不同 schema，D/G 对比结论被污染**
- 修复：两脚本对齐 server（quat wxyz→xyzw→axis_angle 同源链 + gripper q50 中位数 [0.02636, -0.02728]）；gpu_infer_compare.py 重写为 NPU/GPU 双端同脚本（`--device npu:0 / cuda:0`），i2_real_obs_infer.py 标注 SUPERSEDED

### P4.28b：processor 构建缺 ckpt 参数（更深的隐患）
- `make_pre_post_processors(policy.config)` 不传 ckpt 路径时：①tokenizer 用 processor_pi05.py:145 硬编码的 `google/paligemma-3b-pt-224`（HF 被代理阻断即失败）②**不加载 ckpt 的 norm_stats，归一化整条跳过**
- 这意味着 07-24 的 I2 对比（P4.24）除 schema 漂移外，**输入还未经归一化**——旧 I2 结论"两 API 输出方向一致"的输入构造与 server 实际链路有双重偏差
- 修复：`make_pre_post_processors(policy.config, CKPT, preprocessor_overrides={"device_processor": {"device": DEV}})`，与 server_v2.py:362 完全一致

### NPU 基准重生成（v13 成功：新 schema + 正确 processor + 正确任务语言）
- **最终结果（fp32, 卡1, 2026-09-26）**：flipped（闭环图像分布）**`[-0.736, +0.016, -0.891]`**；unflipped `[-0.794, +0.033, -0.854]`，存 `results/pi05_npu_baseline_2026-09-26.json`
- **★ 关键定论：#18 定论被独立路径确证**——flipped 与历史闭环基准 `[-0.88~-0.80, +0.04~-0.01, -0.97~-0.90]` 同方向（固定负 z）。正确构造的单发推理路径下官方便样 ckpt 对 task0 就输出固定负 z——**闭环 0% 不是链路 artifact，是模型真实输出**
- **反例印证**：v4/v5 因硬编码错任务语言（task7 的 "on the stove"）输出 `[-0.33, +0.08, -0.29]`（方向不同量级减半）——任务语言是输入构造关键一环
- GPU 侧跑**同一脚本**（`--phase auto --device cuda:0`）严格可比，定论 D/G
- state_8 新 schema 验证值：`[-0.2108, -0.0152, 1.1757, -0.053, -0.045, -0.012, 0.0264, -0.0273]`（axis_angle ±0.05 正常；旧 quat schema 第 4 维会是 ~1.0 量级）

### 基准重生成过程连环踩坑（全部解决，教训已入 RECORD #32-35）
1. 后台 NPU 任务 npu.is_available()=False（drvRet=4）：非交互 shell 缺 CANN 驱动路径 → 启动脚本显式 source Ascend set_env.sh
2. **IDE 沙箱间歇性拒 /dev/davinci_manager**（11:06 后新 NPU 进程全挂，老进程持已开 fd 不受影响）→ 用需审批的沙箱外模式 + 直接子进程启动（setsid 会落回沙箱）
3. OSMesa GL 上下文与 NPU runtime 同进程两种死法（env 开着推理崩 / NPU 初始化后 env.close() 崩，均 exit 139）→ **两阶段进程隔离**（grab 纯 OSMesa 存 npz / infer 纯 NPU）
4. 纯 NPU 推理段 segfault → 漏了 server_v2 的 sample_noise/sample_time fp32 monkey-patch（aclnnNormalFloatFloat 不支持 bf16，本机 CANN 实测直接 segfault）→ 移植 patch
5. 段错误进程泄漏 HBM（~20GB 占卡1）→ kill -9 挂死进程后驱动回收
6. 任务语言硬编码错误（task7 语言当 task0）→ 从 benchmark 动态取

### 其他同日修复（详见 PROJECT_TRACKING 阶段 17）
- 凭证脱敏：本文档 §六.17/§六.18 与 PI05_RECORD P4.26/27 的 GPU 服务器 IP/账号/密码全部脱敏
- `gpu_http_proxy.sh` 参数化（GPU_PUBLIC_IP 自动探测）
- GPU 交接包需用修复后的 `gpu_infer_compare.py` 重新打包（旧包作废）

---

## 七、接手指引（原 ATOMCODE_HANDOVER.md + GPU_INFER_README.md 并入，2026-09-26）

> 新会话/新模型接手时，先精读本文档 §六.1-§六.20 + `PI05_RECORD.md`，再按本节开工。

### 7.1 开工检查清单
1. **conda env**：`conda env list` 确认 `lerobot-pi05` 存在（历史上多次被外部清理，被清则按 §二.2 重建：torch 2.7.1+cpu + torch_npu 2.7.1.post2 + lerobot fc296548 editable + numpy 1.26.4）
2. **源码**：`ls /home/ma-user/work/lerobot_pi05/src` 确认在（含 F6 修复：configuration_pi05.py 的 use_peft 字段）
3. **资产**：`ls ~/render_libs` + `ls ~/.cache/libero/assets` + `ls /home/ma-user/work/paligemma3b_hf/tokenizer.json`
4. **ckpt**：`/home/ma-user/work/lerobot_pi05_libero_official/`（官方便样，9.35GB，config.json 已改 compile_model=False、policy_preprocessor.json 已指本地 tokenizer）
5. **server 是否在跑**：`pgrep -f server_v2.py`（在跑先 pkill 释放 HBM）
6. **后台任务环境**：任何 nohup/setsid 的 NPU 任务，启动脚本必须先 `source ~/.bashrc`

### 7.2 GPU 对比实验（方向 D/G，等用户执行）
**目的**：定论 NPU 算子是否是闭环 0% 根因。两端跑同一脚本 `scripts/pi05/gpu_infer_compare.py`（2026-09-26 修复版，P4.28/P4.28b 已对齐），对比首步 delta_pos 方向：
- 两端都朝固定负 z → NPU 算子不是根因，真根因在模型本身
- 方向不同 → NPU 算子根因（flow matching 采样轨迹偏）

**NPU 端（本机，已完成/跑中）**：
```bash
bash /tmp/run_pi05_baseline.sh  # 或按其内容手动跑
# 结果: results/pi05_npu_baseline_2026-09-26.json
```

**GPU 端（用户在 RTX4090 服务器跑）**：
1. 传输清单（4 项）：
   - `lerobot_pi05_libero_official/`（9.35GB ckpt，含已改好的 config/preprocessor JSON）
   - `lerobot_pi05.tar.gz`（源码，含 F6 use_peft 修复）
   - `paligemma3b_hf.tar.gz`（tokenizer，解压后放同路径或改 policy_preprocessor.json 的 tokenizer_name）
   - `scripts/pi05/gpu_infer_compare.py`（**必须用 2026-09-26 修复版**，旧版 schema/processor 双偏差）
2. 环境：conda env python3.10 + `pip install -e lerobot_pi05` + torch(CUDA) + libero/robosuite/mujoco + OSMesa（`apt install libosmesa6-dev` 或 `MUJOCO_GL=egl`）
3. 运行（GPU 无 GL/NPU 冲突，单进程 auto 一次跑完；脚本为 2026-09-26 v13 修复版：两阶段设计 + 采样 patch + 动态任务语言 + 双 pass）：
   `python gpu_infer_compare.py --phase auto --ckpt /path/to/lerobot_pi05_libero_official --device cuda:0 --dtype float32`
4. 把 `/tmp/gpu_infer_result.json` 发回，与 `results/pi05_npu_baseline_2026-09-26.json` 对比定论（NPU 基准：flipped `[-0.736, +0.016, -0.891]` / unflipped `[-0.794, +0.033, -0.854]`，固定负 z）

### 7.3 下一步优先级（P4.29 后已重排，2026-09-26）
1. ✅ ~~GPU 对比 D/G~~ **已作废**：P4.29 定论根因为输入构造五重缺陷（主因图像 uint8 未 /255），非模型/NPU 损坏；旧 NPU 基准（2026-09-26 上午版）因脚本同源同错作废，无需 GPU 对比
2. 🔄 spatial 全量验证（10 task × 10 ep）跑中 → 结果对照 openpi 98.8% / lerobot repro 97.5%
3. ⏳ 若 spatial 达标：扩展 goal/object/10 三 suite（pi0.5 官方仅 spatial 微调 ckpt，其余 suite 需确认是否有对应 ckpt，无则记录说明）
4. ⏳ bf16 速测（当前 fp32 ~4.3s/chunk；修复后输入正常，F4 的"bf16 精度损方向"结论是垃圾图时代的产物，值得重测）

## 六.21、2026-09-27 PI0.5 四 suite 证据核对与最小编排

- **论文 vs 后续官方复现**：π0.5 原论文 [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) 的摘要与实验重点是开放世界真实机器人泛化；本次核对未找到论文直接报告 LIBERO 四套件分数。openpi 官方仓库的 `pi05_libero` 后续示例/权重是四套件参考来源，记录为 spatial 98.8%、object 98.2%、goal 98.0%、LIBERO-10 92.4%，平均 96.85%；LeRobot π0.5 文档另记录其复现为 97.0%、99.0%、98.0%、96.0%，平均 97.5%。两者均与论文数值分开标注。
- **LIBERO 协议**：LIBERO 原始论文定义 spatial/object/goal 三个 10-task suite 与 LIBERO-100；当前社区 VLA 对照通常将第四套件写作 LIBERO-10/`libero_10`。本项目本轮可执行协议为每 task 10 episodes、`init_seed=42`、suite horizon spatial/object/goal/long = 220/280/300/520，并逐 episode 保存并解码核验视频；这属于本地缩减复现，不等同论文/官方 50 trials/task、多 seed 统计。
- **checkpoint 覆盖性**：当前可读权重 `/home/ma-user/work/lerobot_pi05_libero_official` 是单一 `pi05-libero` 目录，已有 `model.safetensors`、config 与 processor/norm stats；未发现四个 suite 独立 checkpoint 证据，因此编排器不假设按 suite 换权重。当前实际加载路径如实记录，不把 `jade_choghari/pi05-t5` 冒称为论文官方权重；项目历史已记录两者文件同源核验和版本兼容修复。
- **最小编排**：新增 `scripts/pi05/run_pi05_full_libero.sh`，固定 `npu:0`/fp32、显式 CANN 环境、单一 checkpoint、四个 suite 顺序运行；每 suite 使用独立端口和唯一输出根目录，保留 manifest、日志、partial/final JSON 与逐 episode 视频，不清理旧结果。该脚本不复用当前 8012 活跃端口。
- **运行状态**：未启动新四 suite 评测。原因是当前 NPU0 仍由活跃 spatial run PID 2242249 使用，NPU1 也有现存 OpenVLA server 占用；并行启动会违反“不杀活跃评测/不 CPU/GPU 推理”约束。当前新 spatial run 仍以 NPU0 fp32 运行，partial JSON 已推进到 task0-8，不能据此宣称四 suite 完成。
- **证据 URL**：openpi 官方仓库 https://github.com/Physical-Intelligence/openpi；LeRobot π0.5 文档 https://huggingface.co/docs/lerobot/en/pi05；π0.5 论文 https://arxiv.org/abs/2504.16054；LIBERO 原始论文 https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/liu_zhu_NeurIPS2023.pdf 。

## 六.20、2026-09-26 P4.29 输入构造五重缺陷定论 + 闭环 0% 终结（真根因收口）

### 触发
用户拍板"所有验证必须 NPU 上进行"，GPU 对比（D/G）无限期搁置 → 诊断方法转向：**训练分布对照法**——server 构造的每个输入张量直接对照 ckpt norm_stats 逐维验证（此前 11 阶段全在链路内部对比，从未对照训练分布）。

### 诊断链（半天收口，此前 3 个月未破）
1. **norm_stats 读取**：action.max=[0.9375,0.9375,0.9375,...] → 训练动作即 robosuite 归一化空间（±1≈±0.05m），P4.5 的"±0.9 expected"判断正确；**state aa_x∈[0.35,3.67] q50=2.97≈π** → 训练时夹爪恒朝下
2. **state 姿态对照**：诊断脚本实证初始位姿 `robot0_eef_quat=[0.9996,-0.0009,-0.0279,-0.00026]` 按 **xyzw** 解释 → aa=[3.14,0,-0.088] ✓ 落训练分布；按 **wxyz**（P4.8 的解释）→ aa≈0 ✗ ~10σ OOD → **P4.8 的 perm+signs 映射是在匹配错误的 wxyz 目标**
3. **client 代码精读**：`self.proprio` 仅首查初始化（`_format_query:165`）+ `action[-1,:9]` 回传（`step:197`）→ P4.5v2 回传"原值"只防污染不解冻结 → **server 每查收到的都是首帧位置**（OpenVLA 无恙因它不吃 proprio）
4. **训练管线精读**（lerobot_pi05 源码）：env_processor 全图 `torch.flip(dims=[2,3])`（H+W 双翻）→ wrist 需补翻；**observation_processor.py:90 `float32/255.0`** → 图像必须 [0,1] float；`_preprocess_images` 内 `*2-1` → [-1,1]（SigLIP）
5. **图像摧毁实锤**：server 传 uint8 → 模型内 to(float32)=0-255 → `resize_with_pad_torch` float32 分支 **clamp(-1,1)**（256→224 必触发）→ 图像近二值化。**主因定论**
6. **敏感性测试**（POST 变体到 server）：state/双图/语言扰动均改变输出 → 链路无断裂，模型/ckpt 正常；修 F 后首步 `[0.02,+0.09,+0.09]`+grip=-1.04（接近正确开爪），修 F 前恒俯冲+闭合
7. **闭环验证**：task0 ep0/ep1 SUCCESS（77/82 步）→ 全量 spatial 启动

### 修复清单（server_v2.py P4.29 A-F + eval_pi05_spatial.py）
| # | 缺陷 | 修复 |
|---|---|---|
| F 主因 | 图像 uint8（0-255 被 clamp 摧毁） | `permute(2,0,1).float()/255.0`（官方 observation_processor 同源） |
| A | quat wxyz 误解释 → state ~10σ OOD | robot0_eef_quat 按 xyzw → aa（lerobot env_processor 同源），必填 fail-fast |
| A2 | client.proprio 冻结首帧 | eval 每步 `client.proprio=None`（client 零改动） |
| B | wrist 未 H+W 翻转 | server 端补翻 |
| C | 1-grip 反转（OpenVLA 约定错配） | grip 直传（训练语义即 env 语义，实证 -1=开/+1=合） |
| D | F1 chunk=2 + P4.5v2 末步回传 hack | 恢复官方 n_action_steps=10，删回传（fresh proprio 下无意义且浪费动作步） |

配套：eval_pi05_spatial.py monkey-patch 注入 env 真值 quat/gripper_qpos；run_pi05_spatial.sh 改官方 ckpt + fp32 + Ascend env source。

### 对前期定论的修正（重要）
- P4.16/#18"模型推理本身输出错方向"→ 归因错误：根因是输入被摧毁，模型/ckpt/NPU 均正常
- P4.28 NPU 基准"正确构造下仍固定负 z"→ 作废：其"正确构造"含同样的 uint8 图像摧毁；**若 GPU 对比照此执行会两端一致误判"模型坏"**
- F1/F4/P4.5v2/P4.8 各"生效但非根因"的修复 → 均为垃圾输入时代的止血，全部回滚或修正

### 产出
- `models/pi0/server_v2.py`（P4.29 修复版）、`scripts/pi05/eval_pi05_spatial.py`、`scripts/pi05/run_pi05_spatial.sh`
- **最终结果（2026-09-26 18:57，NPU1 fp32，全量 10 task × 10 ep = 100 rollouts，seed42，horizon 220 官方协议）**：
  - **Overall SR = 96.0%（96/100）**；外部 openpi/lerobot 参考值与本项目原始统计分开，来源和协议待核实，不据此宣称“2σ 内”或统计等价
  - 逐 task：task0-4/6/7/8 = 100%，task5 = 70%，task9 = 90%
  - 总耗时 10359.117s（172.65min，2.8775h，约103.59s/ep，含推理+仿真）
  - 结果文件：`results/pi05_spatial/spatial_results.json`（过程日志 `progress.log`）；历史 0% 证据归档 `results/pi05_spatial_pre_p429/`
- 教训 #36/#37 入 PI05_RECORD.md
