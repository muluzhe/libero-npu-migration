# G0.5 NPU 迁移研究追踪

> 状态：read-only 官方资料研究与迁移准备；本次未下载权重、未改代码、未启动评测，因此不能声称已完成 G0.5 NPU 验证。最多保留两个专项：官方资料核对、NPU 迁移计划。
> 更新：2026-09-27

## 1. 官方资料核对

### 正式名称与公开入口

- 正式名称：**G0.5: One Autoregressive Stream for Robot Reasoning and Action**；项目页将其描述为 Galaxea 的预训练自回归 VLA。
- 官方代码仓库：<https://github.com/OpenGalaxea/GalaxeaVLA>
- 官方项目页/技术报告：<https://opengalaxea.github.io/G05/>、<https://arxiv.org/abs/2608.11739>
- 官方模型组织：<https://huggingface.co/OpenGalaxea>
- 官方模型仓库：<https://huggingface.co/OpenGalaxea/G05>

### 架构、规模、dtype 与后端

- 架构：以 **Qwen3.5-2B** 为初始化基础；单一自回归 transformer decoder 在同一 token 流中生成可选 CoT 与动作 token；ActionCodec 将跨 embodiment 动作编码到 27D 统一布局。项目页还说明包含视觉记忆和多视角输入。
- 公开配置 `configs/model/g05.yaml` 显示 VLM hidden size 2048、24 层，action expert hidden size 1024、24 层；模型 action_dim=20、proprio_dim=20。该配置不是“参数量等于 2B”的独立核验，暂不把模型总参数量写成更精确数字。
- 官方单 GPU 推理要求 >8GB，推荐 RTX 4090；全量微调要求 >70GB。官方环境为 Python 3.10、CUDA 12.8、PyTorch 2.7.1，并明确依赖 CUDA 原生扩展 `flash-attn-4`、`flash-linear-attention`。
- 配置默认 `model_weights_to_bf16=false`、训练启用 bf16；LIBERO 运行命令没有给出 Ascend/NPU 后端说明。当前 NPU 可行性不能由官方 CUDA 运行要求直接推断。
- 迁移重点算子/依赖：Qwen3.5 视觉模块、linear attention backend `fla`、ActionCodec/RVQ tokenizer、10-step flow-matching action expert、CUDA 原生 flash/linear-attention 扩展、WebSocket/msgpack server-client。需逐项做 NPU 算子覆盖与替代方案核验。

### LIBERO checkpoint 与评测协议

- 官方仓库明确提供 `g05-libero`，权重文件为 `g05-libero/model.pt`；旁车要求包括 `action_tokenizer.pt`、`dataset_stats.json`、`.hydra/config.yaml` 和 `hf_processor/`。
- 官方 LIBERO 入口：`bash scripts/run/eval_libero.sh <ckpt_path>`。默认四套件：`libero_goal`、`libero_spatial`、`libero_object`、`libero_10`。
- 默认协议：`--num_trials 50`（每 task 50 trials）、`--num_parallel 10`、`--num_steps_wait 20`、`--action_steps 10`；官方 README 还给出 episode horizon：spatial 220、object 280、goal 300、LIBERO-10 520。支持 `--seed`，但默认命令未固定 seed；正式 NPU 复验前必须显式记录 seed、环境版本、suite/task 数和 horizon。
- 官方项目页报告 LIBERO 98.9%，但当前只作为官方公开摘要数字记录，不能替代本项目逐 task/逐 trial 原始证据；不能将其他 checkpoint 的结果冒充 G0.5 LIBERO 全量结果。
- `g05-libero` 是独立的 LIBERO 微调/评测 bundle。`g05-base`、`g05-droid`、`g05-so101`、`g05-robotwin20` 均不能替代 LIBERO checkpoint；尤其真实机器人或 DROID/RoboTwin checkpoint 不能冒称全量 LIBERO。

### 权重、许可与可下载状态

- HF model card 列出 `g05-libero/model.pt`，说明官方 LIBERO checkpoint 确实被公开声明；但本次 WebFetch 访问该子目录返回“需要登录并同意共享联系信息”，因此**本次未确认无需登录即可下载，也未下载任何权重**。
- 代码和模型采用按提交时间分界的 **G0.5 Community License**；2026-06-16 之后的材料适用 G0.5 Community License，非商业用途可使用/复制/修改/分发，商业使用需另行商业许可。Qwen3.5 上游材料按仓库说明采用 Apache-2.0。发布前仍需人工阅读并随附官方许可证、NOTICE 和修改声明。
- 模型仓库标注 `g05-community-license`；不因仓库可见就推断已获商业使用权或无需接受 HF 条件。

## 2. NPU 迁移可行性与后续条件

### 当前判断

**技术上有条件可行，尚未验证。** 本项目已有 Ascend 910B4 + torch_npu + OSMesa + LIBERO 基础设施，但 G0.5 官方路径以 CUDA 12.8 和 CUDA 原生扩展为前提，且其自回归/ActionCodec/视觉与 attention 代码与现有 X-VLA/OpenVLA/PI0.5 适配链路不同，不能直接复用现有 server 或动作 schema。

### 建议顺序（上传完成后执行）

1. 只取得官方 `g05-libero` bundle 及仓库对应 commit，记录 SHA256、许可证、配置和所有旁车文件；若 HF 条件或网络不可用，记录阻塞，不用其他机器人 checkpoint 替代。
2. 在独立环境做 import/权重加载 smoke test：确认 Python/PyTorch/torch_npu 版本，禁用或替换 CUDA-only `flash-attn-4`/`flash-linear-attention`，逐项记录 NPU 支持情况；优先尝试 eager/SDPA 或项目源码允许的等价后端。
3. 先跑单 task、单 trial、固定 seed 的离线 LIBERO 闭环，核对三路 RGB、20D proprio、ActionCodec 解码、动作维度、gripper 语义、10-step chunk 和 horizon；保存 server/client 日志及视频。
4. 通过单 task smoke 后，再按官方四 suite 顺序运行全量：每 task 50 trials、固定 seed、官方 horizon 220/280/300/520、`--save_videos`，逐 suite 保存原始 JSON、配置 manifest、环境信息和视频核验记录。
5. 最终报告分别给出 NPU 原始统计、官方 98.9% 参考值、协议差异、失败 task 与后端替代项；在四 suite 全部完成前，不写“G0.5 全量 NPU 验证通过”。

### 明确阻塞

- **权重访问阻塞**：官方 HF `G05/g05-libero` 文件页需要登录并同意共享联系信息，本次未授权访问，无法确认实际下载链路。
- **NPU 后端阻塞**：官方安装要求 CUDA 12.8 及 `flash-attn-4`、`flash-linear-attention` 原生扩展，官方资料未给出 torch_npu 支持矩阵或 NPU 替代实现。
- **资源与协议阻塞**：官方默认四 suite × 每 task 50 trials，且需保存视频；当前项目有 PI0.5/OpenVLA 并发资源，必须等上传完成并安排独立 NPU、端口、输出目录后再运行。
- **证据边界**：本次只有官方公开资料核对，没有权重加载、单 task smoke 或任何 LIBERO rollout；因此状态是“研究完成、验证未开始”。

## 可信来源

- [G0.5 官方项目页](https://opengalaxea.github.io/G05/)
- [OpenGalaxea/GalaxeaVLA 官方仓库](https://github.com/OpenGalaxea/GalaxeaVLA)
- [官方 LIBERO README](https://raw.githubusercontent.com/OpenGalaxea/GalaxeaVLA/main/experiments/libero/README.md)
- [官方 LIBERO 评测脚本](https://raw.githubusercontent.com/OpenGalaxea/GalaxeaVLA/main/scripts/run/eval_libero.sh)
- [官方 G0.5 模型配置](https://raw.githubusercontent.com/OpenGalaxea/GalaxeaVLA/main/configs/model/g05.yaml)
- [OpenGalaxea/G05 HF model card](https://huggingface.co/OpenGalaxea/G05)
- [G0.5 论文](https://arxiv.org/abs/2608.11739)
- [G0.5 官方许可证入口](https://github.com/OpenGalaxea/GalaxeaVLA/blob/main/licenses/LICENSE-G0.5)
