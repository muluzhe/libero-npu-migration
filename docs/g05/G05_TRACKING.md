# G0.5 NPU 迁移研究追踪

> 状态：G0.5 四套件 NPU 全量运行已完成；结果、日志和视频产物已保存。独立只读核验确认四套件各 500 个非空 MP4（共 2000/2000），每段视频的流、正时长及首帧实际解码均通过；四套件各抽首/中/尾三段多帧解码通过（12/12）。未对每部视频完整逐帧解码。最多保留两个专项：官方资料核对、NPU 迁移计划。
> 更新：2026-09-29。本文第 1、2 节的早期“尚未验证/明确阻塞”内容是 2026-09-28 前的历史快照；后文“权重获取与 NPU 迁移执行”记录了实际完成情况。当前结果以第 5 节为准：四套件 1977/2000，视频首帧 2000/2000 可解码，抽样多帧 12/12 通过。

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


## 权重获取与 NPU 迁移执行（2026-09-28）

> 前文“明确阻塞”中的权重访问阻塞已解除；CUDA 依赖阻塞经代码核实为推理链路可回退。

### 1. 权重获取（hf-mirror + 授权 token）

- `OpenGalaxea/G05` 为 gated:auto 仓库，匿名经 hf-mirror 返回 403；在完成必要授权后，经 `HF_ENDPOINT=https://hf-mirror.com` 下载成功。
- 下载范围（仅 LIBERO 相关）：`g05-libero/*`、`action_tokenizer.pt`、`qwen3_5_2b_base_processor/*`、`licenses/*`、`README.md`，共 16 文件。
- 关键校验：`model.pt` = 11,440,381,065 bytes，与 HF API 元数据完全一致；`action_tokenizer.pt` 506,886,775 bytes；`dataset_stats.json` 95,006 bytes；processor 8 文件齐全。
- bundle 组装（按官方 `experiments/libero/README.md` 布局）：`action_tokenizer.pt` 与 `hf_processor -> qwen3_5_2b_base_processor` 以软链放入 `g05-libero/`；`GalaxeaVLA/checkpoints -> g05_libero_ckpt` 软链使 hydra config 内相对路径 `checkpoints/g05-libero/...` 可解析。
- 澄清：HF 上 `action_tokenizer.pt` 与 processor 位于仓库根目录而非 `g05-libero/` 内，官方 README 要求软链进 bundle；`hf_processor/` 是软链后的本地名。

### 2. NPU 推理环境（conda env `g05`）

- Python 3.10.20、torch 2.7.1+cpu、torch_npu 2.7.1.post2、transformers 4.57.1（官方要求版本）、numpy 1.26.4。
- CUDA 原生扩展全部未安装：flash-attn-4、flash-linear-attention、causal-conv1d、liger-kernel、bitsandbytes、deepspeed。
- 代码核实（推理链路）：ViT 注意力走 SDPA 回退（vision.py 官方自带）；Gated DeltaNet 走纯 PyTorch chunk/recurrent 回退（gated_deltanet.py 官方自带，`linear_attn_backend: torch`）；ActionCodec fp32 天然走 SDPA；`inferencer.py` autocast 已 device-aware（cuda/npu），`--no-bf16` 为 fp32 验证模式。
- 客户端依赖补齐：hf_libero 0.1.4、robosuite 1.4.0、mujoco 3.3.2、bddl 1.0.1、gym 0.22.0、rootutils、websockets 16.0、msgpack；修复 cffi 被 ModelArts cp39 版本劫持问题（g05 环境安装 cffi 2.1.1 cp310）。
- server 与客户端同环境运行，OSMesa 软件渲染（MUJOCO_GL=osmesa 等，与 PI0.5 链路同源）。

### 3. NPU 算子问题与修复（EZ1001）

- 现象：首次推理报 `AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.`，定位到 `g05_model_qwen35.py` `_forward_vision` 的图像 patch 化：原实现为 9 维 `reshape(K,t,C,gh2,m,p,gw2,m,p).permute(0,3,6,4,7,2,1,5,8)`，超过昇腾 ACL 算子 8 维上限。
- 修复：分解为等价低维链（每步 <=8 维）。关键点：原 reshape 含 `(C,t)->(t,C)` 行主序重切（通道/时间错位），该语义是模型训练时实际使用的数据流，必须逐位复现而非"修正"；新链以 `.reshape(K, C*t, H, W).reshape(K, t, C, H, W)` 显式复现该重切。
- 等价性验证：随机张量下与原 9 维实现逐位对比，K=2/56x56 与 K=4/224x224 均 bitwise equal（max abs diff = 0.0）。

### 4. 冒烟验证（NPU0 fp32，seed 42）

- 单环境冒烟：libero_spatial task0 x 1 trial，SUCCESS（1/1），71 步完成，视频 71 帧解码核验通过（256x512 双相机拼接），infer_calls=8 与 10-step chunk 模式吻合，约 2.3 min/episode。
- 并行冒烟：task0 x 5 trials x num_parallel 5，成功率 100%（5/5），server 批量前向 batch_size=4-5 正常，5 个视频全部生成，总耗时 2.5 min。
- 结果目录：`results/g05/smoke_20260928/`（单环境）、`results/g05/smoke_parallel/`（并行）。

### 5. 四 suite 全量验证（2026-09-28 完成）

- 编排脚本：`scripts/g05/run_g05_full_libero.sh`（server 全 suite 复用 + 四 suite 串行客户端）。
- 协议：对齐官方——4 suite x 10 task x 50 trials = 2000 episodes，horizon 220/280/300/520（官方硬编码），num_parallel 10、max_batch_size 10、seed 42、`--save_videos` 强制视频；manifest 另记录 fp32、action_steps=10、chunk10。
- 结果目录：`results/g05/full_libero/20260928_122503_1236126/`。spatial 496/500（99.2%），object 500/500（100.0%），goal 490/500（98.0%），libero_10 491/500（98.2%），合计 1977/2000（98.85%，按一位小数为 98.9%）。每个 suite 的 JSON 均为 10 tasks x 50 trials。
- NPU 证据：`server.log` 记录 `loaded on npu:0` 与 `client connected`；末尾有 TBE task_distribute main process disappeared 和 resource_tracker 30 leaked semaphore 的收尾提示，结果均已保存。
- 视频独立只读核验：四个 suite 各有 500 个非空 MP4，共 2000/2000；`ffprobe` 检查视频流与正时长全部合格，`ffmpeg` 对每个 MP4 实际解码首帧全部成功；每个 suite 抽首/中/尾三段视频进行多帧解码，合计 12/12 成功。按 suite 统计，MP4 文件名 success/failure 尾缀与原始 JSON 成功/失败数完全一致：spatial 496/4、object 500/0、goal 490/10、libero_10 491/9，合计 1977/23（1977/2000=98.85%）。核验没有对每部视频完整逐帧解码，不能据此声称全部帧无误。
- 官方参考：LIBERO 平均 98.9% 仅作公开参考；其 CUDA bf16 配置与本项目 NPU fp32 + 纯 PyTorch 回退不同，不作统计等价或配置等价断言。
