# G0.5 NPU 推理服务说明

G0.5 推理服务使用官方 GalaxeaVLA 仓库代码，NPU 修改保存在本目录的 `g05_npu.patch`，基线 commit 为 `89f2322`。本项目不复制官方模型源码和权重。

- 官方仓库：`$G05_ROOT`（示例：`$HOME/work/GalaxeaVLA`）
- 推理入口：官方仓库的 `scripts/serve_policy_batched.py`（WebSocket + msgpack，批量推理）
- LIBERO checkpoint：`$G05_CKPT_ROOT/g05-libero/`；`model.pt` 为 11,440,381,065 bytes，仓库内 `checkpoints` 软链指向 `$G05_CKPT_ROOT`
- 验证环境：conda `g05`，Python 3.10.20、torch 2.7.1+cpu、torch_npu 2.7.1.post2、transformers 4.57.1；CUDA 原生扩展未安装

## NPU 修改

| 官方仓库位置 | 修改 | 原因 |
|---|---|---|
| `src/g05/models/g05/g05_model_qwen35.py` | 9 维图像 patch 变换分解为最多 8 维的等价操作；与原输出逐位比对通过 | 昇腾 ACL 算子最多支持 8 维张量 |
| `src/g05/models/g05/inferencer.py` | 按设备选择 autocast，支持关闭 bf16 | CUDA autocast 不作用于 NPU 张量 |
| `configs/model/g05.yaml` | 线性注意力后端设为 `torch` | 使用官方纯 PyTorch 回退 |
| `gated_deltanet.py`、`checkpoint_utils.py` 等 | 修正 autocast 与缓存操作的设备选择 | 避免依赖 CUDA |
| `scripts/serve_policy_batched.py` | 支持 `--device npu:0` 和 `--no-bf16` | NPU fp32 验证 |

ViT 注意力使用官方 SDPA 回退；Gated DeltaNet 使用官方纯 PyTorch 实现；ActionCodec 在 fp32 下使用 SDPA。补丁包含 12 个已修改的官方源码文件，不包含 checkpoint、内核缓存或访问令牌。

## 应用补丁

在单独的、位于基线 commit `89f2322` 的干净 GalaxeaVLA 工作区中运行：

```bash
git apply --check "$PROJECT_ROOT/models/g05/g05_npu.patch"
git apply "$PROJECT_ROOT/models/g05/g05_npu.patch"
```

当前机器的 `~/work/GalaxeaVLA` 已应用这些修改，无需重复执行。更换上游版本时需重新核对补丁，不应直接应用到其他 commit。安装推理依赖时使用 torch-npu，排除 flash-attn、flash-linear-attention、liger-kernel、bitsandbytes、deepspeed 等 CUDA 依赖；LIBERO 权重与 processor 仍从官方受限仓库按其许可证获取。

## 启动服务

```bash
cd "$G05_ROOT"
source "$ASCEND_TOOLKIT_ROOT/set_env.sh"
source "$CONDA_ROOT/etc/profile.d/conda.sh" && conda activate g05
export ASCEND_RT_VISIBLE_DEVICES=0
python scripts/serve_policy_batched.py \
  --ckpt_path checkpoints/g05-libero/model.pt \
  --host 0.0.0.0 --port 8765 \
  eval_embodiment=libero --action_steps 10 \
  --device npu:0 --no-bf16 --max_batch_size 10 --max_wait_ms 500
```

完整评测入口在 `scripts/g05/run_g05_full_libero.sh`，迁移过程与结果记录在 `docs/g05/G05_TRACKING.md`。
