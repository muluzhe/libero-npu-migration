# models/ 目录说明

本目录存放**各模型在 Ascend NPU 上的推理服务（server）实现**。每个模型一个子目录，与 `scripts/<model>/`（评估与编排脚本）、`results/<model>/`（验证产物）一一对应。

## 目录内容

| 子目录 | 模型 | server 形态 | 说明 |
|---|---|---|---|
| `openvla/` | OpenVLA-7B | 自研 HTTP server（`server_v2.py`） | bf16 + SDPA；每 suite 独立 checkpoint + unnorm_key |
| `pi0/` | PI0.5（lerobot） | 自研 HTTP server（`server_v2.py`，P4.29 修复版） | fp32；图像 /255、quat xyzw→aa 等输入对齐训练分布 |
| `xvla/` | X-VLA | 自研 server（`server.py`） | abs 动作，官方 deploy 链路 |
| `g05/` | G0.5（GalaxeaVLA） | **外部官方代码 + NPU patch**（见 `g05/README.md`） | WebSocket server，AR + ActionCodec |
| `smolvla/`、`act/`、`diffusion_policy/` | 其他 | 未实测骨架 server | 仅供新模型适配参考 |

## 两种 server 形态

1. **自研 server**（openvla/pi0/xvla）：模型权重加载、前后处理、推理全部在本目录的 `server*.py` 中实现；评估器通过 HTTP + json_numpy 与之通信。
2. **外部官方代码 + NPU patch**（g05）：推理主体使用模型官方仓库（`~/work/GalaxeaVLA`），本目录只存放适配说明与 patch 清单（`g05/README.md`），不复制官方代码；所有 NPU 修改在官方仓库工作区完成并记录。

## 新模型适配

按 `docs/ADAPTING_NEW_MODEL.md` 与 `docs/EVAL_PROTOCOL.md` 执行；server 骨架可参考 `scripts/model_server_template.py`。
