# 扩展到其他 VLA 模型

本方案的仿真层（OSMesa 渲染 + robosuite patch + LIBERO 客户端）完全可复用，只需替换 `models/<model>/server.py` 的模型加载部分。

## 通用迁移步骤

### 1. 编写模型服务器（约 50 行）

在 `models/<your_model>/server.py` 实现 `/act` 端点，返回 `[T, action_dim]` 动作数组：

```python
@app.post("/act")
def act(req: dict):
    img = json_numpy.loads(req["image0"])  # 256x256x3 uint8
    # 1. 图像预处理（PIL → processor）
    # 2. 加载到 NPU: tensor.to("npu:0", dtype=torch.float32)
    # 3. 模型推理: action = model.predict(...)
    # 4. 返回 {"action": action.tolist()}  # shape [T, action_dim]
```

### 2. NPU 加载模型（通用模式）

```python
import torch_npu
device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
model = Model.from_pretrained(
    path, torch_dtype=torch.float32,        # NPU 默认 fp32
    attn_implementation="eager",            # 规避 flash_attn
).to(device)
```

### 3. 运行验证

```bash
bash scripts/run_eval.sh <your_model> /path/to/model ./results 10
```

## 各模型差异点

| 模型 | 动作格式 | 需额外处理 | NPU 算子风险 |
|---|---|---|---|
| X-VLA | `[pos3, rot6d, grip1]` | rot6d→axis-angle | 低 |
| OpenVLA | 离散 token | bin_centers 反归一化 | 低 |
| PI0 | `[pos3, aa3, grip1]` | flow-matching 采样 | 中（需验证采样循环） |
| SmolVLA | 离散 token | 同 OpenVLA | 低 |
| ACT | action chunk | CVAE 编码 | 中（卷积算子） |
| Diffusion Policy | DDPM 采样 | 50步去噪 | 中（需 eager attn） |

## 仿真层复用清单（所有模型通用）

以下文件/配置对所有 VLA 模型相同，无需重复开发：
- `patches/robosuite_osmesa_render.py`（OSMesa 渲染 + GL 坐标修复）
- `patches/robosuite_mj_fullM.py`（mujoco 3.10 API）
- `scripts/setup_env.sh`（环境搭建）
- LIBERO 客户端（`libero_client.py` 官方原样）
- 环境变量（`NUMBA_DISABLE_JIT` / `MUJOCO_GL=osmesa` 等）

## 已验证模型

- **X-VLA**: ✅ libero_spatial 90% 成功率，视频方向正确
- 其他模型: 待社区验证（服务器代码已提供模板）
