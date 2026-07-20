# LIBERO NPU 迁移项目记录文档

> **用途**: 记录项目过程中遇到的关键问题与解决方法，供后续 NPU 迁移项目参考  
> **项目周期**: 2026-07-02 ~ 2026-07-17  
> **硬件**: Ascend 910B4 NPU

---

## 一、环境搭建问题

### 问题1：OSMesa 软件渲染环境

**现象**: LIBERO 仿真需要 OpenGL 渲染，但 NPU 服务器无 GPU 显示输出。

**解决**: 使用 OSMesa 软件渲染 + 环境变量配置：
```bash
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export PYOPENGL_PLATFORM=osmesa
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
```

### 问题2：openvla 官方 eval import segfault

**现象**: `from experiments.robot.libero.libero_utils import ...` 触发 segfault。

**根因**: prismatic → dlimp → tensorflow import 链，tensorflow 在 OSMesa 环境下 segfault。

**解决**: stub 注入绕过（`server_v2.py` line 28-35）：
```python
import importlib.machinery
for m_name in ('dlimp', 'rlds', 'ox_pilot', 'tensorflow', 'tensorflow.data'):
    if m_name not in sys.modules:
        m = types.ModuleType(m_name)
        m.__spec__ = importlib.machinery.ModuleSpec(m_name, None)
        m.__path__ = []
        sys.modules[m_name] = m
```

---

## 二、NPU 推理问题

### 问题3：bf16 模型 + fp32 inputs 报 dtype 不匹配

**现象**: NPU 上 bf16 模型接收 fp32 inputs 报错。

**解决**: inputs 必须用 `infer_dtype=bf16`，保持模型与 inputs dtype 一致。

### 问题4：attention 实现选择

**现象**: 不同 attention 实现性能差异大。

**对比**:
| attn | 单步推理 | 支持性 |
|---|---|---|
| eager | 64.5s | ✅ 慢 |
| sdpa | 22.1s | ✅ 最佳 |
| flash_attention_2 | - | ❌ NPU 不支持 |

**解决**: 用 `OPENVLA_ATTN=sdpa`，性能最优。

### 问题5：bitsandbytes INT8 量化 NPU 不支持

**现象**: `--load_in_8bit` 在 NPU 上报错。

**根因**: bitsandbytes INT8 算子是 CUDA 专用，NPU 无对应实现。

**解决**: 用 bf16 原生推理（7B → ~14GB HBM）。

### 问题6：jp4 kernel 占用 HBM

**现象**: bf16 加载报 OOM。

**根因**: jp4 kernel 临时占用 Chip0 的 25.7GB HBM。

**解决**: 等待 jp4 kernel 释放后再加载（bf16 需 ~14GB）。

---

## 三、动作处理问题

### 问题7：cv2.LANCZOS4 属性名错

**现象**: `cv2.LANCZOS4` 报 AttributeError。

**解决**: 正确属性名是 `cv2.INTER_LANCZOS4`。

### 问题8：Rodrigues 公式与 client 转换链不等价

**现象**: server 用 Rodrigues 公式 `aa→R→取前两列→rot6d`，client 用 `b1/b2 正交化 + quat 中转`，两端转换不等价（实测差值大：aa_in=[0,0,0]→aa_out=[0,0,-0.588]）。

**解决**: server 改用 client 同源 `AxisAngle_to_Rotate6D`（`T.axisangle2quat → T.quat2mat → Mat_to_Rotate6D`），两端差=[0,0,0] 完全等价。

### 问题9：grip 反转

**现象**: 夹爪符号不匹配。

**根因**: 官方链路 `grip_raw → normalize[-1,+1] → invert → env.step`，我们链路 `server 返 1-grip_raw → client >0.5 离散化 → env.step`。

**解决**: server 端 `grip = 1.0 - grip`，匹配 client 的 `>0.5` 离散化逻辑。

---

## 四、闭环执行问题（核心根因）

### 问题10：OpenVLA delta action vs X-VLA abs 路径（根因）

**现象**: 8 个修复 × 3ep 闭环验证全 0%（每 ep 跑满 600 步没 done）。

**根因定位过程**:
1. 推理链路验证: server 输出与官方 action 逐字节一致 → 推理正确
2. 闭环执行链路分析: X-VLA client `step()` 对 pos 不做任何转换
3. **关键发现**: OpenVLA `openvla.py:47` docstring 明示 "end-effector deltas"，输出 delta action
4. **X-VLA client 默认 `act_type="abs"`** → `robot.controller.use_delta = False`
5. **use_delta=False 时**, env.step 把 action 的 `[pos3]` 当**绝对目标坐标**解释
6. OpenVLA 输出 delta pos `[0.096, 0.035, -0.003]` 被当绝对目标坐标 → 机器人瞬间被指令拉到工作空间外的位置 `[0.096, 0.035, -0.003]`（z=-0.003 在 workspace 下方），永远 `done=False`，闭环 0%

**解决方案（路径A，与官方 eval 等价）**:
- 让 client 走 `act_type="rel"` 路径
- env 保持默认 `use_delta=True`
- server 输出的 7 维 delta action 直接 `env.step(action)`
- `pos=[0.096,0.035,-0.003]` 被正确解释为"末端相对移动 (0.096, 0.035, -0.003)"

**验证结果**: task0 闭环 100% (3/3)，完整 spatial suite 76.0% (38/50)。

---

## 五、性能优化问题

### 问题11：每步推理重复开销

**现象**: 优化前每步推理 ~1.5s，50 rollouts × ~87步 × 1.5s ≈ 148min。

**优化点**:
1. **全局预计算 crop 参数**: `_CROP_H`, `_CROP_Y0` 等常量模块级计算，避免每步 `np.sqrt` + `int`
2. **固定 seed 只设一次**: `do_sample=False` 时推理确定，移除每步 `torch.manual_seed` 重复调用
3. **`json_numpy.patch()` 只调一次**: 移除每步重复 patch
4. **图像预处理合并**: resize → PIL → crop → resize 合并为更紧凑的流程

**优化效果**: 单步推理 1.5s → 0.45s（**3.3x 加速**），优化版验证 100% SR (3/3), 6.2min。

---

## 六、文件管理问题

### 问题12：脚本命名混乱

**现象**: `test_openvla闭环.py`、`test_openvla闭环_v2.py`、`test_openvla闭环_v3_optimized.py` 命名混乱。

**解决**: 
- 删除旧版 `test_openvla闭环.py`（根因定位前版本，已废弃）
- 重命名 `test_openvla闭环_v2.py` → `eval_spatial_task0_v2.py`（task0 修复版验证）
- 重命名 `test_openvla闭环_v3_optimized.py` → `eval_spatial_task0_v3_optimized.py`（优化版验证）

---

## 七、X-VLA 验证问题

### 问题13：X-VLA client 每 10 步推理 1 次

**现象**: X-VLA 设计为 chunk action（一次推理 10 步缓存），但 OpenVLA 输出绝对 action 每步都该推理。

**解决**: monkey-patch `policy._format_query` 强制 `steps=1`，每步推理。

### 问题14：libero assets 路径

**现象**: LIBERO assets 找不到。

**解决**: 正确路径是 `libero/libero/assets`（嵌套），缓存 `~/.cache/libero/assets`。

---

## 八、问题解决方法总结

| # | 问题类别 | 核心方法 |
|---|---|---|
| 1 | 环境搭建 | OSMesa 软件渲染 + 环境变量 |
| 2 | import segfault | stub 注入绕过 |
| 3 | NPU dtype | bf16 模型 + bf16 inputs |
| 4 | attn 选择 | sdpa 最佳（eager 慢 3x, flash_2 不支持） |
| 5 | INT8 量化 | NPU 不支持，用 bf16 原生 |
| 6 | HBM 占用 | 等 jp4 kernel 释放 |
| 7 | cv2 属性名 | `cv2.INTER_LANCZOS4` |
| 8 | rot6d 转换 | 用 client 同源 `AxisAngle_to_Rotate6D` |
| 9 | grip 反转 | server 端 `grip = 1.0 - grip` |
| 10 | **delta action 根因** | **act_type="rel" → use_delta=True** |
| 11 | 性能优化 | 全局预计算 + 固定 seed + patch 一次 |
| 12 | 文件命名 | 规范化为 `eval_*` 格式 |
| 13 | chunk action | monkey-patch steps=1 |
| 14 | assets 路径 | `libero/libero/assets`（嵌套） |

---

## 九、关键教训

1. **推理正确 ≠ 闭环正确**: OpenVLA 推理与官方逐字节一致，但闭环 0%，根因在 delta/abs 参考系不匹配。
2. **docstring 是金矿**: `openvla.py:47` 的 "end-effector deltas" docstring 直接指向根因。
3. **client 框架假设**: X-VLA client 假设 abs action，迁移 delta action 模型时必须显式切换 `act_type="rel"`。
4. **NPU 量化限制**: bitsandbytes INT8/4bit 是 CUDA 专用，NPU 只能用 bf16 原生推理。
5. **性能优化优先级**: 固定开销（seed、patch、预计算）> 算法优化（chunk），因为 delta action 不能简单 chunk 缓存。
