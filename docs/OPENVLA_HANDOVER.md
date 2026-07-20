# OpenVLA LIBERO NPU验证交接文档

> **创建时间**: 2026-07-17 17:23  
> **最后更新**: 2026-07-17 18:15（✅ 闭环验证成功 100%）  
> **目标**: OpenVLA在Ascend 910B4 NPU上闭环LIBERO仿真验证（spatial suite）  
> **当前状态**: ✅ **闭环验证成功** — spatial task0 3ep × 100% 成功率，根因（pos 参考系 delta vs abs）已定位并修复  
> **X-VLA项目完好**: 5个seed结果完好（{spatial:0.9, goal:0.99, object:1.0, long:0.94}），未受影响

---

## 1. 关键文件清单

| 文件 | 说明 | 状态 |
|---|---|---|
| `models/openvla/server_v2.py` (215行) | OpenVLA NPU推理服务器，bf16+sdpa+npu:0 | ✅ 推理正确 |
| `scripts/test_openvla闭环.py` | 3ep闭环验证脚本，含steps=1 monkey-patch | ✅ 跑通但0% |
| `scripts/diag_stage1_official.py` | 官方action推理对比诊断（绕过import segfault） | ✅ 已验证推理一致 |
| `scripts/run_openvla_libero.sh` | 一键验证脚本（路径B，因import segfault弃用） | ⚠️ 弃用 |
| `docs/PROJECT_TRACKING.md` (589行) | 完整项目追踪文档 | ✅ 持续更新中 |
| `/tmp/diag_official_action.json` | 官方action对比基准文件 | ✅ 存在(448B) |
| `$OPENVLA_ROOT/` (2.1M) | openvla仓库（含NPU适配+stub注入） | ✅ |
| `$OPENVLA_ROOT_checkpoints/libero-spatial/` (15G) | libero_spatial微调权重 | ✅ 完整 |

## 2. 已确认正确的部分（无需再查）

### ✅ 推理链路完全正确

**对比诊断证据**（`diag_stage1_official.py`绕过import segfault直接调官方`predict_action`）：
- 官方action(7维): `[0.096, 0.035, -0.003, ~0, ~0, ~0, 0.996]`
- 我们server(10维): `[0.096, 0.035, -0.003, 1,-0,0, 1,-0,0, 0.996]`
- **逐字节一致**: pos3差=[0,0,0] rot6d差=[0,0,0,0,0,0] grip差=0

### ✅ NPU推理配置最优

| 配置 | 值 | 说明 |
|---|---|---|
| device | npu:0 | Ascend 910B4 |
| dtype | bf16 | 原生NPU推理 |
| attn | sdpa | **最佳**（eager 64.5s→sdpa 22.1s，快3倍；flash_attention_2不支持） |
| 量化 | 无 | bf16原生（INT8 bitsandbytes是CUDA专用NPU不支持） |

### ✅ 8个修复已实施（server_v2.py关键点23处）

| # | 修复 | 文件 |
|---|---|---|
| 1 | 夹爪normalize+invert（后被grip反转替代） | server_v2.py |
| 2 | aa3→rot6d转换（非quat→rot6d） | server_v2.py |
| 3 | prompt格式包装（`In: What action...\nOut:`） | server_v2.py |
| 4 | center_crop scale=0.9 + num_steps_wait=10 | server_v2.py + test脚本 |
| 5 | cv2.INTER_LANCZOS4（属性名修复） | server_v2.py |
| 6 | resize224（传processor前先resize到224×224） | server_v2.py |
| 7 | grip反转（`1-grip_raw`匹配client的`>0.5`离散化） | server_v2.py |
| 8 | **同源转换链**（用client的`AxisAngle_to_Rotate6D`替代Rodrigues） | server_v2.py |
| 9 | steps=1 monkey-patch（每步推理替代client每10步推理1次） | test脚本 |

### ✅ 已排除的根因（均实测确认非元凶）

1. 图像视角：`_flip_agentview`=np.flip(np.flip(img,0),1)与官方`img[::-1,::-1]`**完全等价**（实测True）
2. 夹爪后处理：normalize+invert匹配官方
3. aa3转换：用client同源`AxisAngle_to_Rotate6D`两端差=[0,0,0]完全等价
4. prompt格式：`In: What action should the robot take to {lang}?\nOut:`
5. center_crop：scale=0.9 sqrt（匹配官方训练图像增强）
6. resize224：lanczos3插值
7. num_steps_wait：10步dummy action让物体稳定
8. bf16精度：bf16模型+bf16 inputs（fp32 inputs在NPU上报dtype不匹配）
9. 推理链路：与官方action逐字节一致

## 3. 当前阻塞：0%成功率根因在闭环执行

**8个修复×3ep闭环验证全0%**（每ep跑满600步没done）。

### 剩余3个疑点（推理正确但闭环0%说明根因在闭环执行链路）

#### ❓ 疑点1：pos参考系（最可能）

**证据**：
- 单测pos=[0.096, 0.035, -0.003]但当前ee_pos=[-0.211, -0.011, 1.174]
- pos在反归一化范围x[-0.75,0.94]y[-0.66,0.88]z[-0.94,0.93]内但离ee_pos远

**可能根因**：
- OpenVLA输出绝对目标pos但参考系与env不一致
- 或X-VLA client把绝对pos当delta处理（或反之）
- 官方eval用`env.step(action.tolist())`直接给env，我们走client的`step()`→processor转换

**下一步诊断**：
1. 查X-VLA client的`step()`怎么处理pos（是否做坐标系转换或delta/绝对判定）
2. 对比官方eval的`env.step(action.tolist())`与我们client的`env.step(action)`接收action的差异
3. 单步跑env.step(action)，打印env内机器人实际移动到的位置，对比action的pos值

#### ❓ 疑点2：env.step接收的action坐标系

**证据**：官方eval用`env.step(action.tolist())`直接给env（7维[pos3,aa3,grip1]），我们走client的`step()`→`processor.Rotate6D_to_AxisAngle`转换→10维[pos3,rot6d,grip1]再转回7维

**可能根因**：
- client的pos处理可能有问题（把绝对pos当delta或做坐标系转换）
- client的`step()`可能在pos上做了额外处理我们不知道

**下一步诊断**：
1. 完整读X-VLA client的`step()`函数（约30行），看pos怎么处理
2. 对比client输出的action给env.step vs 官方直接给env.step的值差异

#### ❓ 疑点3：max_steps=220

**证据**：官方spatial用220步上限（最长训练demo193步），我们用eval_horizon=600太多

**可能影响**：不应影响success判定（done=True才success），但可能让机器人脱离合理状态

**下一步诊断**：改测试脚本用220步上限看是否done（低优先级，不太可能根因）

## 4. 下一步行动建议（优先级排序）

### 建议1：查X-VLA client的step() pos处理（最高优先级）

```bash
# 完整读step()函数，看pos是否做坐标系转换或delta/绝对判定
sed -n '/def step/,/def [a-z]/p' $X_VLA_ROOT/evaluation/libero/libero_client.py | head -30

# 单步跑env.step(action)，打印机器人实际移动位置对比action的pos
# 若pos是绝对目标但client当delta处理→机器人移动距离远小于预期→抓不到物体
```

### 建议2：对比官方eval与我们client的env.step接收action（高优先级）

```bash
# 官方: env.step(action.tolist()) 直接给7维[pos3,aa3,grip1]
# 我们: client.step() → processor转换 → env.step(7维[pos3,aa3,grip1])
# 对比两者给env.step的值差异，定位pos参考系问题
```

### 建议3：查OpenVLA训练时action的pos坐标系定义（中优先级）

```bash
# 查dataset_statistics.json的action统计量 + OpenVLA训练data loader的pos处理
# 确认pos是绝对还是delta，参考系是什么
grep -rE "action.*pos|coordinate|frame|absolute|delta" $OPENVLA_ROOT/experiments/robot/ | head
```

### 建议4：若疑点1-2都排除，改用官方eval绕过client（低优先级）

官方`run_libero_eval.py`有import segfault（prismatic→dlimp→tensorflow），但已用stub注入绕过（`diag_stage1_official.py`证明可行）。可写完整闭环用官方eval逻辑绕过X-VLA client，直接`env.step(action.tolist())`。

## 5. 关键技术细节（供下次对话参考）

### NPU推理环境变量

```bash
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=swrast PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0
export OPENVLA_ATTN=sdpa  # sdpa最佳（eager太慢64.5s，flash_attention_2不支持）
```

### 启动推理服务器

```bash
cd $OPENVLA_ROOT
export PYTHONPATH=$OPENVLA_ROOT:$PYTHONPATH
nohup setsid python $PROJECT_ROOT/models/openvla/server_v2.py \
  --model_path $OPENVLA_ROOT_checkpoints/libero-spatial \
  --port 8011 --unnorm_key libero_spatial --device auto --bf16 \
  > /tmp/openvla_sdpa.log 2>&1 &
```

### 跑闭环验证

```bash
nohup setsid env NUMBA_DISABLE_JIT=1 MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 \
  MESA_LOADER_DRIVER_OVERRIDE=swrast PYOPENGL_PLATFORM=osmesa HF_HUB_OFFLINE=1 \
  LIBGL_DRIVERS_PATH=$HOME/render_libs/dri \
  LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH \
  python $PROJECT_ROOT/scripts/test_openvla闭环.py \
  > /tmp/openvla_loop.log 2>&1 &
```

### 对比诊断（绕过import segfault）

```bash
python $PROJECT_ROOT/scripts/diag_stage1_official.py
# 存官方action到 /tmp/diag_official_action.json
```

### 关键代码位置

- **server_v2.py** `aa_to_rot6d`: 用client同源`AxisAngle_to_Rotate6D`（line 22-27）
- **server_v2.py** `act()`: 推理+resize224+center_crop+grip反转（line 80-160）
- **server_v2.py** `main()`: bf16+sdpa+npu:0加载（line 170-210）
- **test脚本** steps=1 monkey-patch（line 35-42）
- **X-VLA client** `step()`: pos处理逻辑（待查，疑点1）
- **X-VLA client** `Rotate6D_to_AxisAngle`: b1/b2正交化+quat中转（已确认与我们同源等价）

## 6. 避坑指南（已踩坑汇总）

| 坑 | 解法 |
|---|---|
| openvla官方eval import segfault | prismatic→dlimp→tensorflow在OSMesa环境segfault，用stub注入绕过或走我们server框架 |
| bf16模型+fp32 inputs NPU报错 | dtype不匹配，inputs必须用infer_dtype=bf16 |
| flash_attention_2 NPU不支持 | 用sdpa（最佳）或eager（慢3倍） |
| bitsandbytes INT8是CUDA专用 | NPU不支持INT8算子，用bf16原生推理 |
| jp4 kernel占HBM | 临时占用Chip0的25.7GB，等释放后bf16加载（~14GB） |
| cv2.LANCZOS4属性名错 | 正确是`cv2.INTER_LANCZOS4` |
| Rodrigues公式与client转换链不等价 | 用client同源`AxisAngle_to_Rotate6D`两端差=[0,0,0] |
| X-VLA client每10步推理1次 | OpenVLA输出绝对action每步都该推理，monkey-patch steps=1 |
| libero assets路径 | 正确是`libero/libero/assets`（嵌套），缓存`~/.cache/libero/assets` |

## 7. 官方基准对比（待闭环跑通后用）

| Suite | OpenVLA微调基准(A100) | 我们NPU结果 |
|---|---|---|
| Spatial | 84.7 ± 0.9% | **0%**（闭环根因待查） |
| Object | 88.4 ± 0.8% | 未跑 |
| Goal | 79.2 ± 1.0% | 未跑 |
| Long(10) | 53.7 ± 1.3% | 未跑 |
| 平均 | 76.5% | — |

**注**: 官方用3 seed × 500 rollout，我们用10ep × 1seed（spatial先验证闭环）

---

## 8. 总结

**推理链路完全正确**（与官方action逐字节一致），8个修复后闭环仍0%说明根因在**闭环执行链路**——最可能是**pos参考系**（OpenVLA输出绝对目标pos但X-VLA client的`step()`可能把它当delta处理或坐标系不一致）。

**下一步最高优先级**：查X-VLA client的`step()`函数的pos处理逻辑，对比官方eval的`env.step(action.tolist())`，定位pos参考系差异。
