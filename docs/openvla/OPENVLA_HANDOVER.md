# OpenVLA LIBERO NPU 验证记录（OpenVLA 模型文档①）

> **创建时间**: 2026-07-17 17:23  
> **最后更新**: 2026-09-26（并入原 LIBERO_NPU_RECORD.md 的 OpenVLA 问题记录 + 全量验证启动）  
> **目标**: OpenVLA 在 Ascend 910B4 NPU 上闭环 LIBERO 仿真验证  
> **最新纠正（2026-09-27）**：当前可核对的全量 JSON 为 spatial 77/100、goal 79/100、object 75/100、libero_10 57/100，均为 seed42、每 task 10 ep 的原始统计；object 另有独立视频复验 10 段、5/10，不能并入 object 全量。用户反馈历史 X-VLA/OpenVLA 视频整理丢失；目前未保存且没有可恢复的删除证明。论文参考值与本项目统计分开，来源待核实；后续验证必须保存同次 NPU rollout 视频。
>
> **当前状态**: ✅ 四个 suite 的结果 JSON 已存在；视频证据仅限上述 object 独立复验。
  
> **⚠️ 阅读提示**: §2-§4 是根因定位期间（0% 时代）的**历史快照**，其"当前阻塞 0%"表述已被 §4.2 的 delta 根因修复推翻，保留供追溯诊断过程。  
> **X-VLA 关联说明**：当前可引用的是四套件原始统计；历史视频整理状态按顶部最新纠正声明处理。

---

## 1. 关键文件清单

| 文件 | 说明 | 状态 |
|---|---|---|
| `models/openvla/server_v2.py` (215行) | OpenVLA NPU推理服务器，bf16+sdpa+npu:0 | ✅ 推理正确 |
| `scripts/test_openvla闭环.py` | 3ep闭环验证脚本，含steps=1 monkey-patch | ✅ 跑通但0% |
| `scripts/diag_stage1_official.py` | 官方action推理对比诊断（绕过import segfault） | ✅ 已验证推理一致 |
| `scripts/openvla/run_openvla_libero.sh` | 一键验证脚本（路径B，因import segfault弃用） | ⚠️ 弃用 |
| `docs/PROJECT_TRACKING.md` (589行) | 完整项目追踪文档 | ✅ 持续更新中 |
| `/tmp/diag_official_action.json` | 官方action对比基准文件 | ✅ 存在(448B) |
| `$OPENVLA_ROOT/` (2.1M) | openvla仓库（含NPU适配+stub注入） | ✅ |
| `$OPENVLA_ROOT_checkpoints/libero-spatial/` (15G) | libero_spatial微调权重 | ✅ 完整 |

## 2. 已确认正确的部分（无需再查）

### ✅ 推理链路完全正确

**对比诊断证据**（`diag_stage1_official.py`绕过import segfault直接调官方`predict_action`）：
- 官方action(7维): `[0.096, 0.035, -0.003, ~0, ~0, ~0, 0.996]`
- 我们server(10维): `[0.096, 0.035, -0.003, 1,-0,0, 1,-0,0, 0.996]`
- **历史单步对比记录**：pos3/rot6d/grip 差异记录为 0；这只支持该次输入和实现的数值对比，不扩展为所有协议、环境或 checkpoint 的逐字节一致。

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

## 6. 问题-根因-解决全记录（原 LIBERO_NPU_RECORD.md OpenVLA 部分并入，2026-09-26）

### 6.1 环境与加载

| # | 问题 | 根因 | 解决 |
|---|---|---|---|
| 1 | 官方 eval import segfault | prismatic→dlimp→tensorflow 链在 OSMesa 环境 segfault | stub 注入假模块绕过（server_v2.py:29-35），或直接走我们 server 框架 |
| 2 | bf16 模型 + fp32 inputs 报 dtype 不匹配 | NPU 要求模型与 inputs dtype 一致 | inputs 用 `infer_dtype=bf16` |
| 3 | attention 选型 | eager 64.5s / sdpa 22.1s / flash_attention_2 崩（NPU 不支持 fa2 算子） | `OPENVLA_ATTN=sdpa`（3x 加速，动作值与 eager 完全一致） |
| 4 | bitsandbytes INT8 报错 | INT8 算子 CUDA 专用，NPU 无实现 | bf16 原生推理（7B → ~14GB HBM） |
| 5 | bf16 加载 OOM | jp4 kernel 临时占 Chip0 25.7GB HBM | 等 kernel 释放后加载 |

### 6.2 动作处理

| # | 问题 | 根因 | 解决 |
|---|---|---|---|
| 6 | `cv2.LANCZOS4` AttributeError | 属性名错 | 正确是 `cv2.INTER_LANCZOS4` |
| 7 | Rodrigues 转换链与 client 不等价（aa_in=[0,0,0]→aa_out=[0,0,-0.588]） | server 用 Rodrigues 公式，client 用 b1/b2 正交化+quat 中转 | server 改用 client 同源 `AxisAngle_to_Rotate6D`，两端差=[0,0,0] |
| 8 | 夹爪符号不匹配 | 官方链路 normalize→invert；我们链路 client `>0.5` 离散化 | server 端 `grip = 1.0 - grip`（例：grip_raw=0.996→返 0.004→client -1→env open，匹配官方 invert） |
| 9 | X-VLA client 每 10 步推理 1 次 | client 按 chunk 设计缓存 action，OpenVLA 每步预测 | monkey-patch `policy._format_query` 强制 `steps=1` |

### 6.3 闭环执行（核心根因，详见 §3-§4）

| # | 问题 | 根因 | 解决 | 验证 |
|---|---|---|---|---|
| 10 | 8 个修复 × 3ep 闭环全 0% | **OpenVLA 输出 delta action，client 默认 act_type="abs" 把 delta pos 当绝对目标坐标** → 机器人被拉出 workspace → 永远 done=False | client 走 `act_type="rel"`，env 保持 `use_delta=True`，与官方 `env.step(action.tolist())` 语义一致 | task0 100% (3/3) → spatial 76.0% (38/50) |

### 6.4 性能优化

| # | 问题 | 优化 | 效果 |
|---|---|---|---|
| 11 | 每步推理 ~1.5s（50 rollouts ≈ 148min） | ①全局预计算 crop 参数 ②固定 seed 只设一次 ③`json_numpy.patch()` 只调一次 ④图像预处理合并 | 单步 0.45s（**3.3x**），SR 无损失 |

### 6.5 工程事故记录（2026-09-26 新增）

| # | 问题 | 根因 | 解决 |
|---|---|---|---|
| 12 | 07-17 隐私清理把 `${X_VLA_ROOT:?...}` bash 语法写进 5 个 .py（eval_spatial_full/diag_*/apply_patches 等），脚本此后不可运行 | 机械替换未做语法回归 | 全部改为 `os.environ.get(...)` + 绝对路径默认值；eval_spatial_full.py 重构为 scripts/openvla/eval_openvla_suite.py 后实测跑通 |
| 13 | 官方 4 suite checkpoint HF 不可达 | ModelArts 出站代理拒外网 IP | 找到 modelscope 社区镜像（superpeach/*），4 分片大小与官方逐字节一致后采用 |

## 7. 官方基准对比

| Suite | OpenVLA 微调基准 (A100, 50 trial × 3 seed) | 我们 NPU（首次, 5ep×10task） | 我们 NPU（全量验证, 10ep×10task） |
|---|---|---|---|
| Spatial | 84.7 ± 0.9% | **76.0%** (38/50) | 🔄 跑中 |
| Object | 88.4 ± 0.8% | 未跑 | 🔄 跑中 |
| Goal | 79.2 ± 1.0% | 未跑 | 🔄 跑中 |
| Long(10) | 53.7 ± 1.3% | 未跑 | 🔄 跑中 |
| 平均 | 76.5% | — | 🔄 |

## 8. 4 suite 全量验证（2026-09-26 启动）

**动机**: 首次验证只跑了 spatial 单 suite 50 rollouts，与 OpenVLA 论文的 4 suite 协议不对齐。本次补齐全量。

**协议**（对齐论文，偏差已记录）：
- 4 suite 各用官方对应微调 checkpoint（`msharma11/openvla-7b-13b-libero-*` 的 modelscope 镜像 `superpeach/openvla-7b-fin*`，已验证 4 分片大小与官方逐字节一致，dataset_statistics.json 的 unnorm key 各 suite 正确）
- 官方步数上限：spatial 220 / object 280 / goal 300 / long 520（官方 run_libero_eval.py:173-182）
- seed 42 单 seed，10 ep/task × 10 task = 100 rollouts/suite（论文 50 trial/task，时间预算取 10，与 X-VLA 10ep 约定一致）
- 双卡并行两流：流A（卡0:8011）spatial→goal，流B（卡1:8021）object→libero_10
- 编排：`scripts/openvla/run_openvla_full_validation.sh`（自动起停 server + 分片完整性校验），评估：`scripts/openvla/eval_openvla_suite.py`
- 结果：`results/openvla_full/<suite>_results.json`（增量保存防中断）

**初步观察**（spatial task0）：10/10 全成功（100%），与首次 76.0% 的 task0 100% 一致。

**进度更新（2026-09-26 下午）**：
- ✅ 当前可核对 JSON：spatial 77/100、goal 79/100、object 75/100、libero_10 57/100。
- ✅ object 独立视频复验：10 段实际闭环 MP4，5/10；不替代 object 全量统计。
- 论文/官方参考值与协议来源待核实；不使用“2σ 内”“统计等价”等结论。
- 15:13 notebook 重启曾杀死双流 → 新增 `eval_openvla_suite.py --resume` + 编排器 `RESUME=1`（已完成 task 跳过，配置一致性校验）
- **视频复验（2026-09-27）**：使用同一 OpenVLA libero-object checkpoint、同一 `act_type=rel`、seed42、Ascend NPU1 server（port 8022）独立跑 1 ep/task，共 10 个实际闭环 mp4，5/10 成功；结果 JSON：`results/openvla/video_validation_2026-09-27/libero_object_results.json`，视频：`results/openvla/videos/libero_object/`。该样本用于证明视频链路和成功标记对应关系，不替代 100 rollout 全量统计（全量 object：75/100，官方 88.4%）。

## 9. 关键教训（原 LIBERO_NPU_RECORD.md 教训并入）

1. **推理正确 ≠ 闭环正确**: OpenVLA 推理与官方逐字节一致，但闭环 0%，根因在 delta/abs 参考系不匹配。
2. **docstring 是金矿**: `openvla.py:47` 的 "end-effector deltas" docstring 直接指向根因。
3. **client 框架假设**: X-VLA client 假设 abs action，迁移 delta action 模型时必须显式切换 `act_type="rel"`。
4. **NPU 量化限制**: bitsandbytes INT8/4bit 是 CUDA 专用，NPU 只能用 bf16/fp32 原生推理。
5. **性能优化优先级**: 固定开销（seed、patch、预计算）> 算法层优化，因为 delta action 不能简单 chunk 缓存。
6. **隐私清理必须做语法回归**（2026-09-26 新增）: 机械替换路径变量会把 bash 语法写进 Python，清理后必须 `py_compile` 全量过一遍。

---

## 10. 总结（历史快照：2026-07-17 pos 参考系定位阶段，根因已由 §3-§4 定论修复）

**历史单步推理对比记录显示数值一致**；该证据范围限于当次输入与实现，不能扩展为所有环境和协议。8个修复后闭环仍0%说明根因在**闭环执行链路**——最可能是**pos参考系**（OpenVLA输出绝对目标pos但X-VLA client的`step()`可能把它当delta处理或坐标系不一致）。

**下一步最高优先级**：查X-VLA client的`step()`函数的pos处理逻辑，对比官方eval的`env.step(action.tolist())`，定位pos参考系差异。
> 注：此为 07-17 诊断中段的快照，最终定论见 §3-§4（act_type="rel" 根因）与 §8（76.0%→77.0% 验证结果）。
