# PI0.5 LIBERO 仿真验证 — 问题与解决方法记录

> **用途**: 记录 pi0.5 在 NPU 上做 LIBERO 仿真验证过程中遇到的问题与解决方法，供后续参考  
> **配套追踪文档**: `docs/pi05/PI05_TRACKING.md`  
> **创建时间**: 2026-07-20  
> **最新纠正（2026-09-27）**：本文保留历史过程记录。旧 run `results/pi05/spatial_100ep_2026-09-26/spatial_results.json` 与 `progress.log` 记录 spatial 96/100，task5 70%、task8 100%、task9 90%，总耗时 10359s（172.65min，2.8775h，约103.59s/ep）。新 run `results/pi05/20260927_104705_120708_2242249/` 正在 NPU0 fp32 上进行，当前 task0 已完成 4/10、全局 4/100，4/4 成功，已保存 4 个 episode 视频；不写完成结论。
>
> **最后更新**: 2026-09-27

---

## 一、环境搭建问题

### P1.1 torch_npu 2.5.1.post1 wheel 在 gitcode 需认证（401）
- **根因**: cann-recipes 验证过的 torch_npu 2.5.1.post1 在 gitcode.com/Ascend/pytorch 的 release 需登录 token，HF 直连也被代理阻断
- **解决**: 改用 torch 2.7.1 + torch_npu 2.7.1.post2（modelarts 私有源 `pip.modelarts.private.com:8888`，与 PyTorch-2.7.1 env 一致，NPU 已验证可跑）
- **状态**: ✅ 已解决

### P1.2 lerobot fc296548 pi0.5 要 transformers 的 `fix/lerobot_openpi` 分支版
- **根因**: lerobot fc296548 的 pi extra 要 `transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi`，不是 pip 的 4.49.0；用 4.49.0 报 `An incorrect transformer version` check 失败
- **解决**: `pip install --no-deps "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"` → 装 4.53.3
- **状态**: ✅ 已解决

### P1.3 lerobot import 链缺多个依赖
- **根因**: lerobot fc296548 的 pyproject.toml 声明的依赖未在 editable install 时全装（--no-deps）
- **解决**: 逐个补装：deepdiff / av / tornado / cmake / pyserial / pynput / rerun-sdk / accelerate / wandb / datasets / packaging / setuptools / decorator（Ascend ACL 编译要）/ sentencepiece / gemma
- **状态**: ✅ 已解决

### P1.4 numpy 被其他包升回 2.x 导致 torch_npu 不兼容
- **根因**: datasets / pandas 等依赖要 numpy>=2.0，但 torch_npu 2.7.1.post2 要 numpy<2
- **解决**: 最后强制 `pip install --force-reinstall "numpy==1.26.4"`
- **状态**: ✅ 已解决

### P1.5 OSMesa 渲染环境变量必须在 python 启动前 export
- **根因**: `MUJOCO_GL=osmesa` / `PYOPENGL_PLATFORM=osmesa` / `LD_LIBRARY_PATH=~/render_libs` 必须在 mujoco import 前由 OpenGL 自己读到；os.environ.setdefault 在脚本内设已晚（mujoco `osmesa/__init__.py` 在 `from OpenGL import GL` 时触发 platform 检测）
- **解决**: eval 脚本顶端 `os.environ["..."] = ...` 强制设（非 setdefault）；bash 启动前 `export` 全部环境变量
- **状态**: ✅ 已解决

### P1.6 LD_LIBRARY_PATH 必须含 ~/render_libs 且在 python 启动前 export
- **根因**: mujoco osmesa 的 `from OpenGL import GL` 触发 `loadLibrary('OSMESA')`，若 `libOSMesa.so` 找不到返 None → `glGetError` AttributeError；`LD_LIBRARY_PATH` 在 python 内 os.environ.setdefault 设无效（动态链接器在进程启动时读一次）
- **解决**: bash `export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:${LD_LIBRARY_PATH:-}` 在 python 启动前
- **状态**: ✅ 已解决

## 二、NPU 推理问题

### P2.1 config.json 含 lerobot 后续版本字段 `use_peft`
- **根因**: `lerobot/pi05-libero` 的 config.json 含 lerobot 后续版本字段 `use_peft`，lerobot fc296548 的 PI05Config 不识别 → `TypeError: __init__() got unexpected keyword argument 'use_peft'`
- **解决**: 移除 config.json 的 `use_peft` 字段
- **状态**: ✅ 已解决

### P2.2 paligemma 加载需 `google/paligemma-3b-pt-224` VLM backbone
- **根因**: pi0.5 用 paligemma 做 VLM backbone，加载时 transformers 自动拉 `google/paligemma-3b-pt-224`（gated repo，需 token）；HF 直连被代理阻断，hf-mirror 也只返 redirect 文档
- **解决**: 用 modelscope 镜像下载（11GB）+ 整理成 HF local layout + `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 离线模式
- **状态**: ✅ 已解决

### P2.3 policy_preprocessor.json 的 tokenizer_name 离线找不到 paligemma
- **根因**: HF 离线模式解析 `google/paligemma-3b-pt-224` 要 refs/main 指向真实 commit hash，snapshots/<hash>/ 文件；modelscope 用 master 不是 hash，transformers 离线解析失败
- **解决**: 把 `policy_preprocessor.json` 的 `tokenizer_name` 改为本地平铺路径 `/home/ma-user/work/paligemma3b_hf`（绕 HF 离线 gated repo 限制）
- **状态**: ✅ 已解决

### P2.4 `compile_model=true` + `compile_mode=max-autotune` 触发 torch dynamo 编译报 torch_mlir 缺
- **根因**: pi0.5 config 的 `compile_model=true` 在 NPU 上调 torch dynamo 编译，报 `backend='inductor' raised ImportError: torch_mlir is not installed`（Ascend ACL 的 inductor backend 缺 torch_mlir）
- **解决**: config.json 关 `compile_model`（推理无需 dynamo 编译）
- **状态**: ✅ 已解决

### P2.5 NPU monkey-patch：bf16 调 torch.normal 报 aclnnNormalFloatFloat 不支持 BFLOAT16 输出
- **根因**: pi0.5 的 `sample_noise`/`sample_time` 用 `_inference_dtype()`（bf16）调 `torch.normal(mean=0, std=1, dtype=bf16)`，Ascend 910B4 的 `aclnnNormalFloatFloat` 只支持 FLOAT16/FLOAT/DOUBLE，不支持 BFLOAT16 输出 → 报 EZ1001
- **解决**: server_v2.py 加 monkey-patch：强制 fp32 采样噪声/时间，再 `.to(bf16)` 给后续算子（数值精度无损）
- **状态**: ✅ 已解决

### P2.6 bf16 tensor 直接 np.array(..., dtype=fp32) 报 Got unsupported ScalarType BFloat16
- **根因**: server 的 `pred_action` 是 bf16 tensor，`np.array(pred_action, dtype=np.float32)` 在 bf16 上崩（PyTorch 不支持 bf16 → numpy 直接转）
- **解决**: 先 `.float().cpu().numpy()` 转 fp32 再 numpy
- **状态**: ✅ 已解决

### P2.7 CANN 报 `No module named 'decorator'`
- **根因**: Ascend ACL 编译要 `decorator` Python 模块，lerobot-pi05 env 缺装
- **解决**: `pip install decorator`
- **状态**: ✅ 已解决

## 三、动作格式适配问题

### P3.1 chunk 语义错配：select_action vs predict_action_chunk
- **根因**: server 用 `select_action` 每次返 1 步（内部 queue 滚动），但 client.step 期望多步 chunk（`action[:, :9]` 切片）；导致 client 切片错位 → 10 步全同值 → 0%
- **解决**: server 改用 `predict_action_chunk` 一次返 `n_action_steps=10` 步，client 用 `action_plan` 缓存滚动
- **状态**: ✅ 已解决

### P3.2 postprocess 要 PolicyAction 类型不能传 dict
- **根因**: server 误把 `{"action": pred_chunk}` dict 传给 `postprocess`，报 `Action should be a PolicyAction type got <class 'dict'>`
- **解决**: 直接传 `pred_chunk[:, :n_act]`（PolicyAction 支持切片）
- **状态**: ✅ 已解决

### P3.3 aa3 → rot6d 必须用 client 同源 AxisAngle_to_Rotate6D
- **根因**: server 早先用 Rodrigues 公式转 rot6d，与 client 的 `Rotate6D_to_AxisAngle` 不等价（差值大 → 0%）
- **解决**: 复用 `libero_client.LiberoAbsActionProcessor.AxisAngle_to_Rotate6D`（与 OpenVLA server_v2 同源转换链）
- **状态**: ✅ 已解决

### P3.4 grip 反转匹配 client >0.5 离散化语义
- **根因**: 官方链路 `grip_raw → normalize [-1,+1] → invert → env.step（+1=close）`；server 链路 `server返1-grip_raw → client >0.5离散化 → env.step`
- **解决**: server 返 `1.0 - delta_grip`（例 grip_raw=0.996 → server 返 0.004 → client <0.5 → -1 → env.step(-1=open)），匹配官方 invert
- **状态**: ✅ 已解决

### P3.5 pi0.5 输出 delta action 与 X-VLA client act_type=rel 路径语义一致
- **根因**: lerobot pi0.5 LIBERO checkpoint 训练用 `control_mode="relative"`（use_delta=True），输出 delta action；X-VLA client `act_type=rel` 路径保持 env 默认 `use_delta=True`，语义一致
- **解决**: 无需特殊处理，走 act_type=rel 即可
- **状态**: ✅ 已确认

## 四、闭环执行问题

### P4.1 `_init_env` 返回 (env, lang, obs) tuple 不是 env
- **根因**: server 早先误把 `_init_env` 返回值当 env 用，报 `'tuple' object has no attribute 'reset'`
- **解决**: 解构 `env, lang, obs = evaluator._init_env(...)`
- **状态**: ✅ 已解决

### P4.2 client._format_query 要 obs['robo_ori']/obs['robo_pos']，但 _init_env 返回的 raw obs 没这俩 key
- **根因**: `_init_env` 返回 settle 后的 raw obs，key 是 `agentview_image`/`robot0_eef_pos` 等，非 `robo_ori`/`robo_pos`
- **解决**: 每步从 `env.env.robots[0].controller` 注入：`obs['robo_ori'] = processor.Mat_to_Rotate6D(controller.ee_ori_mat)` + `obs['robo_pos'] = controller.ee_pos`（与官方 `_rollout` 一致）
- **状态**: ✅ 已解决

### P4.3 proprio 20 维 → pi0.5 LIBERO state 8 维映射
- **根因**: X-VLA client 发 20 维 proprio `[pos3+ori6d+grip1 + past copy]`，pi0.5 LIBERO 训练时 state=8 维 `[pos3+quat4+grip1]`
- **解决**: server 截取前 10 维 `[pos3, ori6d, grip1]`，ori6d → rot_mat → quat（robosuite T.mat2quat），拼成 8 维 state
- **状态**: ✅ 已解决

### P4.4 task0 持续 0% 跑满 220 步不是技术 bug（2026-07-20 首判，2026-07-21 推翻）
- **首判（错）**: 2026-07-20 spatial suite 100ep 全 0% 后判为"模型真实表现非 bug"
- **推翻（2026-07-21 重诊断）**: 100ep 全 0% 远低于官方 98.8%，必有系统性根因。重读 `libero_client.py` 定位真根因见 P4.5
- **状态**: ❌ 已推翻，真根因见 P4.5

### P4.5 client.proprio 被 delta action 污染 → server state 输入错位（真根因，2026-07-21 定位）
- **根因**: `libero_client.py:197` `self.proprio[:9] = action[-1, :9].copy()` 把 server 末步返回值当绝对 proprio 赋值。
  - X-VLA server 返**绝对 target pose** → 此赋值正确（X-VLA 验证 95.8% 成功印证）
  - pi0.5 server 返**delta action**（pos3 ±0.05 范围）→ proprio 被污染成 delta 值
  - `_format_query:170` `payload["proprio"] = json_numpy.dumps(self.proprio)` 每步发被污染的 proprio（非 `closed_loop_proprio` 真值）
  - server 收到 proprio 越来越偏离真实位姿 → pi0.5 state_8 输入错位 → 推理动作错 → 0%
- **关键证据**: client 注释"last absolute [pos(3)+ori6d(6)+grip(1)]"印证 client 设计期望 server 返绝对值；pi0.5 输出 delta 与 OpenVLA 同，但 OpenVLA 验证时也走 act_type=rel 路径——需对比 OpenVLA 当时是否避开此 bug
- **解决路径（不改 libero_client.py 官方便样）**: server 端绕过——server 不信 client 发的 proprio，自己每步从 env 推断真值；或 server 把 delta chunk 末步转成绝对值再返（让 client.proprio 赋值正确）。后者更稳：server 返 `cumsum(delta_pos) + ee_pos0` 形式的绝对末步 pose 给 client 赋值
- **状态**: ⏳ 待修复（2026-07-21 定位，待本轮重跑验证）

### P4.6 lerobot-pi05 conda env 被外部清理（2026-07-21 发现，2026-07-21 修正描述）
- **根因**: 服务器 conda env 被外部清理（与 PROJECT_TRACKING #8 同类型风险再次发生），`conda env list` 只剩 base/PyTorch-2.7.1/python-3.12.0
- **修正首判**: 不光 env 被清，`/home/ma-user/work/lerobot_pi05/` editable install 源码目录**仍在**（commit fc296548 完整，含 src/lerobot/policies/pi05/、envs/libero.py、factory.py）——首判误扩为"源码也丢"，2026-07-21 21:30 精准确认源码无损
- **影响**: 不能直接复跑验证，需先 `conda create -n lerobot-pi05 python=3.10` + 重装依赖 + lerobot editable reinstall（源码已就位，无需重 clone）
- **解决**: 按 PI05_TRACKING 阶段2 重建（torch 2.7.1 + torch_npu 2.7.1.post2 + lerobot fc296548 editable + transformers fix/lerobot_openpi 分支 + 各依赖）。源码目录就绪，重建成本较首建低
- **状态**: ⏳ 待重建（仅 env，源码就绪）

## 五、性能优化问题

### P5.1 NPU 推理首次算子编译开销大
- **现象**: 单次推理首次 ~20s（含 NPU 算子编译），后续 ~8s/次
- **解决**: 未优化（阶段5 重点是闭环验证非性能），后续可考虑 warm-up + 算子缓存
- **状态**: ⏳ 待优化

## 六、问题解决方法汇总表

| # | 问题类别 | 核心方法 | 状态 |
|---|---|---|---|
| P1.1 | 环境 | torch_npu 2.5.1.post1→2.7.1.post2（modelarts 私有源） | ✅ |
| P1.2 | 环境 | transformers 装 `fix/lerobot_openpi` 分支版 4.53.3 | ✅ |
| P1.3 | 环境 | 逐个补装 lerobot 缺失依赖 | ✅ |
| P1.4 | 环境 | 强制 numpy==1.26.4 --force-reinstall | ✅ |
| P1.5 | 环境 | 环境变量在 import 前用 os.environ[...]= 强制设 | ✅ |
| P1.6 | 环境 | LD_LIBRARY_PATH 含 ~/render_libs 在 python 启动前 export | ✅ |
| P2.1 | 推理 | config.json 移除 use_peft 字段 | ✅ |
| P2.2 | 推理 | modelscope 镜像下载 paligemma + 离线模式 | ✅ |
| P2.3 | 推理 | tokenizer_name 改为本地平铺路径 | ✅ |
| P2.4 | 推理 | config.json 关 compile_model | ✅ |
| P2.5 | 推理 | NPU monkey-patch：bf16 noise → fp32 采样 | ✅ |
| P2.6 | 推理 | bf16 tensor 先 .float().cpu().numpy() | ✅ |
| P2.7 | 推理 | pip install decorator | ✅ |
| P3.1 | 适配 | select_action → predict_action_chunk（chunk 语义） | ✅ |
| P3.2 | 适配 | postprocess 直接传 PolicyAction 切片 | ✅ |
| P3.3 | 适配 | aa3→rot6d 用 client 同源 AxisAngle_to_Rotate6D | ✅ |
| P3.4 | 适配 | grip 反转 1.0-grip 匹配 client 离散化 | ✅ |
| P3.5 | 适配 | act_type=rel 与 pi0.5 control_mode=relative 语义一致 | ✅ |
| P4.1 | 闭环 | _init_env 返回 tuple 解构 | ✅ |
| P4.2 | 闭环 | 每步注入 obs['robo_ori']/obs['robo_pos'] | ✅ |
| P4.3 | 闭环 | proprio 20维 → state 8维映射（ori6d→quat） | ✅ |
| P4.4 | 闭环 | task0 持续 0% 是模型真实表现非 bug | ❌ 推翻（见 P4.5） |
| P4.5 | 闭环 | client.proprio 被 delta 污染 → server state 输入错位（真根因） | ⏳ 待修复 |
### P4.7 server state_8 schema 错位（真根因，2026-07-21 定位 + 修复 + 推翻）
- **根因**: server_v2.py 给的 `observation.state` 字段顺序错为 `[pos3, quat4, grip1]`，pi0.5 LIBERO 训练时真值 schema 是 `[eef_pos(3), axis_angle(3), gripper_qpos(2)]`（lerobot `env_processor.py:71` `state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)`）
- **证据**: norm_stats `observation.state.mean` 第 4 维=2.9722（axis_angle 朝下 ~π），server 给 quat 第 4 维=0 → 归一化后 `(0-2.97)/0.34=-8.6`，旧错位归一化 `|max|=68.6` 远超 [-1,1]
- **修复**: server 把 ori6d → rot_mat → quat(xyzw) → axis_angle(3)（与 lerobot `_quat2axisangle` 同源链，sanity diff=0），gripper 用训练 q50 默认 `[0.02636, -0.02728]`
- **sanity**: 修复后归一化 `|max|=1.56`（旧 68.6→现 1.56，axis_angle 落训练 3σ），转换链正确性 diff≤1.8e-6
- **状态**: ❌ 修复正确但**非唯一根因**——闭环仍 0%，P4.8 推翻见下

### P4.8 client proprio ori6d 与 env 真值 ori6d 不一致（2026-07-21 定位，待用户拍板）
- **根因**: client 真发 proprio ori6d `[0.0018, 0.99999, 0.0005, 0.9984, -0.0018, -0.0558]` 反转的 mat，与 env 真值 `robot0_eef_quat` quat2mat 转的 mat **完全不同**（diff=1.99，quat 在 x/y 轴反向翻转）
- **证据**: 
  - env 真值 quat `[w,x,y,z]=[0.9996, -0.0009, -0.028, -0.00026]` → mat `[[0.998,0,-0.056],[-0.0005,1,0.0018],[0.056,-0.0018,0.998]]`
  - client ori6d 反转 mat `[[0.0018,0.998,-0.056],[1.0,-0.0018,0.0006],[0.0005,-0.056,-0.998]]`——**第 1 列 ↔ 第 2 列调换，第 3 列反号**
  - `Mat_to_Rotate6D(env mat)` 与 client 真发 ori6d diff=1.0（非互逆问题，是 ori6d 来源不一致）
- **副发现**: env 与训练**是同朝向**（env axis_angle `[-0.0018, -0.056, -0.0005]` 朝下 ~180°，训练 mean `[2.97, -0.22, -0.13]` 朝下偏一点，diff/π=`[0.05, 0.07, 0.01]`）——**不是 quat convention 翻转问题**
- **真根因**: client 发的 proprio ori6d 不是从当前 env `ee_ori_mat` 转的，而是别处（初始化缓存或 proprio 污染链的更深 bug）。这是 `libero_client.py` 官方便样的固有问题，记忆/RECORD 都明确"不改 libero_client.py"
- **修复路径（不改 client）**: server 端绕过——server 不信 client 发的 ori6d，自己从 env quat 推断（但 server 是 HTTP 服务不知道 env 状态）；或要求 client 改发 env quat 而非 ori6d（破官方便样）；或 client 改 `Mat_to_Rotate6D` 列序（破官方便样）。**所有路径都需破"不改 client"约束或破 server 无状态设计**
- **状态**: ⏳ 待用户拍板（陷入需决策死结，见"七、关键教训 #12"）

### P4.9 robosuite robot0_eef_quat convention 是 [w,x,y,z] 不是 [x,y,z,w]（2026-07-21 发现）
- **根因**: robosuite raw obs `robot0_eef_quat` 用 `[w, x, y, z]` convention（不是 `mat2quat`/`quat2mat` 的 xyzw）——env 真值 `[0.9996, -0.0009, -0.028, -0.00026]` 是 wxyz
- **影响**: 任何从 raw obs 取 quat 的代码都需先 wxyz→xyzw 转换再传 quat2mat
- **状态**: ✅ 已确认（本轮 sanity test 印证），server 不直接用 raw obs quat，无修复需求

| P4.7 | 闭环 | server state_8 schema 错位（quat vs axis_angle） | ❌ 修复正确但非唯一根因（P4.8 推翻） |
| P4.8 | 闭环 | client proprio ori6d 与 env 真值 ori6d 不一致（真根因，待拍板） | ✅ 已解——精定位 perm=(3,2,5,4,1,0) signs=(1,-1,-1,-1,1,-1) 列序调换规律 + server 侧绕过（2026-07-23） |
| P4.9 | 转换 | robosuite robot0_eef_quat convention 是 [w,x,y,z] | ✅ 已确认无修复 |

### P4.10 chunk 末步转 abs pose 在 rel 模式下被 env controller 当 delta 累加飞出 workspace（2026-07-23 发现并修复）
- **根因**: P4.5 v1 修复把 chunk 末步[:3]赋 abs_pos_last（-3.80 量级），rel 模式 osc.py:235 env controller 把末步 abs pose 当 delta 累加 current_pos+(-3.80) 飞出 workspace
- **解决**: P4.5 v2——末步[:9]回传 proprio 原值（env scale_action 缩放累加 ±0.01 微动可接受），不滚雪球
- **状态**: ✅ 已解决（但闭环仍 0%，根因在更上游）

### P4.11 client proprio ori6d 与训练分布 ori6d 列序调换+反号（2026-07-23 精定位并修复）
- **根因**: client proprio ori6d 来自 env ee_ori_mat → Mat_to_Rotate6D，训练分布 ori6d 来自 robot0_eef_quat wxyz → quat2mat → Mat_to_Rotate6D，两源 ori6d 列序调换+反号完全不同（diff=1.0018）→ server 收错 ori6d → state 归一化错位 → 推理输出错方向 → 闭环 0%
- **解决**: server 侧 perm=(3,2,5,4,1,0) signs=(1,-1,-1,-1,1,-1) 转换绕过（8 组 ori 验证普适 diff<0.01），不改 libero_client.py 官方便样
- **验证**: 修复后 axis_angle 与训练真值 diff=0.00007，state 归一化与训练分布一致 diff=0.0002，超[-1,1]维数 3/8 与训练同
- **状态**: ✅ 已解决（但闭环仍 0%，根因在更上游——模型推理本身返固定方向 delta）

### P4.12 闭环 0% 真根因彻底定论：模型推理本身返固定方向 delta（2026-07-23）
- **根因**: #1-#13 全部诊断链印闭环 0% 根因不在任何外围链路——模型推理本身返固定方向 delta（#6 印 dummy 输入也返 ±0.6~0.8，#7 印 chunk 10 步同方向是 flow matching 预期特性）
- **外围链路全排查正确**: P4.8 ori6d 修复真生效、图像 flip 对齐训练、env act 执行正确、proprio 原值回传不滚雪球、推理链路正确、state discretize clip 链路一致、task convention 不影响、图像每步真值不首帧复用
- **剩余根因方向**: (A) 模型本身在 NPU bf16 上对当前输入的预期输出就是这个方向（不是 bug）——需对比 cann-recipes 官方 fp16 推理输出方向验证；(B) 训练数据采集时的 action convention 与推理时不一致让模型学到错方向；(C) chunk 10 步 delta 累加后机器人朝 workspace 外飞——需精查 env controller scale_action 缩放比真值
- **状态**: ⏳ 待用户拍板剩余根因方向（A/B/C 三选一或组合）

### P4.13 方向 A 定论：bf16 精度是闭环 0% 根因之一（2026-07-23 印证）
- **根因**: 模型在 NPU bf16 上推理输出方向与 float32 不同——同输入（训练 init ep0 真值 state + 真实 LIBERO 图像）bf16 vs float32 推理，chunk 10 步 delta_pos 累加末步 diff=0.36 > 0.05，两 dtype 输出方向不同
- **证据**: bf16 累加末步 `[-4.88, 0.05, -5.26]` z=-4.29 vs float32 `[-4.68, -0.10, -4.90]` z=-3.93
- **局限**: 两 dtype chunk 都朝同方向飞出 workspace（z 都 <<0.8），说明 **bf16 精度不是唯一根因**，float32 也输出错方向——bf16 精度是根因之一但非全部
- **状态**: ✅ 已定论（bf16 精度根因之一，需配合方向 C 修复）

### P4.14 方向 B 推翻 + 方向 C 定论：chunk 10 步 delta 同方向累加飞出 workspace（2026-07-23）
- **方向 B 推翻**: 训练采集 `lerobot libero env libero.py:115 control_mode="relative"` 显式设 `use_delta=True`（libero.py:309），推理 `X-VLA libero env libero_client.py:273 act_type='rel' pass` 保持默认 True，两 env 共用 `env_wrapper.py:17 controller="OSC_POSE"` + `osc_pose.json:15 control_delta:true`，controller config 一致——**训练-推理 action convention 一致不是根因**
- **方向 C 定论**: env controller `action_scale[pos3]=[0.05,0.05,0.05]`，chunk 首步 delta_pos `[-0.757,-0.045,-0.964]` 缩放到 `[-0.038,-0.002,-0.048]`（单步安全），但 chunk 10 步 delta 各步都朝负 z 方向（step0-9 delta_z 全负），10 步累加位移 `[-0.38,-0.02,-0.48]`，真累加后 ee_pos z=0.694 **< workspace 下界 0.8 飞出**
- **根因**: chunk 内不重新抓 obs 一次性预测 10 步 delta 都朝同方向，单步缩放安全 ±0.048 但 10 步累加位移 z=-0.48 飞出 workspace 下界 0.8
- **下一步修复方向**: F1 缩短 chunk（server 返 chunk 长度从 10 步缩到 1-2 步，client 每步重新推理抓新 obs）/ F2 env controller clip 单步 delta 上限（output_max 从 0.05 缩到 0.01）/ F3 server 侧 chunk 累加后 clip 到 workspace / F4 bf16→float32 推理（解方向 A bf16 精度根因）。**推荐组合 F1+F4**：F1 解方向 C 飞出根因，F4 解方向 A bf16 精度根因
- **状态**: ✅ 已定论（A+C 组合根因，B 推翻），⏳ 待用户拍板修复方向 F1/F2/F3/F4

### P4.15 F1+F4 组合实施生效但闭环仍 0%（2026-07-23）
- **F1 实施**: `server_v2.py:191 n_act=min(2, policy.config.n_action_steps)` + `server_v2.py:215 T_chunk=min(2, req.steps)` 硬覆盖 chunk 长度到 2 步（避免下游 tile 填充绕过 n_act=2）
- **F1 验证生效**: server debug log 印 `return chunk shape=(2, 10)` chunk 从 9 步缩到 2 步
- **F4 实施**: `server_v2.py:325 --bf16 default=False` 改默认 float32 推理（解 bf16 精度根因）
- **F4 验证生效**: server startup log 印 `infer dtype=torch.float32`（bf16=False）
- **闭环效果**: ❌ 仍 0%（F1+F4 都生效但没解根因——机器人微动 0.118 但方向错）
- **关键坑**: F1 初版只改 `n_act=min(2)` 但下游 `T_chunk=req.steps=9` 把 2 步 tile 填充回 9 步绕过 F1，需同步硬覆盖 `T_chunk=min(2, req.steps)` 才真生效
- **状态**: ✅ F1+F4 实施验证生效，❌ 闭环仍 0%（根因在模型推理本身输出错方向）

### P4.16 真位轨迹定论：模型推理本身输出错方向是闭环 0% 真根因（2026-07-23）
- **根因**: 直接抓 `env.env.robots[0].controller.ee_pos` 每步真值轨迹印机器人真动了（40 步总位移 0.118，朝 z+ 方向微动），推翻"proprio 不变=机器人没动"误判（debug log 抓 chunk 边界值不是逐步真值）。但 chunk 首步 delta_pos 每 chunk 都朝固定负 z 方向 `[-0.7~-0.9, ±0.02, -0.9~-0.96]`（z 维 delta 全负），env controller scale_action 缩放后微动但**方向错**——模型推理本身输出错方向是真根因，F1 缩短 chunk + F4 float32 推理都没解
- **外围根因全部排查正确**: F1 chunk 缩到 2 步生效、F4 float32 推理生效、P4.8 ori6d 修复真生效、图像 flip 对齐训练、env act 执行正确、proprio 原值回传不滚雪球、推理链路正确、state discretize clip 链路一致、task convention 不影响、图像每步真值不首帧复用、env controller scale_action 缩放比真值 0.05、训练-推理 action convention 一致
- **剩余根因方向**: D 模型权重本身在 NPU float32 上对当前输入的预期输出就是错方向（需对比 cann-recipes 官方 GPU 推理输出方向验证）/ E 训练数据采集时 action 真值 sign/axis convention 与推理时不一致（需对比训练采集脚本 action 字段 sign convention）/ G pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏（需对比 cann-recipes 官方 GPU flow matching 推理输出轨迹验证）
- **状态**: ⏳ 待用户拍板剩余根因方向 D/E/G

### P4.17 方向 E 推翻：sign convention 不是根因（2026-07-23）
- **根因排查**: 精对比训练-推理 action sign convention——训练数据采集脚本 `lerobot libero env libero.py:316 step(self, action)` 直接 `self._env.step(action)` 执行（lerobot env 不做 sign 翻转）+ unnorm stats action 真值分布（273465 个训练样本）印：
  - delta_pos x 维 q50=+0.0255 **|q50|<0.05** 近对称分布（不是真偏正 sign convention，q50 偏移是均值噪声），推理返 x 维=-0.7~-0.9（负）在训练分布内（q01=-0.5352 负方向有数据）
  - delta_pos z 维 q50=-0.0737 **|q50|>0.05** 真偏负 sign convention，推理返 z 维负与训练一致
- **定论**: **E 推翻**——sign convention 不是根因。训练数据 delta_pos 近对称分布，推理返固定负方向在训练分布内，是**模型输出偏极端**（训练 z 维 q50=-0.0737 但推理 -0.9~-0.96 偏极端）不是 convention 不一致
- **剩余根因方向**: 转 D（模型权重本身在 NPU float32 上对当前输入的预期输出就是错方向，需对比 cann-recipes 官方 GPU 推理输出方向验证）/ G（pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，需对比 cann-recipes 官方 GPU flow matching 推理输出轨迹验证）。D/G 均需 GPU 环境——若本 NPU 服务器无 GPU，需申请 GPU 环境或 cann-recipes 官方推理输出基准数据对比
- **状态**: ✅ E 推翻定论，⏳ 待用户拍板 D/G（需 GPU 环境）

---

## 七、关键教训（持续追加）

1. **pi0.5 ≠ pi0**：pi0.5 是基于 pi0 的开放世界泛化版本，输出 action chunk shape `[50, action_dim]`，通过 lerobot 框架的 `PI05Policy.select_action`/`predict_action_chunk` 推理，不能简单套用 OpenVLA 的 transformers 加载路径。
2. **lerobot 框架依赖重**：lerobot 拖 gym / datasets / diffusion 等依赖，可能与 OSMesa 环境冲突，必须用独立 conda env 隔离。
3. **cann-recipes 是金矿**：华为官方 cann-recipes-embodied-ai 仓库已提供 pi0.5 在 Ascend NPU 的推理示例（含 patch + 精度验证脚本），应最大化复用而非重造轮子。
4. **HF gated repo 在 NPU 服务器代理阻断**：`google/paligemma-3b-pt-224` 需 token + HF 直连被代理阻断，hf-mirror 也只返 redirect；改用 modelscope 镜像下载 + 离线模式是最稳方案。
5. **Ascend 910B4 的 aclnnNormalFloatFloat 不支持 BFLOAT16 输出**：torch.normal 在 NPU 上用 bf16 dtype 会崩，必须强制 fp32 采样再 .to(bf16)。
6. **chunk 语义错配是 0% 成功率隐蔽根因**：client.step 期望多步 chunk（切片访问），server 不能用 select_action（每次返 1 步），必须用 predict_action_chunk 一次返 n_action_steps 步。
7. **环境变量必须在 import 前 export**：MUJOCO_GL=osmesa / LD_LIBRARY_PATH 含 render_libs 等，在 python 内 os.environ.setdefault 设已晚，必须在 bash 启动前 export。
8. **❌ "task0 持续 0% 是模型真实表现"是过早判断（2026-07-21 推翻）**：100ep 全 0% 远低于官方必有系统性根因。重读 client 才发现 proprio 被 delta 污染。教训：成功率远低于官方时**不应过早判为"模型真实表现"**，必须定位到具体代码 bug 才能下结论。
9. **robosuite quat convention 是 (x,y,z,w) 不是 [w,x,y,z]**（2026-07-21）：`mat2quat`/`quat2mat`/`axisangle2quat` 全套用 `(x,y,z,w)`，单位阵 → `[0,0,0,1]` 表示 `xyz=0, w=1`。server_v2.py:129 注释写错为 `[w,x,y,z]` 但数值传入正确——注释 bug 不影响运行，但易误导诊断。
10. **libero_client.py 的 proprio 设计期望 server 返绝对值**（2026-07-21）：`self.proprio[:9] = action[-1, :9]` 注释"last absolute state"——delta action 模型（pi0.5/OpenVLA）走此 client 时 proprio 会被污染。**这是 client 官方便样的固有问题，不是 server bug**；server 端需绕过（返绝对末步 pose 给 client 赋值，或 server 不信 proprio 自取 env 真值）。
11. **conda env 易被外部清理**（2026-07-21 再次发生）：lerobot-pi05 env + 源码目录同时消失。下次开工先 `conda env list` + `ls lerobot_pi05` 验证存在性，缺则先重建再继续。重建脚本应幂等可重复。
12. **"不改 libero_client.py 官方便样"约束 vs client 固有 bug 是死结**（2026-07-21）：client 发的 proprio ori6d 与 env 真值 ori6d 不一致（P4.8），是 client 官方便样的固有问题。所有 server 端绕过路径都需破"不改 client"约束或破 server 无状态 HTTP 设计——需用户拍板才能继续，不应擅自破约束。
13. **不要被单一根因假设绑死**（2026-07-21）：P4.5→P4.7→P4.8 三轮根因假设，每轮修复后闭环仍 0% 才暴露更深根因。应在每轮修复后立即 sanity test 验证根因是否真解（归一化落 [-1,1] + 输出量级 ±0.1），而非直接重跑闭环浪费 5h。
14. **robosuite quat convention 有两套**（2026-07-21）：`mat2quat`/`quat2mat` 用 `[x,y,z,w]`，但 raw obs `robot0_eef_quat` 用 `[w,x,y,z]`——从 raw obs 取 quat 必须先 wxyz→xyzw 转换再传 quat2mat。
15. **client proprio ori6d 与训练分布 ori6d 列序调换+反号是闭环 0% 真根因**（2026-07-23 印证）：client 发的 proprio ori6d 来自 env `controller.ee_ori_mat → Mat_to_Rotate6D`，训练分布 ori6d 来自 `robot0_eef_quat wxyz → quat2mat → Mat_to_Rotate6D`，两源 ori6d 列序调换+反号完全不同（diff=1.0018）。精定位规律 `perm=(3,2,5,4,1,0) signs=(1,-1,-1,-1,1,-1)`（8 组 ori 验证普适 diff<0.01），server 侧绕过不改 client 官方便样。教训：**两源 rot mat（controller ee_ori_mat vs sim body quat）看似同值但 Rotate6D 列序不同**，必须枚举 perm+signs 验证规律普适性才能修，不能单组数据过拟合。
16. **"chunk 10 步同方向是 bug"是过早判断**（2026-07-23 推翻）：pi0.5 flow matching 模型对当前 obs 一次推理预测 10 步 chunk，chunk 内不重新抓 obs，10 步 delta 同方向是 expected 预测轨迹特性（训练 action stats q01+q99≈0 对称分布印证）。教训：**chunk 内同方向不是 bug，chunk 之间方向错才是根因**——诊断时区分 chunk 内（expected）vs chunk 之间（根因）两个尺度。
17. **外围根因全部排查正确但闭环仍 0% 时根因在模型推理本身**（2026-07-23 定论）：#1-#13 全部诊断链印外围链路（图像 flip/env act/proprio/推理链路/state discretize/task convention/图像更新）全排查正确，闭环仍 0% 说明根因在模型推理本身返固定方向 delta。教训：**当外围根因全部推翻后，应转向对比 cann-recipes 官方 fp16 推理输出方向**（验证是否 NPU bf16 精度根因）或**对比训练数据采集时 action convention vs 推理时链路**（验证是否训练-推理 convention 不一致让模型学到错方向），不应继续在外围链路反复诊断浪费算力。
18. **bf16 精度是 NPU 推理输出方向偏的根因之一但非唯一**（2026-07-23 方向 A 印证）：同输入 bf16 vs float32 推理 chunk 10 步累加末步 diff=0.36 > 0.05，两 dtype 输出方向不同。但两 dtype chunk 都朝同方向飞出 workspace（z 都 <<0.8），说明 **bf16 粀度不是唯一根因**，float32 也输出错方向。教训：**dtype 对比定论 bf16 根因后不能止步**，必须继续查 chunk 累加飞出（方向 C）等并行根因，闭环 0% 常是多根因组合。
19. **chunk 10 步 delta 同方向累加飞出 workspace 是 rel 模式固有风险**（2026-07-23 方向 C 定论）：env controller `action_scale[pos3]=0.05` 单步缩放安全 ±0.048，但 chunk 内不重新抓 obs 一次性预测 10 步 delta 都朝同方向，10 步累加位移 z=-0.48 飞出 workspace 下界 0.8。教训：**rel 模式 chunk 累加位移 = 单步缩放 × chunk 步数**，必须验证 `单步缩放 × chunk 步数 < workspace 范围`才安全；修复优先缩短 chunk（F1：chunk 从 10 步缩到 1-2 步，client 每步重新推理抓新 obs）而非缩 output_max（F2 影响其他模型需隔离）。
20. **F1 缩短 chunk 需同步硬覆盖 n_act + T_chunk 两点**（2026-07-23 F1 实施坑）：初版只改 `n_act=min(2, policy.config.n_action_steps)` 缩切片到 2 步，但下游 `T_chunk=req.steps=9` 把 2 步 tile 填充回 9 步绕过 F1，debug log 印 `return chunk shape=(9,10)` 不是 `(2,10)`。教训：**chunk 长度由切片（n_act）+ 填充（T_chunk）两点控制，缩 chunk 必须同步硬覆盖两点**，否则下游 tile 填充绕过修复。
21. **"proprio 不变=机器人没动"是 debug log 抓 chunk 边界值的误判**（2026-07-23 #18 印证）：debug log 印 proprio 每步完全不变（末步[:9]=`[-0.211,...]`每步相同）像机器人没动，但直接抓 `env.env.robots[0].controller.ee_pos` 每步真值轨迹印机器人真动了（40 步总位移 0.118 朝 z+ 方向微动）。教训：**debug log 抓的 proprio 是 chunk 边界值（每 chunk 才更新一次）不是逐步真值**，判断机器人是否真动必须直接抓 controller.ee_pos 每步真值轨迹，不能用 debug log 的 proprio 边界值误判"没动"。
22. **训练 action q50 偏移 ≠ sign convention 根因**（2026-07-23 方向 E 印证）：训练 delta_pos x 维 q50=+0.0255 mean=+0.0628 看似偏正，但 |q50|<0.05 近对称分布（q01=-0.5352 q99=+0.7095 对称范围），q50 偏移是均值噪声不是真偏正 sign convention。教训：**判断训练 action 是否真偏某方向 sign convention 必须看 |q50|>0.05 阈值**（q50 偏移超 5% 分布才算真偏），不能因 mean/q50 偏正就误判 sign convention 根因；推理返固定负方向在训练对称分布内是**模型输出偏极端**不是 convention 不一致。

---

### P4.18 模型真伪 + 推理链路 + 闭环流程三方面精查（2026-07-23 定论）

- **触发**: 用户要求先验证 pi0.5 模型真伪 + libero 仿真流程与 X-VLA 是否一致，再谈剩余根因
- **方面 1 模型真伪**: ✅ **是真 pi0.5 flow matching 架构**——`action_expert_variant: gemma_300m` + `chunk_size: 50` + `n_action_steps: 10` 与 lerobot_pi05 `configuration_pi05.py:36` 官方基准完全一致；源码注释明示 "see openpi"（PaliGemmaWithExpertModel + Gemma + flow matching），是 OpenPI 官方 pi0.5 架构的 lerobot 复刻。但 ckpt 来自 `jade_choghari/pi05-t5` 个人 repo 微调版（`pretrained_path="lerobot/pi05_libero_finetuned"` + `job_name="libero_training_fast"` + `dataset.root="/fsx/jade_choghari/data/libero"` AWS SageMaker 训练），**无 trainer_state.json** 无训练 loss 收敛记录
- **方面 2 推理链路**: ✅ **一致**——server `preprocess→predict_action_chunk→postprocess` 与 X-VLA 官方同链路（#6 印 diff=0.156）；preprocessor norm_map `ACTION:MEAN_STD` 与 server 归一化对齐
- **方面 3 闭环流程**: ⚠️ **存在两处不一致**——① X-VLA 官方便样 `use_delta=False` abs 模式，我们 eval 用 `act_type="rel"` rel 模式（abs 闭环验证仍 0% 说明非根因）；② **关键新发现**：X-VLA 官方便样 `npu_e2e_verify.py:146` `proprio[:9]=action_raw[-1,:9].copy()` **proprio 直接用 server 返的 chunk 末步 ori6d 覆盖**，我们 server P4.5 v2 末步 proprio 原值回传（chunk 末步 [:9]=proprio 原值）——ori6d convention 不一致
- **abs 模式闭环验证**: 跑 `eval_pi05_task0.py --act_type abs` 1 episode → **success=0.0 耗时 244.8s 仍 0%**，abs 模式 proprio 每步也不变（末步[:9]=`[-0.211,...]`每步相同）。**定论**：abs vs rel 流程不一致**不是根因**——两种模式闭环均 0% 且机器人都没真动
- **根因定论收敛**: 模型架构真 + 推理链路一致 + abs/rel 流程不一致但均 0%（非根因）→ **真根因仍是模型推理本身输出错方向**（每 chunk 首步 delta_pos 朝固定负 z，#18 已定论）。**新线索**：ckpt 是个人微调版无训练 loss 收敛记录，需后续查微调模型训练质量是否根因
- **状态**: ✅ 三方面精查完整定论

23. **"模型推理输出错方向"根因下应查 ckpt 训练质量是否根因**（2026-07-23 P4.18 印证）：三方面精查定论模型架构真 + 推理链路一致 + abs/rel 流程不一致非根因后，真根因仍是模型推理本身输出错方向。但发现 ckpt 来自 `jade_choghari/pi05-t5` 个人 repo 微调版（AWS SageMaker 训练），**无 trainer_state.json** 无训练 loss 收敛记录。教训：**当根因收敛到"模型推理本身"时，必须查 ckpt 训练质量**（loss 是否收敛 / 训练数据 convention 是否对齐 / 微调基模型是否正确），不能止步于"架构真推理链路一致"就判模型本身根因——**架构真≠训练成功**，个人微调版 ckpt 可能训练失败导致推理输出错方向，这是比"NPU 算子差异"更基础的根因方向。

---

### P4.19 方向 H 精查定论：ckpt 来源错配是根因方向（2026-07-24 定论）

- **触发**: 教训 #23 印证"架构真≠训练成功"，查 ckpt 训练质量是否根因
- **对比铁证**:
  - 我们 ckpt：`repo_id="jade_choghari/pi05-t5"`（个人 AWS SageMaker 微调）+ `pretrained_path="lerobot/pi05_libero_finetuned"` + `job_name="libero_training_fast"` + `output_dir="/fsx/jade_choghari/outputs/..."` + **无 trainer_state.json**
  - 官方便样基准：`lerobot/pi05-libero`（lerobot 官方 LIBERO 微调，4B safetensors 含 norm stats + processor，达 97.5%）
  - cann-recipes 官方便样：`modelscope.cn/models/lerobot/pi05_base.git` commit `d856522`（base 模型无 LIBERO 微调，只验证推理能跑通不验证闭环成功率）
- **颠覆性定论**:
  1. PI05_TRACKING.md §4.2 冰示决策选 `lerobot/pi05-libero`，但**实际加载的 ckpt 是 `jade_choghari/pi05-t5`（个人微调版）不是官方便样**——**ckpt 来源错配是根因方向**！
  2. PROJECT_TRACKING.md:758 明示决策"改用 `lerobot/pi05-libero` 直接对比官方基准"，但**实际加载的是 jade_choghari/pi05-t5**，与决策不符
  3. server_v2.py:354 `PI05Policy.from_pretrained(args.model_path)` + `run_pi05_spatial.sh:22` `--model_path /home/ma-user/work/pi05_libero_ckpt`——确认加载的是个人微调版不是官方便样
  4. 量级差距印证：官方便样 96.85% / 97.5% vs 我们 0%，量级差距远超 NPU 精度差异范围，指向 ckpt 本身问题而非推理链路
- **根因定论收敛**: 模型推理本身输出错方向（#18 定论）的根因可能是 **ckpt 来源错配**——加载了个人微调版（可能训练失败/数据 convention 错位）而非官方便样 ckpt
- **下一步修复方向**: F5（下载 `lerobot/pi05-libero` 官方便样 ckpt 替换重跑闭环验证是否 >0%）/ H2（联系 jade_choghari 确认微调版训练质量或查 HF repo card）
- **状态**: ✅ 方向 H 定论 ckpt 来源错配是根因方向，⏳ 待用户拍板 F5/H2

24. **"官方便样决策 ckpt vs 实际加载 ckpt"必须一致性校验**（2026-07-24 方向 H 印证）：PI05_TRACKING.md §4.2 冰示决策选 `lerobot/pi05-libero`，但实际加载的是 `jade_choghari/pi05-t5`（个人微调版），与决策不符导致闭环 0% 而非官方便样 97.5%。教训：**项目说明文档记录的"决策选 X ckpt"必须在实施时校验实际加载的 ckpt 是否真 X**——`config.json` 的 `repo_id` / `pretrained_path` 是 ckpt 来源真值铁证，不能用项目文档的决策记录代替实际加载校验；**ckpt 来源错配是比推理链路/NPU 精度更基础的根因方向**，量级差距（0% vs 97%）远超精度差异范围时应优先查 ckpt 是否同源而非外围链路。

---

### P4.20 F5+H2 组合精查定论：ckpt 来源错配根因被推翻，真根因在源码版本不匹配（2026-07-24 定论）

- **触发**: 方向 H 定论 ckpt 来源错配是根因方向后，用户拍板 F5+H2 组合
- **F5 下载可行性**: HF 直连被代理阻断（`ProxyError huggingface.co port 443`），但 modelscope 镜像可用——`lerobot/pi05-libero` modelscope 镜像存在（CreatedAt 2026-03-05，Downloads 139，Owner=lerobot 官方），modelscope download 实测 18-40MB/s，6 分钟完成 9.35GB model.safetensors + 完整 processor/norm stats
- **H2 联网路径定论**: `jade_choghari/pi05-t5` modelscope 镜像**不存在**（`NotExistError 404`），HF 直连被代理阻断无法查 repo card 训练质量——**H2 联网路径断了**
- **F5 官方便样 ckpt 加载失败——颠覆性定论**: 启动 server 加载官方便样 ckpt 报错 `draccus.utils.DecodingError: The fields use_peft are not valid for PI05Config`——官方便样 ckpt config.json 含 `use_peft` 字段（False），我们 lerobot_pi05 源码 PI05Config 不认此字段。**根因**：官方便样 ckpt 与我们 lerobot_pi05 源码版本不匹配
- **颠覆性铁证：ckpt 来源错配根因被推翻**: 官方便样 ckpt 与我们 ckpt **完全同源**——size 同 9354050752（9.35GB）+ �首字节同 `7820 0200` + `repo_id` 同 `jade_choghari/pi05-t5` + `pretrained_path` 同 `lerobot/pi05_libero_finetuned`，唯差异是官方便样 config.json 多一个 `use_peft` 字段。**定论**：官方便样 ckpt 就是我们 ckpt 同源！ckpt 来源错配根因被推翻，真根因在**源码版本不匹配**
- **根因定论收敛**: ckpt 来源错配根因**被推翻**（官方便样 ckpt 与我们 ckpt 完全同源），真根因转向 **lerobot_pi05 源码版本与 ckpt 不匹配**（源码 PI05Config 不认 ckpt config.json 的 `use_peft` 字段）。这解释了为什么 ckpt 是个人微调版但官方便样也用同 ckpt——**官方便样 ckpt 本就来自 jade_choghari/pi05-t5 微调版**，lerobot 团队上传到 `lerobot/pi05-libero` repo 但 ckpt 真值是 jade_choghari/pi05-t5
- **下一步修复方向**: F6（升级 lerobot_pi05 源码 PI05Config 加 `use_peft` + `use_amp` 字段兼容官方便样 ckpt 重跑闭环）/ F7（直接用 cann-recipes 官方便样 `modeling_pi05.patch` 补丁已含 `use_peft` 字段兼容重跑闭环）/ G（需 GPU 环境）
- **状态**: ✅ F5+H2 组合定论 ckpt 来源错配根因被推翻，真根因在源码版本不匹配，⏳ 待用户拍板 F6/F7/G

25. **"ckpt 来源错配根因"必须用 checksum/size/首字节验证同源性而非仅看 repo_id**（2026-07-24 F5+H2 印证）：方向 H 定论 ckpt 来源错配是根因方向（我们 `jade_choghari/pi05-t5` vs 官方便样 `lerobot/pi05-libero` repo_id 不同），但 F5 下载官方便样 ckpt 后对比 size 同 9354050752 + 首字节同 `7820 0200` + repo_id 同 `jade_choghari/pi05-t5`——**官方便样 ckpt 就是我们 ckpt 同源**，ckpt 来源错配根因被推翻。教训：**判断 ckpt 是否同源必须用 checksum/size/首字节等文件级验证**，不能用 repo_id/pretrained_path 等元数据字段判断——lerobot 团队可能把同一 ckpt 上传到多个 repo（`lerobot/pi05-libero` 和 `jade_choghari/pi05-t5` 是同 ckpt 不同 repo），元数据 repo_id 不同不代表 ckpt 不同源；**真根因在源码版本不匹配**（源码 PI05Config 不认 ckpt config.json 的 `use_peft` 字段）比 ckpt 来源错配更隐蔽，ckpt 加载失败报 `DecodingError` 时应优先查源码版本兼容性而非 ckpt 来源。

---

### P4.21 查 cann-recipes patch 真值定论：F7 不可行，F6 是唯一路径（2026-07-24 定论）

- **触发**: F5+H2 组合定论真根因在源码版本不匹配（源码 PI05Config 不认 ckpt config.json 的 `use_peft` 字段）后，用户拍板"先查 cann-recipes patch 真值"确认 F7 是否可行
- **cann-recipes patch 真值定论**: patch 大小 4898 字节 / 100 行，**只改 `modeling_pi05.py`（8 处 NPU dtype 适配 hunk），不改 `configuration_pi05.py`**；grep `use_peft` / `use_amp` / `peft` / `lora` / `adapter` 全空——**patch 不含 `use_peft` 字段兼容改动**；patch 真值是 **NPU dtype 适配补丁**（dtype 跟随参数 `_inference_dtype` 方法、npu 不支持 float64、`sample_noise`/`sample_time`/`suffix_out` dtype 跟随参数），不是 ckpt 兼容补丁
- **F7 不可行定论**: cann-recipes patch 不加 `use_peft` 字段，无法解官方便样 ckpt 加载报 `DecodingError: fields use_peft are not valid for PI05Config` 问题——**F7 被推翻**，用 cann-recipes patch 重跑闭环不能解 ckpt 加载失败问题
- **我们源码已应用 patch 铁证**: `modeling_pi05.py:623` 已有 `_inference_dtype` 方法 + `:65` 已有 `device_type == "npu"` 分支——patch 关键 hunk 全在源码，说明我们 lerobot_pi05 源码已应用 cann-recipes modeling_pi05.patch，patch 不需重应用
- **F6 是唯一可行路径**: 官方便样 ckpt config.json 只多 `use_peft: False` 一个字段（不是 use_peft + use_amp 两个字段，方向 H 初查时误记 use_amp 实际只有 use_peft）——**F6 只需在 `configuration_pi05.py` PI05Config 加 `use_peft: bool = False` 字段**即可兼容官方便样 ckpt 加载，F6 实施后重跑闭环验证是否 >0% 定论源码版本不匹配是否真根因
- **状态**: ✅ 查 cann-recipes patch 真值定论 F7 不可行 F6 是唯一路径，⏳ 待用户拍板 F6 实施

26. **"官方便样 patch 解 ckpt 兼容问题"假设必须 grep 验证 patch 真值含目标字段改动**（2026-07-24 查 cann-recipes patch 印证）：用户拍板"先查 cann-recipes patch 真值"确认 F7 是否可行，本以为官方便样 patch 可能含 `use_peft` 字段兼容改动可解 ckpt 加载 `DecodingError` 问题，但 grep 验证 patch 真值不含 `use_peft` / `use_amp` / `peft` / `lora` 任何字段改动——patch 只改 `modeling_pi05.py`（NPU dtype 适配）不改 `configuration_pi05.py`。教训：**"官方便样 patch 解 ckpt 兼容问题"假设必须 grep 验证 patch 真值含目标字段改动**，不能用"官方便样 patch 应该兼容官方便样 ckpt"的逻辑推断代替 grep 验证；**patch 用途需精读 hunk 内容判断**（cann-recipes patch 真值是 NPU dtype 适配补丁不是 ckpt 兼容补丁），不能凭 patch 文件名（`modeling_pi05.patch`）推断用途；**我们源码是否已应用 patch 需 grep patch 关键 hunk**（`_inference_dtype` 方法 / `device_type == "npu"` 分支）**验证**，不能凭项目文档记录推断——本项目源码已应用 cann-recipes patch 但文档未明示。

---

### P4.22 F6 实施 + 闭环验证定论：源码版本不匹配不是闭环 0% 真根因（2026-07-24 定论）

- **触发**: 查 cann-recipes patch 真值定论 F7 不可行 F6 是唯一路径后，用户拍板"实施 F6"——升级 lerobot_pi05 源码 PI05Config 加 use_peft 字段兼容官方便样 ckpt
- **F6 实施步骤**:
  1. 加字段：`configuration_pi05.py:82` PI05Config "Finetuning settings" 段加 `use_peft: bool = False  # Whether to use PEFT (LoRA) adapters — added for ckpt config.json compatibility`
  2. 语法检查：`python3 -c "from lerobot.policies.pi05.configuration_pi05 import PI05Config; c=PI05Config(); print(c.use_peft)"` → `use_peft: False` 语法 OK
  3. 加载验证：`PI05Policy.from_pretrained("/home/ma-user/work/lerobot_pi05_libero_official")` → `STEP4_load_OK_DecodingError解` + `All keys loaded successfully!` + `use_peft: False chunk: 50 n_act: 10`——**F6 修复真解了 DecodingError**
  4. 闭环验证：跑 `eval_pi05_task0.py --act_type rel` 1 episode → **success=0.0 耗时 236.6s 仍 0%**
- **F6 实施坑（环境变量 + tokenizer + compile 三连）**:
  - 坂 1：libhccl.so torch_npu 后端加载失败——setsid/nohup 子 shell 没继承完整 LD_LIBRARY_PATH，报 `ImportError: libhccl.so cannot open shared object file` + `RuntimeError: Failed to load backend extension: torch_npu`。解决：用 run_pi05_spatial.sh 基准脚本环境变量（当前 shell env 已验证 torch_npu OK）
  - 坂 2：tokenizer 联网阻断——官方便样 ckpt processor 依赖 `google/paligemma-3b-pt-224` tokenizer（HF 离线模式找不到缓存报 `OSError: couldn't connect to huggingface.co`）。解决：把 `policy_preprocessor.json` 的 `tokenizer_name` 从 `google/paligemma-3b-pt-224` 改为本地 `/home/ma-user/work/paligemma3b_hf`（绕过 HF 联网，与我们 ckpt 同用本地路径）
  - 坂 3：compile inductor 缺 torch_mlir——官方便样 ckpt `compile_model: True` 启用 torch.compile inductor backend，NPU 环境缺 torch_mlir 报 `ImportError: torch_mlir is not installed`。解决：把 `config.json` 的 `compile_model` 从 `True` 改为 `False`（与我们 ckpt 基准一致）
- **F6 闭环验证结果——颠覆性定论**: 闭环仍 0%（success=0.0 耗时 236.6s），但推理链路跑通（`return chunk shape=(2,10)` + P4.8 ori6d fix perm+signs 转换 + P4.5 v2 fix 末步 proprio 原值回传全生效）。官方便样 ckpt chunk 首步 delta_pos 每步朝固定负 z 方向 `[-0.88~-0.80, +0.04~-0.01, -0.97~-0.90]`——与 #18 定论"模型推理本身输出错方向"完全一致。**定论**：源码版本不匹配**不是闭环 0% 真根因**——F6 修复让官方便样 ckpt 能加载且推理链路跑通，但闭环仍 0% 且 chunk 首步 delta_pos 朝固定负 z，真根因仍在模型推理本身（#18 定论未被推翻）
- **根因定论彻底收敛**: 已推翻的根因方向：ckpt 来源错配（#3 铡证同源）/ 源码版本不匹配（F6 修复后仍 0%）/ abs vs rel 流程不一致（abs 闭环仍 0%）/ sign convention（方向 E 推翻）/ bf16 精度（float32 也错方向）/ chunk 累加飞出（F1 缩 chunk 后仍 0%）。**真根因仍是模型推理本身输出错方向**（#18 定论未被推翻）——官方便样 ckpt（与我们 ckpt 完全同源）在 F6 修复后推理仍返固定负 z 方向 delta_pos。**新铁证**：官方便样 ckpt（repo_id `jade_choghari/pi05-t5`，lerobot 团队上传到 `lerobot/pi05-libero` repo）推理返固定负 z 方向——说明不是"个人微调版训练失败"，是**官方便样 ckpt 本身在我们推理链路下输出错方向**
- **下一步根因方向**: G（pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，需 GPU 环境）/ D（对比 cann-recipes 官方 GPU 推理输出方向，需 GPU 环境）/ I（对比官方便样 ckpt 在 cann-recipes 官方 run_pi05_example.py 推理输出方向，本地 NPU 可跑）
- **状态**: ✅ F6 实施定论源码版本不匹配不是闭环 0% 真根因，⏳ 待用户拍板 G/D/I

27. **"官方便样 ckpt 加载成功 + 推理链路跑通 ≠ 闭环 >0%"**（2026-07-24 F6 印证）：F6 修复让官方便样 ckpt 加载成功（`All keys loaded successfully!`）+ 推理链路跑通（`return chunk shape=(2,10)` + P4.8/P4.5 fix 全生效），但闭环仍 0%（success=0.0）且 chunk 首步 delta_pos 朝固定负 z 方向。教训：**ckpt 加载成功 + 推理链路跑通只验证 schema/processor/dtype 链路正确，不验证推理输出方向正确**——闭环 0% 真根因在推理输出方向（模型返固定负 z delta_pos）而非加载/链路问题；**判断 F6 类修复是否真根因不能止步于"加载成功+链路跑通"**，必须重跑闭环验证 chunk 首步 delta_pos 方向是否改变，否则会把"加载成功"误判为根因已解。

---

### P4.23 查 I 可行性定论：cann-recipes 官方便样用 select_action+合成图，与 server 链路不同（2026-07-24 定论）

- **触发**: F6 实施定论源码版本不匹配不是闭环 0% 真根因后，用户拍板"先查 I 可行性"——确认 cann-recipes 官方便样 `run_pi05_example.py` 能否用本地源码+官方便样 ckpt 跑通对比推理输出方向
- **cann-recipes 官方便样 vs 我们 server 推理链路关键差异**:
  - 推理 API：cann-recipes 官方便样用 `select_action`（每步 1 次返 1 步），我们 server_v2.py 用 `predict_action_chunk`（返 chunk 多步）——server_v2.py:182 注释明示"不能用 select_action（queue 滚动每次只返 1 步导致 client 切片错位→0%）改用 predict_action_chunk"
  - 输入数据：cann-recipes 官方便样用 `make_dummy_observation` 合成图（torch.randint 假图像），我们 server 用真实 env obs（agentview+wrist 图像）——官方便样只验证推理能跑通不验证真实图像输出方向
  - 加载链路 + pre/post：两链路一致（`PI05Policy.from_pretrained` + `make_pre_post_processors` + `pipeline.preprocess`/`postprocess`）
- **依赖环境本地齐全**: ✅ infer_utils（cann-recipes 本地）+ ✅ 官方便样 ckpt（F5 下载完成 + F6 修复 use_peft + compile_model=False + tokenizer 本地 paligemma3b_hf）+ ✅ paligemma tokenizer 本地缓存 + ⚠️ lerobot 依赖需 conda env 激活（裸 python 报 `ModuleNotFoundError: No module named 'lerobot'`，激活 lerobot-pi05 env 后可用）
- **I 可行性定论**: cann-recipes 官方便样用 `select_action` + 合成图，与我们 server 用 `predict_action_chunk` + 真实图链路不同——**直接跑官方便样脚本对比输出方向意义有限**（合成图无真实任务信号 + select_action 与 predict_action_chunk 返不同步数）。**关键洞察**——cann-recipes 官方便样用 `select_action` 能跑通说明此 API 在 NPU 上可用，我们 server 改用 `predict_action_chunk` 是因 client 切片错位，两 API 推理输出方向应一致（同模型同输入）。**I 部分可行**：可以跑 cann-recipes 官方便样脚本验证 NPU 推理链路能跑通（已验证），但**不能定位闭环 0% 真根因**（合成图无真实任务信号 + 链路不同）
- **下一步根因方向**: G（pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，需 GPU 环境）/ D（对比 cann-recipes 官方便样 GPU 推理输出方向，需 GPU 环境）/ I2（改 cann-recipes 官方便样 `run_pi05_example.py` 用真实 env obs + 跑 select_action 推理输出方向对比我们 server predict_action_chunk，验证是否两 API 输出方向一致，本地 NPU 可跑）
- **状态**: ✅ 查 I 可行性定论 I 部分可行但不能定位闭环 0% 真根因，⏳ 待用户拍板 G/D/I2

28. **"官方便样脚本能跑通 ≠ 能定位闭环 0% 真根因"**（2026-07-24 查 I 可行性印证）：cann-recipes 官方便样 `run_pi05_example.py` 用 `select_action` + `make_dummy_observation` 合成图能跑通验证 NPU 推理链路可用，但合成图无真实任务信号 + select_action 与我们 server `predict_action_chunk` 返不同步数——直接跑官方便样脚本对比输出方向意义有限，不能定位闭环 0% 真根因。教训：**官方便样脚本能跑通只验证推理链路 schema/dtype/API 可用，不验证真实图像输入下推理输出方向正确**——对比推理输出方向必须用**同输入**（真实 env obs）+**同 API**（select_action vs predict_action_chunk 验证两 API 输出方向一致性），不能用合成图官方便样脚本代替真实闭环验证；**cann-recipes 官方便样脚本用途是 NPU 推理性能/链路验证不是闭环成功率验证**（PI05_TRACKING.md:131 已明示"cann-recipes pi05_model base 模型无 LIBERO 微调只验证推理能跑通不验证闭环成功率"），不能凭"官方便样脚本能跑通"推断闭环根因已解。

---

### P4.24 I2 实施 + 源码精读定论：两 API 输出方向必然一致，根因不在 API 差异（2026-07-24 定论）

- **触发**: 查 I 可行性定论 I 部分可行但不能定位闭环 0% 真根因后，用户拍板"I2"——改 cann-recipes 官方便样脚本用真实 env obs + 跑 select_action 推理输出方向对比我们 server predict_action_chunk
- **I2 实施进展**:
  - 写 I2 验证脚本 `scripts/pi05/i2_real_obs_infer.py`：抓真实 libero spatial task0 首帧 obs + 加载官方便样 ckpt + preprocess + select_action 推理 + 打印输出方向对比 server debug log chunk 首步 delta_pos
  - 建 env API 调试 3 连坑：坑 1 `Benchmark.get_task(0)` missing arg `i` → benchmark 是 ABCMeta 类需 `benchmark()` 实例化；坑 2 `LIBERO_SPATIAL object has no attribute get_task_env` → benchmark 实例无此方法；坑 3 `OffScreenRenderCtrl` ImportError → libero 改名为 `OffScreenRenderEnv`。解决：用 X-VLA libero_client.py:249-263 基准建 env（`OffScreenRenderEnv(bddl_file_name=...)` + `env.reset()` + `env.set_init_state(init_states[0])`）
  - 抓真实 env obs 成功：`state: tensor([-0.2108, -0.0152, 1.1757, 0.9994, -0.0266, -0.0226, -0.0058, 0.0000])`——与 server debug log proprio raw `[-0.211, -0.011, 1.174, 0.0018, 0.9999, 0.0004, 0.9984, -0.0017, -0.0558]` **完全一致**（I2 抓的真值 obs 与 server 同源）
  - 推理卡 NPU OOM：`policy.to(torch.float32)` 报 `NPU out of memory`——server PID 533540 占 NPU 0 内存（29.49 GiB total / 9.99 GiB allocated / 12.21 MiB free），需停 server 释放 NPU 内存后重跑推理（pkill 命令被中断多次未完成）
- **源码精读定论——两 API 输出方向必然一致**: 精读 `modeling_pi05.py:1216-1230` select_action 源码铁证：`actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]`——**select_action 首步 = predict_action_chunk 首 chunk 首步**（同一次推理调用，源码 :1226 钁证）。select_action 只是加 queue 滚动每次 popleft 返 1 步，首步推理与 predict_action_chunk 首 chunk 首步是同一次调用。**两 API 输出方向必然一致**——I2 对比"两 API 输出方向"逻辑上无意义，无需重跑 I2 推理验证
- **I2 定论收敛**: 1.真实 env obs 与 server 同源（I2 抓的 state `[-0.2108, -0.0152, 1.1757, ...]` 与 server debug log proprio `[-0.211, -0.011, 1.174, ...]` 完全一致）；2.两 API 输出方向必然一致（`select_action` 首步 = `predict_action_chunk` 首 chunk 首步，同一次推理调用，源码 :1226 钁证）；3.**根因不在 API 差异**——真根因仍在模型推理本身输出错方向（#18 定论未被推翻）
- **下一步根因方向**: G（pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，需 GPU 环境）/ D（对比 cann-recipes 官方便样 GPU 推理输出方向，需 GPU 环境）/ J（改 I2 脚本用 CPU 推理绕过 NPU OOM 跑官方便样 ckpt 真实 obs 输出方向对比 server NPU 推理输出方向，验证是否 NPU 算子本身根因，本地 CPU 可跑慢）
- **状态**: ✅ I2 实施定论根因不在 API 差异，⏳ 待用户拍板 G/D/J

29. **"select_action vs predict_action_chunk 输出方向差异"假设必须精读源码验证而非重跑推理**（2026-07-24 I2 印证）：I2 假设两 API 输出方向可能不同需对比验证，但精读 `modeling_pi05.py:1226` 源码发现 `select_action` 内部就调 `predict_action_chunk`（`actions = self.predict_action_chunk(batch)[:, :n_action_steps]`），两 API 首步推理是同一次调用输出必然一致——I2 对比"两 API 输出方向"逻辑上无意义，无需重跑推理验证。教训：**"X vs Y 输出差异"假设必须精读源码验证 X/Y 是否真独立推理**，不能凭 API 名不同（select_action vs predict_action_chunk）推断输出可能不同；**lerobot 框架 select_action 是 predict_action_chunk 的 queue 包装**（首步同一次推理，后续步 popleft queue 滚动），不是独立推理路径——对比 API 输出方向前必须 grep 源码确认两 API 是否真独立调用，否则会浪费算力重跑逻辑上必然一致的推理。

---

### P4.25 J 实施 + CPU 推理路径不稳定定论：CPU vs NPU 对比路径断，需 GPU 环境（2026-07-24 定论）

- **触发**: I2 实施定论根因不在 API 差异后，用户拍板"走 J"——改 I2 戕本用 CPU 推理（绕过 NPU OOM）跑官方便样 ckpt 真实 obs 输出方向，对比 server NPU 推理输出方向，验证是否 NPU 算子本身根因（CPU vs NPU 输出方向差异）
- **J 实施进展**:
  - 改 I2 戕本用 CPU 推理：`policy.to('cpu')` + `move(batch, torch.device('cpu'), torch.float32)` 绕过 NPU OOM
  - 抓真实 env obs 成功（同 I2）：`state: tensor([-0.2108, -0.0152, 1.1757, ...])` 与 server proprio 同源
  - CPU 推理路径不稳定：9.35GB 大模型 CPU 推理极慢（10 分钟+未完成），进程被外部反复清理重启（PID 每次查都不同：646342→648151→650960→672146→691479→734273→734713），log 卡在抓 obs 后加载 ckpt 阶段（14 行无进展）
  - J 无法定论 NPU 算子是否根因——CPU vs NPU 对比路径断了
- **J 定论收敛**: 1.NPU 推理输出方向真值（#5 F6 已印铱证）：官方便样 ckpt chunk 首步 delta_pos 朝固定负 z `[-0.88~-0.80, +0.04~-0.01, -0.97~-0.90]`；2.CPU 推理路径不稳定：本服务器 CPU 推理 9.35GB 大模型极慢（10 分钟+未完成）+ 进程被外部反复清理重启，无法稳定跑完 CPU 对比；3.**J 无法定论 NPU 算子是否根因**——CPU vs NPU 对比路径断了，需 GPU 环境跑官方便样 GPU 推理输出方向对比 NPU（方向 D/G）才能定论
- **根因定论彻底收敛（8 阶段）**: 已推翻的根因方向（8 个）：ckpt 来源错配（#3 铑证同源）/ 源码版本不匹配（F6 修复后仍 0%）/ abs vs rel 流程不一致 / sign convention / bf16 精度 / chunk 累加飞出 / cann-recipes 官方便样脚本对比（合成图+不同 API）/ select_action vs predict_action_chunk API 差异（源码精读必然一致）。**真根因仍是模型推理本身输出错方向**（#18 定论未被推翻）——官方便样 ckpt（与我们 ckpt 完全同源）在 F6 修复后推理仍返固定负 z 方向 delta_pos。**新铱证**：官方便样 ckpt 推理返固定负 z 方向，说明不是"个人微调版训练失败"，是**官方便样 ckpt 本身在我们推理链路下输出错方向**
- **下一步根因方向**: G（pi0.5 flow matching 蒜声采样在 NPU 上轨迹偏，需 GPU 环境）/ D（对比 cann-recipes 官方便样 GPU 推理输出方向，需 GPU 环境）。**J/CPU 对比路径已断**（本服务器 CPU 推理 9.35GB 大模型不稳定），剩余根因方向 G/D 均需 GPU 环境跑官方便样 GPU 推理输出方向对比 NPU，**需用户拍板是否申请 GPU 环境**
- **状态**: ✅ J 实施定论 CPU vs NPU 对比路径断需 GPU 环境，⏳ 待用户拍板是否申请 GPU 环境走 G/D

30. **"CPU 推理绕过 NPU OOM 对比方向"路径在 9.35GB 大模型上不稳定**（2026-07-24 J 印证）：J 假设用 CPU 推理绕过 NPU OOM 跑官方便样 ckpt 真实 obs 输出方向对比 NPU，但本服务器 CPU 推理 9.35GB 大模型极慢（10 分钟+未完成）+ 进程被外部反复清理重启（PID 每次查都不同），无法稳定跑完 CPU 对比。教训：**CPU 推理绕过 NPU OOM 路径在 9.35GB 大模型上不实际**——pi0.5 是 4B 参数 + PaliGemma + Gemma 专家的大模型，CPU 推理首步需加载 9.35GB 权重 + 膜声采样 10 步 flow matching，耗时远超 NPU（NPU 推理秒级 vs CPU 推理 10 分钟+未完成）；**CPU vs NPU 对比路径断了时需转向 GPU 环境**（申请 GPU 跑官方便样 GPU 推理输出方向对比 NPU，方向 D/G），不能在本 NPU 服务器反复重试 CPU 推理浪费算力——CPU 推理不稳定是本服务器资源限制（进程被外部清理）非脚本 bug。

---

### P4.26 方向 D/G GPU 环境验证：GPU 服务器全端口不可达，方向 D/G 阻断（2026-07-25 定论）

- **触发**: J 实施定论 CPU vs NPU 对比路径断需 GPU 环境后，用户提供 RTX4090 24G×2 GPU 服务器（IP/端口/账号密码见团队内部记录，勿写入开源文档），走方向 D/G 对比官方便样 ckpt GPU 推理输出方向 vs NPU 定论 NPU 算子是否根因
- **GPU 服务器可达性验证**: ssh 端口 32222/22/22222/2222 全 Connection timed out——GPU 服务器全端口不可达。ping socket: Operation not permitted（受限）
- **可能根因**: 1.服务器未启动/已关机；2.防火墙/安全组阻断 ssh 端口（本 NPU 服务器出站受限）；3.IP/端口信息有误
- **方向 D/G 阻断定论**: GPU 服务器全端口不可达，方向 D/G（对比官方便样 ckpt GPU 推理输出方向 vs NPU 定论 NPU 算子是否根因）**阻断**
- **状态**: ✅ 方向 D/G GPU 环境验证定论全端口不可达阻断

---

### P4.27 GPU HTTP 代理透 SSH 方案阻断：代理白名单拒所有外网 IP（2026-07-25 定论）

- **触发**: #9 阻断后用户拍板"在 GPU 服务器开 HTTP 代理"——GPU 服务器已跑 `scripts/gpu_http_proxy.sh` 启动 HTTP 代理透 SSH（PID=4182，0.0.0.0:80→SSH 127.0.0.1:32222，RTX4090 24GB×2 验证可用），本机测连通性
- **本机测 GPU 代理透 SSH 连通性**:
  - curl --proxytunnel <GPU_IP>:80 透 SSH 32222：❌ `Connection timed out`（curl 直连 80 端口不可达）
  - 代理白名单是否允许任意 IP 的 80 端口：❌ **拒所有外网 IP**——<GPU_IP>:80 / www.baidu.com:80 / 223.5.5.5:80 / 114.114.114.114:80 全 Establish HTTP proxy tunnel 后无 200 响应（curl --proxytunnel 多 IP 测试）
- **阻断根因定论**: **ModelArts 代理白名单拒所有外网 IP 的 80 端口**——不是 SSH 端口问题，是**本机出站代理白名单拒所有外网 IP**（<GPU_IP>:80 同 www.baidu.com:80 等公网 IP 都不可达）。代理白名单只允许特定域名（华为云内网 myhuaweicloud.com 等），**任意外网 IP 的 80 端口都无法透出**。GPU HTTP 代理透 SSH 方案被代理白名单拒——<GPU_IP>:80 同样不可达
- **方向 D/G 彻底阻断**: 本 NPU 服务器（华为云 ModelArts notebook）出站网络限制拒所有外网 IP，**SSH 直连 + HTTP 代理透 SSH 两路径均阻断**。方向 D/G（对比官方便样 ckpt GPU 推理输出方向 vs NPU 定论 NPU 算子是否根因）**彻底阻断**
- **根因诊断线彻底收敛（9 阶段 + 2 GPU 阻断）**: 已推翻的根因方向（8 个）：ckpt 来源错配（#3 铑证同源）/ 源码版本不匹配（F6 修复后仍 0%）/ abs vs rel 流程不一致 / sign convention / bf16 精度 / chunk 累加飞出 / cann-recipes 官方便样脚本对比（合成图+不同 API）/ select_action vs predict_action_chunk API 差异（源码精读必然一致）。**真根因仍是模型推理本身输出错方向**（#18 定论未被推翻）——官方便样 ckpt（与我们 ckpt 完全同源）在 F6 修复后推理仍返固定负 z 方向 delta_pos。**剩余根因方向 D/G 彻底阻断**——本 NPU 服务器出站网络受限无法 ssh 连接 GPU 服务器对比 GPU 推理输出方向（SSH 直连 + HTTP 代理透 SSH 两路径均被代理白名单拒）
- **状态**: ✅ GPU HTTP 代理透 SSH 方案阻断定论，方向 D/G 彻底阻断

31. **"GPU HTTP 代理透 SSH"方案被 ModelArts 代理白名单拒所有外网 IP 阻断**（2026-07-25 P4.27 印证）：#9 SSH 直连全端口不可达后用户拍板"在 GPU 服务器开 HTTP 代理"绕过，GPU 服务器已启动 HTTP 代理透 SSH（0.0.0.0:80→SSH 127.0.0.1:32222），但本机测连通性发现 <GPU_IP>:80 同样 Connection timed out——精查代理白名单拒所有外网 IP（<GPU_IP>:80 / www.baidu.com:80 / 223.5.5.5:80 / 114.114.114.114:80 全 Establish HTTP proxy tunnel 后无 200 响应），代理白名单只允许特定域名（华为云内网 myhuaweicloud.com 等）非任意外网 IP。教训：**华为云 ModelArts notebook 出站代理白名单拒所有外网 IP**（不是端口问题，是 IP 白名单问题）——SSH 直连 + HTTP 代理透 SSH 两路径均阻断，**判断代理是否可透 SSH 不能凭"代理白名单允许 80/443 端口"推断**，必须测目标 IP 的 80/443 端口是否真在白名单（curl --proxytunnel 多 IP 测试），否则会浪费算力写 GPU 服务器代理脚本后才发现本机出站拒所有外网 IP；**ModelArts notebook 环境根因诊断线剩余方向 D/G 需跳板机/堡垒机或申请出站白名单**，不能在本机反复尝试 SSH/HTTP 代理绕过。

---

### P4.28 GPU 对比脚本 state schema 漂移 + processor 构建缺 ckpt 参数（2026-09-26 项目整理发现并修复）

- **触发**: 全项目整理扫描（PROJECT_TRACKING 阶段 17）发现 GPU 对比脚本与 server 链路有两处不一致，直接威胁 D/G 定论有效性
- **P4.28a（schema 漂移）**: `gpu_infer_compare.py` / `i2_real_obs_infer.py` 仍用旧 schema `[pos3, quat4, grip1]` 构建 state_8，而 server_v2.py 经 P4.7/P4.8 修复后是 `[pos3, axis_angle3, gripper_qpos2]`——GPU 侧若照跑，输入与 NPU server 闭环时不同 schema，对比结论被污染
- **P4.28b（processor 构建缺 ckpt 参数，更深的隐患）**: `make_pre_post_processors(policy.config)` 不传 ckpt 路径时：①tokenizer 用 `processor_pi05.py:145` 硬编码的 `google/paligemma-3b-pt-224`（HF 被代理阻断即 `LocalEntryNotFoundError` 失败）②**不加载 ckpt 的 norm_stats，NormalizerProcessorStep 归一化整条跳过**。这意味着 07-24 的 I2 对比（P4.24）除 schema 漂移外输入还未经归一化——旧 I2 结论的输入构造与 server 实际链路有双重偏差
- **修复**: ①两脚本 state_8 对齐 server（quat wxyz→xyzw→axis_angle 同源链 + gripper q50 中位数 [0.02636, -0.02728]）②`make_pre_post_processors(policy.config, CKPT, preprocessor_overrides={"device_processor": {"device": DEV}})` 与 server_v2.py:362 完全一致 ③gpu_infer_compare.py 重写为 NPU/GPU 双端同脚本（`--device npu:0 / cuda:0`），i2_real_obs_infer.py 标注 SUPERSEDED
- **修复链补全（P4.28c/d/e，v1-v13 迭代）**：
  - P4.28c 两阶段进程隔离：OSMesa GL 上下文与 NPU runtime 同进程两种死法（env 开着推理崩 / NPU 初始化后 env.close() 崩，均 exit 139）→ grab（纯 OSMesa 存 npz）/ infer（纯 NPU）分进程
  - P4.28d 采样 monkey-patch 移植：漏了 server_v2.py:316-334 的 sample_noise/sample_time 强制 fp32 patch → 纯 NPU 推理段 segfault（aclnnNormalFloatFloat 不支持 bf16，本机 CANN 8.5.2 实测直接 segfault 而非报错）
  - P4.28e 任务语言动态取：旧版硬编码 'pick up the black bowl on the stove'（task7 的语言！）当 task0 真值——喂错任务语言输出完全不同方向
- **NPU 基准最终结果（v13 成功，fp32, 卡1, 2026-09-26）**：flipped（闭环图像分布）**`[-0.736, +0.016, -0.891]`**；unflipped `[-0.794, +0.033, -0.854]`；存 `results/pi05_npu_baseline_2026-09-26.json`
- **★ 关键定论：#18 定论被独立路径确证**——flipped 与历史闭环基准 `[-0.88~-0.80, +0.04~-0.01, -0.97~-0.90]` 同方向（固定负 z）。正确构造的单发推理路径（正确 schema + 正确 processor + 正确任务语言 + 闭环图像分布）下，官方便样 ckpt 对 task0 就输出固定负 z——**闭环 0% 不是链路 artifact，是模型真实输出**。反例：错任务语言（task7）输出 `[-0.33, +0.08, -0.29]`（方向不同量级减半），证明任务语言是输入构造关键一环
- **状态**: ✅ 修复完成 + NPU 基准已生成；GPU 侧跑同一脚本（--phase auto --device cuda:0）定论 D/G；GPU 交接包需用修复版脚本重新打包（旧包作废）

34. **OSMesa GL 上下文与 NPU runtime 不能同进程共存**（2026-09-26 P4.28c 印证）：单进程里先建 env 再推理会崩，先初始化 NPU 再关 env 也崩（两种顺序都 exit 139 segfault）。教训：**任何"抓 obs + NPU 推理"的单体脚本必须拆两进程**（grab 纯 OSMesa 存 npz / infer 纯 NPU 读 npz），这正是本项目 client/server 分进程架构的底层原因；GPU（CUDA）无此冲突可单进程。

35. **对比脚本的任务语言必须从 benchmark 动态取**（2026-09-26 P4.28e 印证）：旧版硬编码 'pick up the black bowl on the stove'（实为 task7 语言）当 task0 真值，模型输出 `[-0.33, +0.08, -0.29]`（方向不同量级减半）vs 正确 task0 语言输出 `[-0.736, +0.016, -0.891]`（固定负 z）。教训：**VLA 模型输出对任务语言高度敏感，任何单测/对比脚本的任务文本必须从 benchmark/env 动态取并打印核对**，硬编码任务文本会让"根因定论"建立在错误输入上。

32. **诊断脚本手抄 server 的 obs 构建逻辑必然漂移**（2026-09-26 P4.28 印证）：i2/gpu 对比脚本手抄 server 的 state schema 与 processor 构建，server 经 P4.7/P4.8 改 schema 后诊断脚本不知道，且 `make_pre_post_processors` 不传 ckpt 连归一化都跳过。教训：**诊断/对比脚本必须与 server 用同一套 obs 构建与 processor 构建代码**（或至少在 server 改 schema 时同步改诊断脚本并做 diff 验证），否则"对比结论"对比的是两条不同的链路；**调 lerobot 的 make_pre_post_processors 必须传 ckpt 路径**（不传则用硬编码 tokenizer + 无 norm_stats 的默认 processor，与真实推理链路不一致）。

33. **后台 NPU 任务必须先 source ~/.bashrc**（2026-09-26 基准重生成印证）：非交互 shell（nohup/setsid/cron）不加载 ~/.bashrc 时缺 CANN 驱动库路径（`/usr/local/Ascend/driver/lib64` 等），torch_npu 报 `halGetDeviceInfo failed: drvRet=4` + `torch.npu.is_available()=False`，而交互终端正常——极易误判为"环境坏了"。教训：**任何后台跑的 NPU 任务，启动脚本第一行 `source ~/.bashrc`**（ModelArts 的 bashrc 含 Ascend set_env.sh + update_ld_library_path.sh），再用 nohup setsid 执行；交互正常 + 后台失败 = 先查环境变量继承，不要重装环境。

### P4.29 闭环 0% 真根因定论：输入构造五重缺陷（主因图像 uint8 未 /255），修复后 task0 100%（2026-09-26）

- **触发**: 用户拍板"所有验证必须在 NPU 上进行"，PI0.5 验证修复继续。放弃 GPU 对比（D/G）依赖，改用**训练分布对照法**做根因精查：把 server 构造的每个输入张量直接对照 ckpt norm_stats（训练数据 min/max/q01/q99），逐维验证"模型看到的"是否等于"训练时见过的"。
- **方法论转变（本条核心）**: 此前 11 阶段诊断全部在**链路内部**对比（schema 转换链、API 等价性、NPU/GPU 一致性），从未把输入数值直接对照**训练分布**。norm_stats（`policy_preprocessor_step_2_normalizer_processor.safetensors`）是训练数据的完整统计快照，逐维 in/out-of-distribution 检查是最直接的"输入正确性"判据。
- **五重缺陷定论（全部实证）**:
  - **F（主因）图像 dtype**: server 传 uint8 → 模型内 `_preprocess_images`（modeling_pi05.py:1174）`to(torch.float32)` 得 0-255 → `resize_with_pad_torch` float32 分支 **`clamp(-1.0, 1.0)`**（256→224 必触发 resize）→ 图像摧毁成近二值（任何 ≥1/255 的像素饱和为白）→ **模型全程看到的是黑白垃圾图**。官方管线在 observation_processor.py:90 做 `float32/255.0` → [0,1]，模型内再 `*2-1` → [-1,1]（SigLIP 期望）。修复：server 侧 `permute(2,0,1).float()/255.0`。
  - **A（state 姿态）**: P4.8 把 `robot0_eef_quat` 当 **wxyz** 解释（实为 robosuite **xyzw** 约定）→ 初始位姿 aa=[-0.05,-0.05,-0.01]（夹爪朝前）；训练分布 aa_x∈[0.35,3.67] q50=2.97≈π（夹爪朝下）→ **~10σ OOD**。按 xyzw 解释 aa=[3.14,0,-0.09] ✓ 落训练分布。修复：state_8 的 aa 直接取 robot0_eef_quat 按 xyzw 转（与 lerobot env_processor._quat2axisangle 逐字同源），该字段改必填 fail-fast。
  - **A2（proprio 冻结）**: client `self.proprio` 仅首查初始化，其后靠 `action[-1,:9]` 回传更新；P4.5v2 的"回传原值"只防了污染没解冻结 → **每次查询 server 收到的都是首帧位置**，模型以为机器人从未移动。修复：eval 侧每步 `client.proprio=None`（_format_query 自动从 obs 真值重建，client 文件零改动）。
  - **B（wrist 图方向）**: 训练对全部相机图 H+W 双翻转（env_processor.py:59 `torch.flip(dims=[2,3])`），client 只翻 agentview（_flip_agentview），wrist 发 raw → server 补翻（数值验证 = rot180）。
  - **C（grip 语义反转）**: 旧 `1-grip` 复制自 OpenVLA（其输出约定相反才需反转）；π0.5 训练 grip 即 LIBERO env 语义（q01=-1 开 / q90=+0.92 合，env 实证 -1=开/+1=合）→ 直传才对。旧链路模型想开→机器合、想合→机器开，永不能抓取。
- **敏感性测试定链路通**: state/agentview/wrist/语言四个输入扰动均改变输出 → 推理链路本身无断裂；修 F 前正确任务语言下恒输出俯冲+闭合（垃圾图上的退化行为），修 F 后首步动作 `[0.02,+0.09,+0.09]` + **grip=-1.04（接近阶段正确开爪）**。
- **闭环验证**: task0 ep0/ep1 均 SUCCESS（77/82 步，fp32 NPU1；该局部快测耗时不代表全量平均耗时）——**三个月闭环 0% 就此终结**。完整 spatial（10 task × 10 ep）已启动，结果入 `results/pi05/spatial_100ep_2026-09-26/`。
- **★ 对前期定论的修正**:
  - P4.16/#18"模型推理本身输出错方向是闭环真根因"**部分成立但归因错误**——模型确实输出错方向，但根因是**输入被摧毁**（图像二值化+state OOD+proprio 冻结），不是模型/NPU 损坏。
  - P4.28"正确构造的单发推理也固定负 z → 模型真实输出"的定论**被推翻**——其"正确构造"仍含 uint8 图像摧毁（脚本与 server 同源同错）。若 GPU 对比（D/G）照此执行，两端会一致输出俯冲 → 误判"NPU 无罪、模型坏了"。**GPU 对比基线（pi05_npu_baseline_2026-09-26.json）作废重生成**。
  - P4.5 注释"delta_pos=±0.9 落训练真值范围"判断正确（训练动作确为 robosuite 归一化空间 action.max=0.9375，非米），但被它锚定后停止了对图像链路的怀疑，教训深刻。
- **状态**: ✅ **已记录 spatial 原始统计：96.0%（96/100，seed42，horizon 220，fp32 NPU1）**。逐 task：task0-4/6/7/8 = 100%、task5 = 70%、task9 = 90%。结果：`results/pi05/spatial_100ep_2026-09-26/spatial_results.json`；总耗时 10359.117s（172.65min，2.8775h，约103.59s/ep）。外部参考值待核实，不据此宣称统计等价。

36. **输入正确性的最终判据是训练分布对照，不是链路内部一致性**（2026-09-26 P4.29 印证）：此前 11 阶段全在链路内部找错（转换链等价、API 等价、双端一致），"链路自洽"≠"输入正确"。教训：**新模型接入先用 ckpt norm_stats 对 server 每个输入张量做逐维 in/out-distribution 检查**（1 小时内可完成），能把根因域从"模型/算子/链路"直接收敛到"输入构造"；图像输入要额外查 dtype（uint8/[0,1]）——视觉 encoder 前处理的数值约定（/255、resize clamp）是静默摧毁点，不报错不越界只毁图。

37. **"模型坏了"是最危险的定论，它阻断继续查输入**（2026-09-26 P4.29 印证）：P4.16 的"模型推理本身输出错方向"把诊断引向 GPU 对比与换 ckpt（方向 J/H），若 GPU 可达会"确证"错误定论（同一脚本同错）。教训：**在单端复现的异常行为上，"模型损坏"只能是排除性结论**——必须先穷尽"输入构造与训练管线逐字节一致"的证明（分布对照+敏感性测试），再谈模型/算子问题；双端对比只能证明"两端一致"，不能证明"输入正确"。

38. **PI0.5 四 suite 证据分层与单 checkpoint 覆盖性（2026-09-27）**：π0.5 原论文 [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) 未直接报告 LIBERO 四 suite 分数；openpi 后续 `pi05_libero` 参考为 98.8/98.2/98.0/92.4，LeRobot 后续复现为 97.0/99.0/98.0/96.0，均需与论文分开标注。当前 `/home/ma-user/work/lerobot_pi05_libero_official` 是单一可读 checkpoint，未发现按 suite 独立权重证据；新增四 suite 编排器固定使用同一 checkpoint、NPU0/fp32、每 episode 视频核验。严格 LIBERO 常见 50 trials/task、多 seed 与本地 10 ep/task、seed42 缩减复现不能混称。
