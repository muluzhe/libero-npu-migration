"""
PI0.5 NPU 推理 server（适配 LIBERO 接口，对接 X-VLA client）
- 不走 lerobot 官方 eval 脚本（lerobot-eval 默认 GPU，未适配 NPU + OSMesa）
- 用 lerobot 的 PI05Policy.predict_action_chunk + make_pre_post_processors 做推理
- NPU 加载：fp32 + npu:0（F4：bf16 有精度风险，验证用 fp32）
- 动作格式适配：pi0.5 输出 [T, 7] delta action（[delta_pos3, delta_aa3, delta_grip1]）
  → X-VLA client 期望 [T, 10] 即 [pos3, rot6d, grip1]
  → 用同源 AxisAngle_to_Rotate6D 把 aa3 转 rot6d，匹配 client 的 Rotate6D_to_AxisAngle 反向转换
- act_type=rel：pi0.5 输出 delta action，走 OpenVLA 验证过的 rel 路径（use_delta=True）

=== 2026-09-26 P4.29 输入构造三重修复（闭环 0% 真根因） ===
前置事实（norm_stats 实证）：
  1. 训练动作空间即 robosuite 归一化空间（action.max ±0.9375，±1≈±0.05m），非米
  2. 训练 state aa_x ∈ [0.35, 3.67]、q50≈π（夹爪朝下）
  3. 训练图像（含 wrist）全部 H+W 双翻转（lerobot env_processor: torch.flip(dims=[2,3])）
  4. LIBERO env 夹爪命令：-1=开、+1=合（settle 实证：10 步 -1 后 qpos 仍全开）
修复：
  A. state_8 的 aa 直接取 robot0_eef_quat 按 (x,y,z,w) 解释（与 lerobot
     env_processor._quat2axisangle 逐字同源）。旧 P4.8 链路把 quat 当 wxyz 解释 +
     ori6d perm/signs 转换 → aa≈0（夹爪朝前），相对训练分布 ~10σ OOD —— 这才是
     "每 chunk 首步固定负 z" 的真根因（模型被喂了不可能存在的状态）。
  B. wrist 视图 H+W 双翻转（client 只翻 agentview；训练对双视图都翻）。
  C. grip 去掉 1-grip 反转直传：训练 grip 即 env 语义（-1=开/+0.92=合），
     client >0.5→+1(合)、否则 -1(开)。旧 1-grip 使模型开→机器合、合→机器开。
  D. chunk 恢复官方协议 n_action_steps=10（F1=2 是错误输入下的止血，不再需要），
     并删除 P4.5v2 末步 proprio 回传（eval 侧已每步发 fresh proprio，回传 hack
     反而浪费一半动作步）。
  E. robot0_eef_quat 必填（fail-fast）；gripper_qpos 优先用 client 回传真值。
  F. 【终极根因·闭环 0% 主因】图像 float/255 → [0,1]（官方 observation_processor.py:90
     同源）。旧链路传 uint8 → 模型内 to(float32)=0-255 → resize_with_pad_torch float32
     分支 clamp(-1,1) → 图像摧毁成近二值 → 模型全程看到垃圾图。此前 11 阶段诊断的
     "模型推理输出错方向"实为图像输入摧毁所致；A-E 修复虽正确但不足，F 才是主因。

关键文件依赖：
  - lerobot_pi05/src/lerobot/policies/pi05/modeling_pi05.py（已应用 modeling_pi05.patch）
  - X-VLA/evaluation/libero/libero_client.py（官方便样，零改动）
  - scripts/pi05/eval_pi05_spatial.py（配套：每步重置 client.proprio + 注入 env 真值 quat/qpos）
"""
import argparse
import os
import sys
import types
import importlib.machinery
import logging
import numpy as np
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

# === 路径设置：lerobot pi0.5 框架 + X-VLA client（同源转换链）===
LEROBOT_ROOT = os.environ.get("LEROBOT_PI05_ROOT", os.path.expanduser("~/work/lerobot_pi05"))
sys.path.insert(0, os.path.join(LEROBOT_ROOT, "src"))
X_VLA_ROOT = os.environ.get("X_VLA_ROOT", os.path.expanduser("~/work/X-VLA"))
sys.path.insert(0, os.path.join(X_VLA_ROOT, "evaluation/libero"))
from libero_client import LiberoAbsActionProcessor  # noqa: E402
_proc_rot = LiberoAbsActionProcessor()


def aa_to_rot6d(aa):
    """axis-angle(3) → rot6d，用 client 同源 AxisAngle_to_Rotate6D 确保两端等价。
    与 OpenVLA server_v2.py 同源转换链（cann-recipes 验证：两端差=[0,0,0]）。
    """
    return _proc_rot.AxisAngle_to_Rotate6D(np.asarray(aa, dtype=np.float32))


# === torch_npu（NPU 推理）===
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

# === lerobot pi0.5 框架（延迟 import 到 main 后，避免 FastAPI 启动时加载）===
PI05Policy = None
make_pre_post_processors = None

# === 固定 seed（cann-recipes 验证：NPU 推理确定化）===
_SEED = int(os.environ.get("PI05_SEED", "42"))
torch.manual_seed(_SEED)
if _HAS_NPU and torch.npu.is_available():
    torch.npu.manual_seed_all(_SEED)
np.random.seed(_SEED)

# json_numpy patch 只需一次（与 OpenVLA server_v2 一致的性能优化）
try:
    import json_numpy
    json_numpy.patch()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pi05_server")

app = FastAPI()
policy = None
preprocess = None
postprocess = None
device = None
infer_dtype = None
# pi0.5 LIBERO 配置：action_dim=7, chunk_size=50, n_action_steps=10
# 输入 schema（lerobot/pi05-libero config.json）：
#   observation.images.image  (3,256,256)  ← client 的 agentview（已 _flip_agentview）
#   observation.images.image2 (3,256,256)  ← client 的 wrist view
#   observation.state (8,)                 ← client 的 proprio 前 8 维（pos3+quat4+grip1，去除 legacy past copy）
# pi0.5 输出：[T, 7] = [delta_pos3, delta_aa3, delta_grip1]（LIBERO 训练用 relative EEF action）
_ACTION_DIM = 7
_GRIP_DIM = 1


class ActRequest(BaseModel):
    image0: str = ""        # agentview (256x256x3 uint8，client 已 _flip_agentview)
    image1: str = ""        # wrist view（client 发 raw，server 端补 H+W 翻转对齐训练）
    language_instruction: str = ""
    proprio: str = ""       # 20维 [pos3+ori6d+grip1 + past copy]（X-VLA client 闭合环 proprio 格式）
    robot0_eef_quat: str = ""  # P4.29 必填：env 真值 eef quat（robosuite (x,y,z,w) 约定，
                              # 与训练 robot_state.eef.quat 同源 → aa 落训练分布 [0.35,3.67]≈π）
    robot0_gripper_qpos: str = ""  # P4.29 可选：env 真值夹爪 qpos(2)（不发用训练 q50 默认）
    domain_id: int = 0
    steps: int = 10         # client 期望的 action chunk 长度


@app.post("/act")
def act(req: ActRequest):
    try:
        import cv2
        import json_numpy
        # === 解码图像 ===
        img_main = json_numpy.loads(req.image0) if req.image0 else None
        if img_main is None:
            return {"error": "no image"}
        if img_main.ndim == 1:
            img_main = cv2.imdecode(img_main, cv2.IMREAD_COLOR)
        img_wrist = json_numpy.loads(req.image1) if req.image1 else None
        if img_wrist is None:
            img_wrist = img_main.copy()  # pi0.5 要双视图，缺 wrist 用 main 填充
        elif img_wrist.ndim == 1:
            img_wrist = cv2.imdecode(img_wrist, cv2.IMREAD_COLOR)

        # === proprio 解码：client 发 20维 [pos3+ori6d(6)+grip1(填0) + past copy(10)] ===
        # P4.29：eval 侧每步重置 client.proprio → 每次查询携带 fresh closed_loop_proprio
        # （旧链路 client.proprio 冻结在首帧 env 状态，server 以为机器人从未移动）
        proprio_raw = json_numpy.loads(req.proprio) if req.proprio else None
        if proprio_raw is None:
            return {"error": "no proprio"}
        proprio_raw = np.array(proprio_raw, dtype=np.float32).flatten()
        pos3 = proprio_raw[:3]

        # === P4.29 修复 A：aa 取 env 真值 robot0_eef_quat，按 (x,y,z,w) 解释 ===
        # 与训练链路（lerobot env_processor：raw_obs["robot0_eef_quat"] → _quat2axisangle）
        # 逐字同源。旧 P4.8 把它当 wxyz 解释 + ori6d perm/signs 转换 → aa≈0，
        # 相对训练分布（aa_x∈[0.35,3.67]≈π，夹爪朝下）~10σ OOD。
        # 实证（2026-09-26 诊断脚本）：初始位姿 quat=[0.9996,-0.0009,-0.0279,-0.00026]，
        #   按 xyzw → aa=[3.141,0,-0.088] ✓ 落训练分布；按 wxyz → aa≈0 ✗ 分布外。
        if not req.robot0_eef_quat:
            return {"error": "robot0_eef_quat required (P4.29: 训练同源 state 构造必填，"
                             "配套 scripts/pi05/eval_pi05_spatial.py 会注入)"}
        quat_xyzw = np.array(json_numpy.loads(req.robot0_eef_quat), dtype=np.float32).flatten()[:4]
        # axis_angle = 2*acos(w)*(xyz/||xyz||)；与 lerobot env_processor._quat2axisangle 同源
        w_q = float(np.clip(quat_xyzw[3], -1.0, 1.0))
        den = np.sqrt(max(1.0 - w_q * w_q, 0.0))
        if den < 1e-10:
            axis_angle = np.zeros(3, dtype=np.float32)  # 单位 quat → aa=0
        else:
            angle = 2.0 * np.arccos(w_q)
            axis_angle = (quat_xyzw[:3] / den * angle).astype(np.float32)
        # === P4.29 修复 B：wrist 视图 H+W 双翻转 ===
        # 训练对全部相机图 torch.flip(dims=[2,3])（env_processor），client 只翻 agentview，
        # wrist 发的是 raw → server 端补翻，使双视图与训练分布一致
        img_wrist = np.ascontiguousarray(np.flip(np.flip(img_wrist, 0), 1))
        # === P4.29 修复 E：夹爪 qpos 优先用 env 真值（训练 dims 6-7 范围 ~±0.04）===
        if req.robot0_gripper_qpos:
            gripper_qpos2 = np.array(json_numpy.loads(req.robot0_gripper_qpos), dtype=np.float32).flatten()[:2]
        else:
            gripper_qpos2 = np.array([0.02636, -0.02728], dtype=np.float32)  # 训练 q50 中位数默认
        state_8 = np.concatenate([pos3, axis_angle, gripper_qpos2])  # 8维，与训练 schema 逐维同源
        logger.info("P4.29 state_8=%s | aa_x=%.3f(训练分布[0.35,3.67]) | fresh pos3=%s",
                    state_8.tolist(), axis_angle[0], pos3.tolist())

        # === 构造 lerobot pi0.5 期望的 obs dict ===
        # lerobot libero env 的 _format_raw_obs 用 pixels+robot_state，但 pi0.5 processor
        # 的 schema（config.json）是 observation.images.image/image2 + observation.state
        # P4.29 修复 F（终极根因，闭环 0% 的主因）：图像必须 float [0,1]
        #   官方管线：observation_processor.py:90 → float32/255.0 → [0,1]
        #   模型内部 _preprocess_images（modeling_pi05.py:1189）：[0,1] ×2-1 → [-1,1]（SigLIP 期望）
        #   旧链路传 uint8 → 模型内 to(float32) 得 0-255 → resize_with_pad_torch 的 float32
        #   分支 clamp(-1,1) → 图像被摧毁成近二值（≥1/255 全饱和白）→ 模型看到垃圾图
        obs = {
            "observation.images.image": torch.from_numpy(img_main).permute(2, 0, 1).float() / 255.0,  # (3,H,W) [0,1]
            "observation.images.image2": torch.from_numpy(img_wrist).permute(2, 0, 1).float() / 255.0,
            "observation.state": torch.from_numpy(state_8).float(),
            "task": req.language_instruction,
        }
        # 加 batch 维
        for k in ("observation.images.image", "observation.images.image2", "observation.state"):
            obs[k] = obs[k].unsqueeze(0)

        # === preprocess → predict_action_chunk → postprocess ===
        # 关键：client.step 期望 server 返多步 chunk（action[:, :9] 切片），
        # 不能用 select_action（它内部 queue 滚动每次只返 1 步，导致 client 切片错位 → 0%）
        # 改用 predict_action_chunk 一次返 n_action_steps=10 步，client 用 action_plan 缓存滚动
        with torch.inference_mode():
            batch = preprocess(obs)
            # 把 batch 内 float tensor 移到 device + infer_dtype
            batch = _move_to_device_dtype(batch, device, infer_dtype)
            # predict_action_chunk → PolicyAction 包装的 [B, chunk_size=50, action_dim=7]
            pred_chunk = policy.predict_action_chunk(batch)
            # 取前 n_action_steps 步（pi0.5 LIBERO config: n_action_steps=10，官方协议）
            # P4.29 修复 D：F1 的 chunk=2 是错误输入（aa OOD+proprio 冻结）下的止血，
            # 输入修复后恢复官方协议 n_action_steps=10，chunk 内开环执行、边界重新查询
            n_act = int(policy.config.n_action_steps)
            # PolicyAction 支持切片取前 n_act 步；postprocess 要 PolicyAction 类型不能传 dict
            pred_action = pred_chunk[:, :n_act]
            # postprocess 反归一化（用 norm_stats）
            pred_action = postprocess(pred_action)

        # === pred_action shape: [B, n_action_steps, 7] 或 [n_action_steps, 7]
        # bf16 tensor 不能直接 np.array(...,dtype=fp32)（Got unsupported ScalarType BFloat16），
        # 先 .float() 转 fp32 再 .cpu().numpy()
        if isinstance(pred_action, torch.Tensor):
            pred_action = pred_action.float().cpu().numpy()
        elif isinstance(pred_action, dict) and "action" in pred_action:
            t = pred_action["action"]
            pred_action = (t.float() if isinstance(t, torch.Tensor) else torch.as_tensor(t)).cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)
        a = np.array(pred_action, dtype=np.float32)
        if a.ndim == 3:
            a = a[0]  # 去 batch
        # a shape: [T, 7] = [delta_pos3, delta_aa3, delta_grip1]
        # 截断到 client 期望的 steps 长度（P4.29 修复 D：去掉 F1 硬覆盖，min 与官方 n_action_steps 对齐）
        T_chunk = min(n_act, req.steps) if req.steps else n_act
        if a.shape[0] < T_chunk:
            # 不够长，复制末步填充（不应发生，chunk_size=50 >> steps=10）
            a = np.tile(a[-1:], (T_chunk - a.shape[0] + 1, 1))[:T_chunk] if a.shape[0] > 0 else np.zeros((T_chunk, 7), dtype=np.float32)
        else:
            a = a[:T_chunk]

        # === 转成 X-VLA client 期望的 [T, 10] = [pos3, rot6d, grip1] ===
        delta_pos = a[:, :3]
        delta_aa = a[:, 3:6]    # axis-angle 3维
        delta_grip = a[:, 6:7]  # 夹爪
        # aa3 → rot6d（同源转换链，与 OpenVLA server_v2 一致）
        rot6d = np.stack([aa_to_rot6d(aa) for aa in delta_aa], axis=0)  # [T, 6]

        # === P4.29 修复 C：grip 直传（去掉旧 1-grip 反转） ===
        # 训练 grip 即 LIBERO env 语义（-1=开 / +0.92=合，norm_stats q01=-1、q90=0.92），
        # client 离散化 >0.5→+1（env 合）、否则 -1（env 开）——直传即语义对齐。
        # 旧 1-grip 复制自 OpenVLA（其输出约定相反才需反转），用在 pi0.5 上是
        # 模型开→机器合、模型合→机器开，永不能抓取。
        grip_out = delta_grip

        action_10d = np.concatenate([delta_pos, rot6d, grip_out], axis=-1)  # [T, 10]
        logger.info("return chunk shape=%s | 首步 delta_pos=%s | grip=%s",
                    str(action_10d.shape), delta_pos[0].tolist(), delta_grip.flatten().tolist())
        return {"action": action_10d.tolist()}

    except Exception as e:
        import traceback
        logger.error("act failed: %s", e)
        return {"error": str(e), "trace": traceback.format_exc()[-800:]}


def _move_to_device_dtype(data, dev, dtype):
    """递归把 batch 内的 float tensor 移到 device + dtype，long/int 保持。"""
    if isinstance(data, torch.Tensor):
        if data.is_floating_point():
            return data.to(dev, dtype=dtype)
        return data.to(dev)
    if isinstance(data, dict):
        return {k: _move_to_device_dtype(v, dev, dtype) for k, v in data.items()}
    if isinstance(data, list):
        return [_move_to_device_dtype(v, dev, dtype) for v in data]
    return data


def main():
    global policy, preprocess, postprocess, device, infer_dtype
    global PI05Policy, make_pre_post_processors

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="pi0.5 LIBERO checkpoint 路径（可通过 PI05_CKPT 配置）")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="bf16 加载到 NPU（pi0.5 config.json 默认 dtype=bfloat16，但 F4 修复改 default=False 用 float32 解 bf16 精度根因）")
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    args = parser.parse_args()

    # === 延迟 import lerobot pi0.5 框架（main 时才加载，避免 FastAPI 启动慢）===
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy as _PI05Policy
    from lerobot.policies.pi05 import modeling_pi05 as _mpi05
    from lerobot.policies.factory import make_pre_post_processors as _make_pp
    PI05Policy = _PI05Policy
    make_pre_post_processors = _make_pp

    # === NPU 适配：Ascend 910B4 的 aclnnNormalFloatFloat 不支持 BFLOAT16 输出 ===
    # pi0.5 的 sample_noise/sample_time 用 _inference_dtype()（bf16）调 torch.normal → 报 EZ1001
    # monkey-patch：强制 fp32 采样噪声/时间，再 .to(bf16) 给后续算子（数值精度无损）
    _OrigPI05Pytorch = _mpi05.PI05Pytorch
    _fp32_noise = _mpi05.PI05Pytorch.sample_noise
    _fp32_time = _mpi05.PI05Pytorch.sample_time

    def _patched_sample_noise(self, shape, device):
        with torch.no_grad():
            return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device).to(self._inference_dtype())

    def _patched_sample_time(self, bsize, device):
        with torch.no_grad():
            t = _fp32_time(self, bsize, device)
            return t.to(dtype=self._inference_dtype(), device=device)

    _mpi05.PI05Pytorch.sample_noise = _patched_sample_noise
    _mpi05.PI05Pytorch.sample_time = _patched_sample_time
    logger.info("NPU monkey-patch applied: sample_noise/sample_time forced fp32 (aclnnNormalFloatFloat 不支持 BFLOAT16 输出)")

    # === 设备选择 ===
    if args.device == "auto":
        if _HAS_NPU and torch.npu.is_available():
            device = torch.device("npu:0")
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info("device: %s", device)

    # === dtype：pi0.5 config 默认 bfloat16，NPU 原生支持 ===
    infer_dtype = torch.bfloat16 if args.bf16 else torch.float32
    logger.info("infer dtype: %s", infer_dtype)

    # === 加载 pi0.5 policy + processor ===
    logger.info("loading PI05Policy from: %s", args.model_path)
    policy = PI05Policy.from_pretrained(args.model_path)
    policy = policy.to(device).to(infer_dtype)
    policy.eval()
    logger.info("PI05Policy loaded | chunk_size=%d n_action_steps=%d action_dim=%d",
                policy.config.chunk_size, policy.config.n_action_steps,
                policy.config.output_features.get("action").shape[0] if policy.config.output_features.get("action") else -1)

    # === pre/post processor（含 normalize/unnormalize norm_stats）===
    preprocess, postprocess = make_pre_post_processors(
        policy.config, args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    logger.info("processors ready")

    logger.info("PI0.5 NPU server listening on port %d (bf16=%s)", args.port, args.bf16)
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
