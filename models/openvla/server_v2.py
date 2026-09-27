"""
OpenVLA NPU 推理服务器（路径A：绕过openvla官方eval的import问题，用我们验证过的server框架）
- 不走 openvla 仓库的 experiments/robot/ 代码（避免 prismatic/dlimp/tf import 链 segfault）
- 直接用 transformers AutoModelForVision2Seq 加载 openvla checkpoint，走 PrismaticProcessor
- NPU 加载：fp32 + eager attn + npu:0
- 动作格式适配：OpenVLA 输出 7维(pos3+quat4+grip1) → X-VLA client 期望 10维(pos3+rot6d+grip1)
"""
import argparse, os, sys, json, types
import numpy as np
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel

# 同源转换链：用X-VLA client的AxisAngle_to_Rotate6D确保两端等价
sys.path.insert(0, os.environ.get('X_VLA_ROOT', os.path.expanduser('~/work/X-VLA')) + '/evaluation/libero')
from libero_client import LiberoAbsActionProcessor
_proc_rot = LiberoAbsActionProcessor()


def aa_to_rot6d(aa):
    """axis-angle(3) → rot6d，用client同源AxisAngle_to_Rotate6D确保两端等价。
    之前用Rodrigues公式与client的Rotate6D_to_AxisAngle不等价（差值大→0%成功率根因）。
    """
    return _proc_rot.AxisAngle_to_Rotate6D(np.asarray(aa, dtype=np.float32))

# === 阻断 prismatic/dlimp/tf 加载链（segfault fix）===
import importlib.machinery
for m_name in ('dlimp', 'rlds', 'ox_pilot', 'tensorflow', 'tensorflow.data'):
    if m_name not in sys.modules:
        m = types.ModuleType(m_name)
        m.__spec__ = importlib.machinery.ModuleSpec(m_name, None)
        m.__path__ = []
        sys.modules[m_name] = m

# torch_npu
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

# transformers + prismatic（stub后能通）
from transformers import AutoModelForVision2Seq, AutoProcessor

# 注册 openvla 到 transformers Auto Classes（prismatic会自动注册，但stub后需手动）
try:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor as HFAutoProcessor
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    HFAutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    print("[*] prismatic 注册 OK")
except Exception as e:
    print(f"[!] prismatic 注册失败（尝试 AutoModel 自动加载）: {e}")

app = FastAPI()
model = None
processor = None
device = None
unnorm_key = None  # 各 suite 对应不同 unnorm_key
infer_dtype = torch.float32  # 默认值，main()里会按 --bf16 改


class ActRequest(BaseModel):
    image0: str = ""       # agentview (256x256x3 uint8)
    image1: str = ""       # wrist view
    language_instruction: str = ""
    proprio: str = ""      # 20维 [pos3+ori6d+grip1 + past copy]
    domain_id: int = 0
    steps: int = 10


# === 性能优化：全局预计算，避免每步重复 ===
_CROP_SCALE = 0.9
_CROP_H = int(224 * np.sqrt(_CROP_SCALE))
_CROP_W = int(224 * np.sqrt(_CROP_SCALE))
_CROP_Y0 = (224 - _CROP_H) // 2
_CROP_X0 = (224 - _CROP_W) // 2
_SEED = int(os.environ.get("OPENVLA_SEED", "42"))
# 固定seed只需设一次（do_sample=False 时推理确定）
torch.manual_seed(_SEED)
if _HAS_NPU and torch.npu.is_available():
    torch.npu.manual_seed_all(_SEED)
np.random.seed(_SEED)
# json_numpy.patch 只需一次
try:
    import json_numpy
    json_numpy.patch()
except Exception:
    pass


@app.post("/act")
def act(req: ActRequest):
    import cv2
    try:
        # 解码图像
        import json_numpy
        img = json_numpy.loads(req.image0) if req.image0 else None
        if img is None:
            return {"error": "no image"}
        if img.ndim == 1:
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        # === 官方预处理链：rotate180 → resize224(lanczos3) → center_crop scale=0.9 ===
        # X-VLA client已做了_flip_agentview(=rotate180)，这里不再重复
        # 1. resize到224×224（匹配官官get_libero_image的resize_size=224，lanczos3插值）
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        # 2. center_crop scale=0.9 + resize回224（合并为一次 crop→resize）
        pil_img = Image.fromarray(img).convert("RGB")
        img_np = np.array(pil_img)[_CROP_Y0:_CROP_Y0+_CROP_H, _CROP_X0:_CROP_X0+_CROP_W]
        img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        pil_img = Image.fromarray(img_np).convert("RGB")

        with torch.no_grad():
            # OpenVLA prompt格式（官官：In: What action should the robot take to {lang}?\nOut:）
            lang = req.language_instruction.lower()
            prompt = f"In: What action should the robot take to {lang}?\nOut:"
            # INT8量化模型在CPU上（bitsandbytes是CUDA专用，NPU不支持），inputs也放CPU
            # bf16模型+fp32 inputs在NPU上报错（dtype不匹配），用infer_dtype保持一致
            inputs = processor(prompt, pil_img)
            if os.environ.get("OPENVLA_USE_8BIT") == "1":
                inputs = inputs.to("cpu", dtype=torch.float32)
            else:
                inputs = inputs.to(device, dtype=infer_dtype)
            # OpenVLA predict_action：unnorm_key=task_suite_name, do_sample=False
            action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            # action 是 7维 numpy: [pos3, quat4, grip1]

        # 转成 X-VLA client 期望的 10维 [pos3, rot6d, grip1]
        # OpenVLA返回7维 [pos3, aa3(axis-angle), grip1]，aa3→rot6d给client再转回aa
        a = np.array(action, dtype=np.float32).flatten()  # 7维
        pos = a[:3]
        aa = a[3:6]  # axis-angle 3维（不是quat！）
        grip = a[6:7] if len(a) >= 7 else np.array([a[-1]])  # 夹爪 ∈ [0,1]（原始）
        # aa3 → rot6d
        rot6d = aa_to_rot6d(aa)  # 6维

        # === grip反转：让X-VLA client的>0.5离散化后符号匹配官官invert ===
        # 官方链路: grip_raw→normalize[-1,+1]→invert→env.step（+1=close）
        # 我们链路: server返1-grip_raw → client >0.5离散化 → env.step
        # 例: grip_raw=0.996→server返0.004→client<0.5→-1→env.step(-1=open) ✅匹配官官invert
        grip = 1.0 - grip

        action_10d = np.concatenate([pos, rot6d, grip])  # 10维

        # X-VLA client 期望 [T, 10] 格式（steps维）
        # OpenVLA 只返回当前步，复制 steps 次满足 client 期望
        T = req.steps if req.steps else 10
        action_batch = np.stack([action_10d] * T, axis=0)  # [T, 10]

        return {"action": action_batch.tolist()}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-500:]}


def main():
    global infer_dtype, model, processor, device, unnorm_key
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--unnorm_key", default="libero_spatial",
                        help="各 suite 对应不同 unnorm_key: libero_spatial/libero_object/libero_goal/libero_10")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="INT8量化加载（7B→~7GB，解决HBM不足）")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="4bit量化加载（7B→~4GB）")
    parser.add_argument("--bf16", action="store_true",
                        help="bf16加载到NPU（7B→~14GB，原生NPU推理，推荐）")
    args = parser.parse_args()

    unnorm_key = args.unnorm_key

    # 设备选择
    if args.device == "auto":
        if _HAS_NPU and torch.npu.is_available():
            device = torch.device("npu:0")
        elif torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"🧠 device: {device}" + (" (torch_npu)" if _HAS_NPU else ""))

    # 推理dtype：bf16优先（NPU原生），否则fp32
    infer_dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"[*] 推理dtype: {infer_dtype}")

    # 加载模型
    attn_impl = os.environ.get("OPENVLA_ATTN", "eager")
    print(f"[*] attn实现: {attn_impl}")
    print(f"[*] 加载 OpenVLA checkpoint: {args.model_path}" + (" (bf16)" if args.bf16 else (" (INT8量化)" if args.load_in_8bit else " (fp32)")))
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    load_kwargs = {
        "attn_implementation": attn_impl,
        "torch_dtype": infer_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, **load_kwargs)
    if not (args.load_in_8bit or args.load_in_4bit):
        model = model.to(device).to(infer_dtype)
    # INT8/4bit 模型已自动放到正确设备，不能再 .to()
    model.eval()
    print(f"✅ 模型加载完成，unnorm_key={unnorm_key}")
    print(f"[*] 启动服务器 port={args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
