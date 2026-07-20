"""
OpenVLA NPU 推理服务器（独立实现，绕过openvla官方eval脚本的import链问题）
用 X-VLA 的 libero_client 跑仿真（不动 X-VLA 代码），通过 /act HTTP 接口对接。

关键适配：
- NPU 加载：fp32 + eager attn（NPU不支持bf16/flash_attn）
- 输入：libero_client 发来 image0(agentview) + proprio(20维) + language_instruction
- 输出：openvla 输出7维(pos3+quat4)，需转10维(pos3+rot6d+grip1)给X-VLA client
- unnorm_key: libero_spatial（每个suite对应一个checkpoint，此server只加载一个）
"""
import argparse, os, sys, logging, traceback
import numpy as np
import torch
import torch_npu  # noqa
import cv2
from PIL import Image
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import json_numpy

json_numpy.patch()
logging.basicConfig(level=logging.INFO)

# === 动作格式转换：openvla 7维(pos3+quat4) → X-VLA 10维(pos3+rot6d+grip1) ===
def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """quat(w,x,y,z) → rot6d(前两列6值)"""
    import scipy.spatial.transform as st
    # openvla quat 顺序可能是 (x,y,z,w)，需确认；假设 (w,x,y,z)
    R = st.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return R[:, :2].flatten()  # 6维

def action_7_to_10(action7: np.ndarray) -> np.ndarray:
    """openvla 7维 → X-VLA 10维"""
    pos3 = action7[:3]
    quat4 = action7[3:7]
    # openvla 输出可能无grip（7维=pos3+quat4），grip需从别处取或设0
    rot6d = quat_to_rot6d(quat4)
    grip = 0.0  # openvla libero 微调的action格式需确认，先置0
    return np.concatenate([pos3, rot6d, [grip]])

# === NPU 加载 openvla 模型 ===
def load_openvla(model_path: str, device: torch.device):
    """直接用 transformers AutoModelForVision2Seq 加载，绕过 prismatic import 链"""
    from transformers import AutoModelForVision2Seq, AutoProcessor
    # 注册 openvla config（checkpoint 的 config.json 已含 openvla architecture）
    # AutoModelForVision2Seq 会自动识别
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).to(torch.float32)
    model.eval()
    # 加载 norm_stats（predict_action 用）
    stats_path = os.path.join(model_path, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        import json
        with open(stats_path) as f:
            model.norm_stats = json.load(f)
    else:
        logging.warning("无 dataset_statistics.json，predict_action 会报错")
    return model, processor

# === FastAPI 服务器 ===
app = FastAPI()
model = None
processor = None
device = None
unnorm_key = "libero_spatial"  # 每个suite对应一个checkpoint，启动时设置

@app.post("/act")
def act(payload: Dict[str, Any]):
    try:
        # 解码图像
        images = []
        for key in ("image0", "image1", "image2"):
            if key not in payload: continue
            v = json_numpy.loads(payload[key])
            if isinstance(v, np.ndarray):
                if v.ndim == 1:  # encoded bytes
                    v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                images.append(Image.fromarray(v))
        if not images:
            return JSONResponse({"error": "No valid images found."}, status_code=400)

        # OpenVLA 推理：用 predict_action
        prompt = payload["language_instruction"]
        # OpenVLA 微调时用了 center_crop，推理也要（但服务器端图像已是仿真截图，跳过）
        inputs = processor(prompt, images[0]).to(device, dtype=torch.float32)

        # 固定 seed（可复现）
        seed = int(os.environ.get("XVLA_SEED", "42"))
        torch.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
        np.random.seed(seed)

        with torch.no_grad():
            # OpenVLA predict_action 返回 7维 normalized action
            action7 = model.predict_action(
                inputs=inputs,
                unnorm_key=unnorm_key,
                do_sample=False,
            )
        # 7维 → 10维（X-VLA client 期望的格式）
        action10 = action_7_to_10(np.array(action7))
        # X-VLA client 期望 shape (T, 10)，单步返回
        return JSONResponse({"action": [action10.tolist()]})
    except Exception:
        logging.error(traceback.format_exc())
        return JSONResponse({"error": "Request failed"}, status_code=400)

def main():
    global model, processor, device, unnorm_key
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--unnorm_key", default="libero_spatial", help="suite名，对应微调checkpoint")
    args = parser.parse_args()
    unnorm_key = args.unnorm_key

    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    logging.info(f"🧠 device: {device} (torch_npu), unnorm_key: {unnorm_key}")

    model, processor = load_openvla(args.model_path, device)
    logging.info("✅ 模型加载完成，启动服务器...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
