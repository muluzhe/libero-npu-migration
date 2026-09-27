"""
⚠️ DEPRECATED（2026-09-26）：本文件是未实测骨架，API 假设错误（model.predict_action(**inputs, steps=...)
对 lerobot PI05Policy 不成立），请勿使用。
正确实现：models/pi0/server_v2.py（适配 lerobot 框架，推理+闭环已跑通），
配合 scripts/pi05/run_pi05_spatial.sh 使用，详见 docs/pi05/PI05_TRACKING.md。

PI0 / PI0.5 NPU 推理服务器
flow-matching 采样循环，需 eager attn。
"""
import argparse, torch, uvicorn, numpy as np
try:
    import torch_npu  # noqa: F401
except ImportError:
    pass
from fastapi import FastAPI
from transformers import AutoModel, AutoProcessor

app = FastAPI()
model = None
processor = None
device = None

@app.post("/act")
def act(req: dict):
    import json_numpy; json_numpy.patch()
    img = json_numpy.loads(req.get("image0",""))
    if img is None: return {"error": "no image"}
    from PIL import Image
    if img.ndim == 1: import cv2; img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    pil = Image.fromarray(img)
    inputs = processor(pil, req.get("language_instruction",""), return_tensors="pt")
    inputs = {k: v.to(device, dtype=model.dtype if hasattr(v,'is_floating_point') and v.is_floating_point() else torch.long) for k,v in inputs.items()}
    with torch.no_grad():
        action = model.predict_action(**inputs, steps=req.get("steps",10))
    return {"action": np.array(action).tolist()}

def main():
    global model, processor, device
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--port", type=int, default=8010)
    args = p.parse_args()
    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=True).to(device)
    model.eval()
    print(f"✅ PI0 on {device}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
