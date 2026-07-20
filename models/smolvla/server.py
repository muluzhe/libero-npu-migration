"""SmolVLA NPU 推理服务器（与 X-VLA 架构类似，transformers AutoModel）"""
import argparse, torch, uvicorn, numpy as np
try: import torch_npu  # noqa
except: pass
from transformers import AutoModelForVision2Seq, AutoProcessor
from fastapi import FastAPI
app = FastAPI(); model=None; processor=None; device=None

@app.post("/act")
def act(req: dict):
    import json_numpy; json_numpy.patch()
    img = json_numpy.loads(req.get("image0",""))
    if img is None: return {"error":"no image"}
    from PIL import Image
    if img.ndim==1: import cv2; img=cv2.imdecode(img,cv2.IMREAD_COLOR)
    inputs = processor(Image.fromarray(img), req.get("language_instruction",""), return_tensors="pt")
    inputs = {k:v.to(device, dtype=model.dtype if hasattr(v,'is_floating_point') and v.is_floating_point() else torch.long) for k,v in inputs.items()}
    with torch.no_grad(): action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return {"action": np.array(action).tolist()}

def main():
    global model,processor,device
    p=argparse.ArgumentParser(); p.add_argument("--model_path",required=True); p.add_argument("--port",type=int,default=8010); a=p.parse_args()
    device=torch.device("npu:0" if torch.npu.is_available() else "cpu")
    processor=AutoProcessor.from_pretrained(a.model_path,trust_remote_code=True)
    model=AutoModelForVision2Seq.from_pretrained(a.model_path,torch_dtype=torch.float32,trust_remote_code=True).to(device); model.eval()
    print(f"✅ SmolVLA on {device}"); uvicorn.run(app,host="0.0.0.0",port=a.port)

if __name__=="__main__": main()
