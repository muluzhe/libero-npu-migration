"""Diffusion Policy NPU 推理服务器（DDPM 采样，需 eager attn）"""
import argparse, torch, uvicorn, numpy as np
try: import torch_npu  # noqa
except: pass
from fastapi import FastAPI
app = FastAPI(); model=None; device=None

@app.post("/act")
def act(req: dict):
    import json_numpy; json_numpy.patch()
    img = json_numpy.loads(req.get("image0",""))
    if img is None: return {"error":"no image"}
    obs = torch.as_tensor(np.array(img)).float().unsqueeze(0).to(device)/255.0
    with torch.no_grad():
        action = model.predict_action(obs)  # DDPM 采样
    return {"action": action.cpu().numpy().tolist()}

def main():
    global model,device
    p=argparse.ArgumentParser(); p.add_argument("--model_path",required=True); p.add_argument("--port",type=int,default=8010); a=p.parse_args()
    device=torch.device("npu:0" if torch.npu.is_available() else "cpu")
    from diffusion_policy.models.diffusion.unet import ConditionalUnet1D
    model=ConditionalUnet1D.from_pretrained(a.model_path).to(device).float(); model.eval()
    print(f"✅ DiffusionPolicy on {device}"); uvicorn.run(app,host="0.0.0.0",port=a.port)

if __name__=="__main__": main()
