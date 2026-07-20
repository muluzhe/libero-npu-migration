"""ACT (Action Chunking Transformer) NPU 推理服务器（CVAE+transformer）"""
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
    # ACT: 输入图像+本体感觉，输出 action chunk
    img_t = torch.as_tensor(np.array(img)).float().permute(2,0,1).unsqueeze(0).to(device)/255.0
    proprio = torch.as_tensor(np.array(json_numpy.loads(req.get("proprio","[]")))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action = model(img_t, proprio)  # ACT forward
    return {"action": action.squeeze(0).cpu().numpy().tolist()}

def main():
    global model,device
    p=argparse.ArgumentParser(); p.add_argument("--model_path",required=True); p.add_argument("--port",type=int,default=8010); a=p.parse_args()
    device=torch.device("npu:0" if torch.npu.is_available() else "cpu")
    # ACT 模型加载（用户需提供模型类）
    from models.act.model import ACTPolicy
    model=ACTPolicy.from_pretrained(a.model_path).to(device).float(); model.eval()
    print(f"✅ ACT on {device}"); uvicorn.run(app,host="0.0.0.0",port=a.port)

if __name__=="__main__": main()
