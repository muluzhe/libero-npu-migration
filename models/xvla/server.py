"""
X-VLA NPU 推理服务器（torch_npu 在线推理）
零模型改动，仅 --device auto 切换 NPU。
"""
import argparse, os, sys
import torch
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = True
except ImportError:
    _HAS_NPU = False

sys.path.insert(0, os.environ.get("X_VLA_ROOT", os.path.expanduser("~/work/X-VLA")))
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # 自动设备选择
    if args.device == "auto":
        if _HAS_NPU and torch.npu.is_available():
            device = torch.device("npu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"🧠 device: {device}" + (" (torch_npu)" if _HAS_NPU else ""))

    is_npu = device.type == "npu"
    load_kwargs = dict(trust_remote_code=True, torch_dtype=torch.float32)
    if is_npu:
        load_kwargs["attn_implementation"] = "eager"  # NPU 规避 flash_attn

    processor = XVLAProcessor.from_pretrained(args.model_path)
    model = XVLA.from_pretrained(args.model_path, **load_kwargs).to(device).to(torch.float32)
    model.eval()
    print("✅ 模型加载完成，启动服务器...")
    model.run(processor=processor, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
