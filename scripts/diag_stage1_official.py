"""对比诊断阶段1：官方get_vla_action推理，存action到文件（避免同时2模型占HBM）
用法: python diag_compare_action.py stage1
"""
import os, sys, types, importlib.machinery, numpy as np, torch, json

os.environ.update(NUMBA_DISABLE_JIT='1', MUJOCO_GL='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
                   MESA_LOADER_DRIVER_OVERRIDE='swrast', PYOPENGL_PLATFORM='osmesa')
os.environ['LIBGL_DRIVERS_PATH'] = os.path.expanduser('~/render_libs/dri')
os.environ['LD_LIBRARY_PATH'] = os.path.expanduser('~/render_libs') + ':/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH','')

for m in ('dlimp','rlds','ox_pilot','tensorflow','tensorflow.data'):
    if m not in sys.modules:
        mm = types.ModuleType(m); mm.__spec__ = importlib.machinery.ModuleSpec(m, None); mm.__path__ = []; sys.modules[m] = mm

import torch_npu
sys.path.insert(0, os.environ.get('OPENVLA_ROOT', os.path.expanduser('~/work/openvla')))
sys.path.insert(0, os.environ.get('X_VLA_ROOT', os.path.expanduser('~/work/X-VLA')) + '/evaluation/libero')

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import cv2

CKPT = os.environ.get('CKPT_BASE', os.path.expanduser('~/work/openvla_checkpoints')) + '/libero-spatial'
print("=== 加载官方OpenVLA模型（bf16+sdpa）===")
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(CKPT, attn_implementation='sdpa', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to('npu:0').to(torch.bfloat16)
model.eval()
print("✅ 官方模型加载完成")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bd = benchmark.get_benchmark_dict(); suite = bd['libero_spatial']()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(142); env.reset(); env.set_init_state(suite.get_task_init_states(0)[0])
for _ in range(10): obs,_,_,_ = env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32))

# 官方get_libero_image: rotate180 + resize224
img = obs['agentview_image'][::-1,::-1]
img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LANCZOS4)
# 官方center_crop scale=0.9
crop_scale=0.9; h=w=224
crop_h, crop_w = int(h*np.sqrt(crop_scale)), int(w*np.sqrt(crop_scale))
y0, x0 = (h-crop_h)//2, (w-crop_w)//2
img = img[y0:y0+crop_h, x0:x0+crop_w]
img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LANCZOS4)
pil_img = Image.fromarray(img).convert('RGB')

prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
inputs = processor(prompt, pil_img).to('npu:0', dtype=torch.bfloat16)
with torch.no_grad():
    action = model.predict_action(**inputs, unnorm_key='libero_spatial', do_sample=False)
action = np.array(action, dtype=np.float32).flatten()

ee_pos = np.array(obs['robot0_eef_pos']).astype(np.float32)
ee_quat = np.array(obs['robot0_eef_quat']).astype(np.float32)
env.close()

result = {'action_off': action.tolist(), 'ee_pos': ee_pos.tolist(), 'ee_quat': ee_quat.tolist(),
          'task_lang': task.language}
import os
os.makedirs('results', exist_ok=True)
with open('results/diag_official_action.json','w') as f: json.dump(result, f)
print(f"✅ 官方action(7维): {action.round(4)}")
print(f"  pos3={action[:3].round(3)} aa3={action[3:6].round(3)} grip={action[6]:.3f}")
print(f"  ee_pos={ee_pos.round(3)} ee_quat={ee_quat.round(3)}")
print(f"✅ 已存 results/diag_official_action.json")
