"""对比诊断：官方get_vla_action vs 我们server，同一步对比action差异定位0%根因
绕过官方import segfault：直接import get_vla_action（prismatic stub已在server注入）
"""
import os, sys, types, importlib.machinery, numpy as np, torch, requests, json_numpy

# 渲染环境
os.environ.update(NUMBA_DISABLE_JIT='1', MUJOCO_GL='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
                   MESA_LOADER_DRIVER_OVERRIDE='swrast', PYOPENGL_PLATFORM='osmesa')
os.environ['LIBGL_DRIVERS_PATH'] = os.path.expanduser('~/render_libs/dri')
os.environ['LD_LIBRARY_PATH'] = os.path.expanduser('~/render_libs') + ':/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH','')

# 阻断 prismatic/dlimp/tf 加载链（segfault fix）
for m in ('dlimp','rlds','ox_pilot','tensorflow','tensorflow.data'):
    if m not in sys.modules:
        mm = types.ModuleType(m); mm.__spec__ = importlib.machinery.ModuleSpec(m, None); mm.__path__ = []; sys.modules[m] = mm

import torch_npu
sys.path.insert(0, os.environ.get('OPENVLA_ROOT', os.path.expanduser('~/work/openvla')))
sys.path.insert(0, os.environ.get('X_VLA_ROOT', os.path.expanduser('~/work/X-VLA')) + '/evaluation/libero')

# 加载官方模型（bf16+eager，匹配我们server配置）
print("=== 加载官方OpenVLA模型（bf16+sdpa）===")
from transformers import AutoModelForVision2Seq, AutoProcessor
CKPT = os.environ.get('CKPT_BASE', os.path.expanduser('~/work/openvla_checkpoints')) + '/libero-spatial'
processor_off = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
model_off = AutoModelForVision2Seq.from_pretrained(CKPT, attn_implementation='sdpa', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to('npu:0').to(torch.bfloat16)
model_off.eval()
print("✅ 官方模型加载完成")

# 官方 get_vla_action 逻辑（直接复制，绕过import）
from PIL import Image
def get_vla_action_off(obs_full_image, task_label, unnorm_key='libero_spatial', center_crop=True):
    """官方get_vla_action的完整逻辑（bf16模型+bf16 inputs，NPU适配）"""
    image = Image.fromarray(obs_full_image).convert('RGB')
    # center_crop scale=0.9（官官crop_and_resize）
    if center_crop:
        crop_scale = 0.9
        h, w = image.size[1], image.size[0]
        crop_h, crop_w = int(h*np.sqrt(crop_scale)), int(w*np.sqrt(crop_scale))
        y0, x0 = (h-crop_h)//2, (w-crop_w)//2
        import cv2
        img_np = np.array(image)[y0:y0+crop_h, x0:x0+crop_w]
        img_np = cv2.resize(img_np, (w,h), interpolation=cv2.INTER_LANCZOS4)
        image = Image.fromarray(img_np).convert('RGB')
    # prompt格式
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    # inputs（bf16匹配我们server，非官官fp32因NPU要求dtype一致）
    inputs = processor_off(prompt, image).to('npu:0', dtype=torch.bfloat16)
    with torch.no_grad():
        action = model_off.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return np.array(action, dtype=np.float32).flatten()  # 7维

# 我们server的action（通过HTTP）
def get_server_action(image0, task_label):
    payload = {'image0':json_numpy.dumps(image0),'image1':json_numpy.dumps(np.zeros((256,256,3),dtype=np.uint8)),'language_instruction':task_label,'proprio':json_numpy.dumps(np.zeros(20)),'domain_id':3,'steps':10}
    r = requests.post('http://127.0.0.1:8011/act', json=payload, timeout=60).json()
    if 'error' in r: return None, r['error']
    a = np.array(r['action'])
    return a[0], None  # 取第一步（10维）

# 跑同一步对比
print("\n=== 同一步对比：官方action vs 我们server action ===")
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero_client import _flip_agentview
json_numpy.patch()
bd = benchmark.get_benchmark_dict(); suite = bd['libero_spatial']()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(142); env.reset(); env.set_init_state(suite.get_task_init_states(0)[0])
# 前10步dummy让物体稳定
for _ in range(10): obs,_,_,_ = env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32))

# 官方action（用旋转180后的image，匹配get_libero_image的img[::-1,::-1]）
img_off = obs['agentview_image'][::-1,::-1]  # 官方get_libero_image的rotate180
action_off = get_vla_action_off(img_off, task.language)
print(f"官方action(7维): {action_off.round(4)}")
print(f"  pos3={action_off[:3].round(3)} aa3={action_off[3:6].round(3)} grip={action_off[6]:.3f}")

# 我们server action（X-VLA client的_flip_agentview=rotate180）
img_srv = _flip_agentview(obs['agentview_image'])  # 等价img[::-1,::-1]
action_srv, err = get_server_action(img_srv, task.language)
if err:
    print(f"❌ server错误: {err[:200]}")
else:
    print(f"\n我们server action(10维): {action_srv.round(4)}")
    print(f"  pos3={action_srv[:3].round(3)} rot6d={action_srv[3:9].round(3)} grip={action_srv[9]:.3f}")
    # 对比关键维度
    print(f"\n=== 差异分析 ===")
    print(f"pos3 官方vs server: {action_off[:3].round(3)} vs {action_srv[:3].round(3)} 差={np.abs(action_off[:3]-action_srv[:3]).round(4)}")
    print(f"grip 官方vs server: {action_off[6]:.3f} vs {action_srv[9]:.3f}")
    # 官方aa3 → rot6d → aa3验证转换链等价性
    def aa_to_rot6d(aa):
        angle = np.linalg.norm(aa)
        if angle < 1e-8: return np.eye(3)[:,:2].flatten()
        axis = aa/angle; c,s = np.cos(angle), np.sin(angle); x,y,z = axis
        R = np.array([[c+x*x*(1-c), y*x*(1-c)-z*s, z*x*(1-c)+y*s],
                      [x*y*(1-c)+z*s, c+y*y*(1-c), z*y*(1-c)-x*s],
                      [x*z*(1-c)-y*s, y*z*(1-c)+x*s, c+z*z*(1-c)]])
        return R[:,:2].flatten()
    rot6d_from_off = aa_to_rot6d(action_off[3:6])
    print(f"\nrot6d 官方aa转rot6d vs server: {rot6d_from_off.round(3)} vs {action_srv[3:9].round(3)}")
    print(f"  差={np.abs(rot6d_from_off-action_srv[3:9]).round(4)}")
    print(f"当前ee_pos: {np.array(env.env.robots[0].controller.ee_pos).round(3)}")
    print(f"当前ee_quat: {np.array(env.env.robots[0].controller.ee_ori).round(3)}")
env.close()
print("\n=== 对比诊断完成 ===")
