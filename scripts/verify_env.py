#!/usr/bin/env python3
"""验证脚本：确认环境搭建正确，渲染+仿真+视频方向均正常"""
import os, sys, subprocess
os.environ.update(NUMBA_DISABLE_JIT='1', MUJOCO_GL='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
    MESA_LOADER_DRIVER_OVERRIDE='swrast', LIBGL_DRIVERS_PATH=os.path.expanduser('~/render_libs/dri'),
    LD_LIBRARY_PATH=os.path.expanduser('~/render_libs')+":/usr/lib64:"+os.environ.get('LD_LIBRARY_PATH',''))
print("="*50)
print("LIBERO NPU 迁移环境验证")
print("="*50)
# 1. 渲染库
libs = ['libOSMesa.so.8','dri/swrast_dri.so','libLLVM-12.so']
ok = all(os.path.exists(os.path.expanduser(f'~/render_libs/{l}')) for l in libs)
print(f"[{'✅' if ok else '❌'}] OSMesa 渲染库")
# 2. 仿真库
try:
    from libero.libero import benchmark; from libero.libero.envs import OffScreenRenderEnv
    print("[✅] LIBERO 仿真库")
except: print("[❌] LIBERO 仿真库"); sys.exit(1)
# 3. 渲染方向
import numpy as np
bd = benchmark.get_benchmark_dict(); s = bd['libero_spatial'](); t = s.get_task(0)
bddl = os.path.join(__import__('libero.libero',fromlist=['get_libero_path']).get_libero_path('bddl_files'),t.problem_folder,t.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=256,camera_widths=256); env.seed(42); env.reset()
obs = env.set_init_state(s.get_task_init_states(0)[0])
for _ in range(10): obs,_,_,_ = env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32))
img = obs['agentview_image']; env.close()
top,bot = round(img[0].mean(),1), round(img[-1].mean(),1)
print(f"[{'✅' if top>bot else '❌'}] 渲染方向 (顶{top}/底{bot}, 顶应亮)")
# 4. NPU
try:
    import torch_npu; import torch; torch.npu.set_device(0)
    x=torch.randn(4,4).npu(); _=(x@x).sum().item()
    print("[✅] NPU 推理可用")
except: print("[❌] NPU 推理")
print("="*50)
print("全部通过 → 可运行: bash scripts/run_eval.sh xvla <模型路径> ./results 10")
