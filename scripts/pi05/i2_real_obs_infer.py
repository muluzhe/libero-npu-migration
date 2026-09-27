#!/usr/bin/env python3
"""⚠️ SUPERSEDED（2026-09-26）：本脚本已被 scripts/pi05/gpu_infer_compare.py 取代——后者已修复
P4.28 state schema 漂移（对齐 server_v2.py 的 [pos3, axis_angle3, gripper_qpos2]）并支持
NPU/GPU 双端运行（--device npu:0 / cuda:0），NPU vs GPU 用同一脚本同一 obs 严格对比。
本脚本保留仅作历史参考（其旧 schema [pos3, quat4, grip1] 会污染对比结论，勿再使用）。

I2验证：用真实libero env obs跑官方便样ckpt推理，对比server predict_action_chunk输出方向。

逻辑链：
1. 抓真实libero env首帧obs（agentview+wrist图像+state），与server同schema
2. 跑select_action推理（内部调predict_action_chunk首步，与我们server首chunk首步同一次推理）
3. 打印action输出方向（delta_pos首3维），对比server debug log的chunk首步delta_pos
4. 若方向一致→根因不在API差异/我们server链路，真根因在模型推理本身
若方向不一致→定位我们server链路bug

关键：select_action首步 = predict_action_chunk首chunk首步（同一次推理调用）
"""
import os, sys, json, traceback
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'swrast'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

sys.path.insert(0, os.path.join(os.environ.get('LEROBOT_PI05_ROOT', os.path.expanduser('~/work/lerobot_pi05')), 'src'))

import torch
import numpy as np

CKPT = os.environ.get('PI05_CKPT', os.path.expanduser('~/work/lerobot_pi05_libero_official'))

def grab_real_obs():
    """抓真实libero env首帧obs，与server同schema（agentview+wrist图像+state 8维）"""
    print('=== 抓真实libero env首帧obs ===', flush=True)
    from libero_env_init import init_libero_env  # 若无此模块用直接方式
    return None

def grab_real_obs_direct():
    """直接用robosuite抓libero spatial task0首帧obs（X-VLA libero_client.py基准建env）"""
    print('=== 直接用robosuite抓libero spatial task0首帧obs ===', flush=True)
    import os
    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    benchmark = get_benchmark("libero_spatial")()  # ABCMeta类需实例化
    task = benchmark.get_task(0)
    task_name = task.name
    print(f'task0: {task_name}', flush=True)
    # X-VLA libero_client.py:249-263 基准建env
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
    env = OffScreenRenderEnv(**env_args)
    obs = env.reset()
    init_states = benchmark.get_task_init_states(0)
    obs = env.set_init_state(init_states[0])  # ep0 init_state
    print(f'obs keys: {list(obs.keys())}', flush=True)
    return obs, env

def build_server_schema_obs(obs):
    """把raw obs转成server schema（observation.images.image/image2 + observation.state 8维）"""
    import cv2
    # agentview (main) + wrist view
    img_main = obs.get('agentview_image')  # (H,W,3) uint8
    img_wrist = obs.get('robot0_eye_in_hand_image')
    if img_wrist is None:
        img_wrist = img_main.copy()
    # server: client已 _flip_agentview 双翻 H+W，这里抓raw obs需同flip对齐训练
    # 但I2只对比推理输出方向，flip对齐训练是P5已验证一致，这里跳过flip用raw obs
    img_main_t = torch.from_numpy(img_main).permute(2, 0, 1)  # (3,H,W)
    img_wrist_t = torch.from_numpy(img_wrist).permute(2, 0, 1)
    # state 8维：pos3 + quat4 + grip1（server:118 真值）
    eef_pos = obs.get('robot0_eef_pos')  # (3,)
    eef_quat = obs.get('robot0_eef_quat')  # (4,) wxyz
    gripper = obs.get('robot0_gripper_qpos')  # (2,)
    # server:118 state_8 = [pos3, quat4, grip1]
    grip1 = gripper.mean() if gripper is not None else 0.0
    state_8 = np.concatenate([eef_pos, eef_quat, [grip1]])[:8]
    state_8_t = torch.from_numpy(state_8).float()
    return {
        'observation.images.image': img_main_t.unsqueeze(0),  # (1,3,H,W)
        'observation.images.image2': img_wrist_t.unsqueeze(0),
        'observation.state': state_8_t.unsqueeze(0),  # (1,8)
        'task': 'pick up the black bowl on the stove\n',  # task0真值
    }

def main():
    try:
        # 1. 抓真实env首帧obs
        obs, env = grab_real_obs_direct()
        print(f'抓真实obs成功', flush=True)
        # 2. 转server schema
        infer_obs = build_server_schema_obs(obs)
        print(f'infer_obs keys: {list(infer_obs.keys())}', flush=True)
        print(f'image shape: {infer_obs["observation.images.image"].shape}', flush=True)
        print(f'state: {infer_obs["observation.state"][0]}', flush=True)
        # 3. 加载官方便样ckpt + preprocess
        print('=== 加载官方便样ckpt ===', flush=True)
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        policy = PI05Policy.from_pretrained(CKPT)
        policy.to('cpu')  # J：用CPU推理绕过NPU OOM
        policy.to(torch.float32)
        policy.eval()
        preprocess, postprocess = make_pre_post_processors(policy.config)
        print('加载+processor OK（CPU推理）', flush=True)
        # 4. preprocess推理
        with torch.inference_mode():
            batch = preprocess(infer_obs)
            # 移到cpu + float32
            def move(d, dev, dt):
                if isinstance(d, torch.Tensor):
                    d = d.to(dev)
                    if d.dtype == torch.float32: d = d.to(dt)
                    return d
                if isinstance(d, dict): return {k: move(v, dev, dt) for k, v in d.items()}
                if isinstance(d, list): return [move(v, dev, dt) for v in d]
                return d
            batch = move(batch, torch.device('cpu'), torch.float32)
            # 5. select_action推理（CPU，内部调predict_action_chunk首步）
            action = policy.select_action(batch)
            action = postprocess(action)
        # 6. 打印输出方向
        if isinstance(action, torch.Tensor):
            act = action.float().cpu().numpy()
        elif isinstance(action, dict) and 'action' in action:
            act = action['action'].float().cpu().numpy()
        else:
            act = np.array(action)
        print(f'=== I2推理输出 ===', flush=True)
        print(f'action shape: {act.shape}', flush=True)
        print(f'action首步[0]: {act[0] if act.ndim >= 2 else act}', flush=True)
        if act.ndim >= 2:
            delta_pos = act[0, :3]  # 首步delta_pos
            print(f'首步delta_pos[:3]: {delta_pos}', flush=True)
            print(f'首步delta_pos方向: x={delta_pos[0]:.3f} y={delta_pos[1]:.3f} z={delta_pos[2]:.3f}', flush=True)
        print(f'=== 对比server debug log chunk首步delta_pos ===', flush=True)
        print(f'server chunk首步delta_pos（官方便样ckpt+F6修复）: [-0.88~-0.80, +0.04~-0.01, -0.97~-0.90]', flush=True)
        print(f'若I2首步delta_pos也朝固定负z方向→根因不在API差异/server链路，真根因在模型推理本身', flush=True)
    except Exception as e:
        print(f'FAIL: {type(e).__name__}: {e}', flush=True)
        traceback.print_exc()

if __name__ == '__main__':
    main()
