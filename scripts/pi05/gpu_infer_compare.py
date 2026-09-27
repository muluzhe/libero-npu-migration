#!/usr/bin/env python3
"""推理输出方向对比脚本（NPU / GPU 通用）——定论 NPU 算子是否是 pi0.5 闭环 0% 根因。

★ 两阶段设计（2026-09-26，P4.28c）：OSMesa GL 上下文与 NPU runtime 同进程共存/交错 teardown
  均会 segfault（exit 139，实测两种死法：env 开着推理崩 / NPU 初始化后 env.close() 崩）。
  NPU 端必须分两个进程跑：
    阶段1 grab ：纯 OSMesa（不 import torch_npu），抓真实 obs 存 npz，关 env 退出
    阶段2 infer：纯 NPU（不 import mujoco），读 npz 推理，存结果 JSON
  GPU 端（CUDA 无此冲突）可用 --phase auto 单进程一次跑完，也可同样两阶段保持完全一致。

★ P4.28 修复链（2026-09-26）：
  a) state schema 对齐 server_v2.py P4.7/P4.8：[pos3, axis_angle3, gripper_qpos2]
  b) processor 构建传 ckpt 路径（make_pre_post_processors(config, ckpt)），否则用硬编码
     tokenizer + 跳过 norm_stats 归一化
  c) 两阶段进程隔离（本条）
  d) 任务语言从 benchmark 动态取（旧版硬编码了 task7 的语言，喂错任务）
  e) 双 pass 推理：unflipped + flipped（client _flip_agentview 等价），两个结果都进 JSON

用法：
  # NPU 端（本机，两阶段）：
  python gpu_infer_compare.py --phase grab                       # 存 /tmp/pi05_obs.npz
  python gpu_infer_compare.py --phase infer --device npu:0 --dtype float32
  # GPU 端（RTX4090，单进程即可）：
  python gpu_infer_compare.py --phase auto --ckpt /path/to/ckpt --device cuda:0 --dtype float32
  # 结果：/tmp/gpu_infer_result.json（两端互相对比定论 D/G）
"""
import os, sys, json, argparse

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
# 注意：LD_LIBRARY_PATH（OSMesa 渲染库）必须在 python 启动前 export（动态链接器只读一次）；
# GPU 服务器请确保系统已装 libosmesa6 或改用 MUJOCO_GL=egl。

DEFAULT_OBS_NPZ = '/tmp/pi05_obs.npz'
DEFAULT_RESULT = '/tmp/gpu_infer_result.json'


def quat_wxyz_to_axis_angle(eef_quat_wxyz):
    """mujoco sim body quat [w,x,y,z] → axis_angle(3)。与 server_v2.py P4.8 同源链：
    wxyz → xyzw → 2*acos(w)*(xyz/||xyz||)，与 lerobot env_processor._quat2axisangle 一致。"""
    import numpy as np
    q = np.array(eef_quat_wxyz, dtype=np.float32).flatten()[:4]
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
    w_q = float(np.clip(quat_xyzw[3], -1.0, 1.0))
    norm_xyz = float(np.linalg.norm(quat_xyzw[:3]))
    if norm_xyz < 1e-10 or abs(w_q) > 1.0 - 1e-10:
        return np.zeros(3, dtype=np.float32)  # 单位 quat → 0 aa
    angle = 2.0 * np.arccos(w_q)
    return (quat_xyzw[:3] / norm_xyz) * angle


# ============ 阶段 1：抓 obs（纯 OSMesa，勿 import torch/torch_npu） ============

def phase_grab(obs_npz):
    print('=== [grab] 抓真实 libero spatial task0 首帧 obs（纯 OSMesa 进程）===', flush=True)
    import numpy as np
    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    benchmark = get_benchmark('libero_spatial')()  # ABCMeta 类需实例化
    task = benchmark.get_task(0)
    task_lang = getattr(task, 'language', None) or task.name.replace('_', ' ')
    print(f'task0: {task.name}', flush=True)
    print(f'task0 language: {task_lang}', flush=True)
    task_bddl_file = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
    obs = env.reset()
    init_states = benchmark.get_task_init_states(0)
    obs = env.set_init_state(init_states[0])  # ep0 init_state
    env.close()  # 本进程无 NPU runtime，close 安全
    np.savez_compressed(
        obs_npz,
        agentview=obs['agentview_image'],                      # (256,256,3) uint8
        wrist=obs['robot0_eye_in_hand_image'],                 # (256,256,3) uint8
        eef_pos=np.asarray(obs['robot0_eef_pos'], dtype=np.float32),
        eef_quat=np.asarray(obs['robot0_eef_quat'], dtype=np.float32),
        task_lang=task_lang,
    )
    print(f'=== [grab] obs 已存 {obs_npz}（agentview/wrist/eef_pos/eef_quat/task_lang）===', flush=True)


# ============ 阶段 2：推理（纯 NPU/GPU，勿 import mujoco） ============

def build_server_schema_obs(npz, flip_agentview=False):
    """npz obs → server schema（observation.images.image/image2 + observation.state 8维）

    state_8 = [pos3, axis_angle3, gripper_qpos2]，与 server_v2.py:165 完全一致（P4.28a）。
    flip_agentview=True 时对 agentview 做 img[::-1,::-1]（与 X-VLA client 的 _flip_agentview
    完全一致，libero_client.py:69/160），匹配闭环时模型实际收到的图像分布；wrist 图像
    client 不翻转（libero_client.py:161），这里同样保持原样。
    """
    import torch
    import numpy as np
    img_main = npz['agentview']
    img_wrist = npz['wrist']
    if flip_agentview:
        img_main = img_main[::-1, ::-1].copy()  # client _flip_agentview 等价（双翻 H+W）
    img_main_t = torch.from_numpy(img_main).permute(2, 0, 1)  # (3,H,W)
    img_wrist_t = torch.from_numpy(img_wrist).permute(2, 0, 1)
    axis_angle = quat_wxyz_to_axis_angle(npz['eef_quat'])
    # gripper_qpos(2维)：对齐 server——client 不传真实 qpos，server 用训练 q50 中位数
    gripper_qpos2 = np.array([0.02636, -0.02728], dtype=np.float32)
    state_8 = np.concatenate([npz['eef_pos'], axis_angle, gripper_qpos2])
    print(f'state_8 (P4.7/P4.8 schema): {state_8} | flip_agentview={flip_agentview}', flush=True)
    return {
        'observation.images.image': img_main_t.unsqueeze(0),  # (1,3,H,W)
        'observation.images.image2': img_wrist_t.unsqueeze(0),
        'observation.state': torch.from_numpy(state_8).float().unsqueeze(0),  # (1,8)
        'task': str(npz['task_lang']),  # 从 benchmark 动态取的 task0 真值语言
    }


def run_inference(policy, preprocess, postprocess, infer_obs, DEV, DT):
    """preprocess → select_action（内部调 predict_action_chunk 首步）→ postprocess，返回 numpy"""
    import torch
    with torch.inference_mode():
        batch = preprocess(infer_obs)
        def move(d, dev, dt):
            if isinstance(d, torch.Tensor):
                d = d.to(dev)
                if d.dtype == torch.float32:
                    d = d.to(dt)
                return d
            if isinstance(d, dict):
                return {k: move(v, dev, dt) for k, v in d.items()}
            if isinstance(d, list):
                return [move(v, dev, dt) for v in d]
            return d
        batch = move(batch, torch.device(DEV), DT)
        action = policy.select_action(batch)
        action = postprocess(action)
    if isinstance(action, torch.Tensor):
        return action.float().cpu().numpy()
    if isinstance(action, dict) and 'action' in action:
        return action['action'].float().cpu().numpy()
    import numpy as np
    return np.array(action)


def phase_infer(args, npz):
    import numpy as np
    import torch
    DT = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    DEV = args.device
    CKPT = args.ckpt

    # 0. 设备可用性
    print(f'=== [infer] 设备可用性（{DEV}）===', flush=True)
    print(f'torch: {torch.__version__}', flush=True)
    if DEV.startswith('cuda'):
        assert torch.cuda.is_available(), 'CUDA 不可用！请检查 nvidia-smi + cuda 驱动'
        print(f'GPU0: {torch.cuda.get_device_name(0)}', flush=True)
    elif DEV.startswith('npu'):
        import torch_npu
        assert torch.npu.is_available(), 'NPU 不可用！请检查 npu-smi + CANN 环境（后台任务需先 source Ascend set_env.sh）'
        print(f'npu available: True count: {torch.npu.device_count()}', flush=True)
    else:
        print('CPU 推理（慢，仅小规模验证用）', flush=True)

    # 1. 加载官方便样 ckpt（需含 F6 use_peft 修复的 lerobot 源码）
    print('=== [infer] 加载官方便样 ckpt ===', flush=True)
    lerobot_root = os.environ.get('LEROBOT_PI05_ROOT', os.path.expanduser('~/work/lerobot_pi05'))
    sys.path.insert(0, os.path.join(lerobot_root, 'src'))
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05 import modeling_pi05 as _mpi05
    from lerobot.policies.factory import make_pre_post_processors

    # ★ P4.28d：NPU 必须移植 server_v2.py:316-334 的采样 monkey-patch——
    # 原版 sample_noise 用 _inference_dtype()（config 默认 bf16）调 torch.normal，
    # aclnnNormalFloatFloat 在 NPU 上对 bf16 输出不支持（本机 CANN 8.5.2 实测直接 segfault）。
    # server_v2 靠此 patch 跑通全部闭环推理；本脚本漏掉它 → 纯 NPU 推理段 segfault（v11 实测）。
    # 数值上与原版等价（fp32 采样再 cast），GPU 端不打 patch 保持上游行为。
    if DEV.startswith('npu'):
        _fp32_time = _mpi05.PI05Pytorch.sample_time
        def _patched_sample_noise(self, shape, device):
            with torch.no_grad():
                return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device).to(self._inference_dtype())
        def _patched_sample_time(self, bsize, device):
            with torch.no_grad():
                t = _fp32_time(self, bsize, device)
                return t.to(dtype=self._inference_dtype(), device=device)
        _mpi05.PI05Pytorch.sample_noise = _patched_sample_noise
        _mpi05.PI05Pytorch.sample_time = _patched_sample_time
        print('NPU monkey-patch applied: sample_noise/sample_time forced fp32（与 server_v2 一致）', flush=True)

    policy = PI05Policy.from_pretrained(CKPT)
    policy.to(DEV)
    policy.to(DT)
    policy.eval()
    # ★ P4.28b：必须传 CKPT 路径——不传则用 processor_pi05.py:145 硬编码的
    # google/paligemma-3b-pt-224（HF 被代理阻断即失败）且不加载 norm_stats（归一化跳过）。
    # 与 server_v2.py:362 的构建方式完全一致。
    preprocess, postprocess = make_pre_post_processors(
        policy.config, CKPT,
        preprocessor_overrides={"device_processor": {"device": DEV}},
    )
    print(f'加载+processor OK（device={DEV}, dtype={DT}）', flush=True)
    print(f'config: chunk_size={policy.config.chunk_size} n_action_steps={policy.config.n_action_steps} use_peft={policy.config.use_peft}', flush=True)

    # 2. 双 pass 推理（unflipped / flipped 各一次，同一模型只加载一次）
    print(f'=== [infer] 推理（unflipped / flipped 各一次）===', flush=True)
    results_2pass = {}
    for flip in (False, True):
        infer_obs = build_server_schema_obs(npz, flip_agentview=flip)
        act = run_inference(policy, preprocess, postprocess, infer_obs, DEV, DT)
        tag = 'flipped' if flip else 'unflipped'
        print(f'=== {DEV} 推理输出（{tag}）===', flush=True)
        print(f'action shape: {act.shape}', flush=True)
        if act.ndim >= 2:
            d = act[0, :3]
            print(f'首步 delta_pos[:3] ({tag}): {d}', flush=True)
            results_2pass[tag] = {'delta_pos_first3': d.tolist(), 'action_shape': list(act.shape)}
        else:
            print(f'action ({tag}): {act}', flush=True)
            results_2pass[tag] = {'delta_pos_first3': act.tolist(), 'action_shape': list(act.shape)}

    # 3. 对比基准
    print('=== 对比基准 ===', flush=True)
    print('历史 NPU 闭环 server debug 基准（官方便样 ckpt + F6 修复，P4.7/P4.8 schema）:', flush=True)
    print('  chunk 首步 delta_pos: [-0.88~-0.80, +0.04~-0.01, -0.97~-0.90]（朝固定负 z）', flush=True)
    for tag, r in results_2pass.items():
        print(f'  本机 {tag}: {r["delta_pos_first3"]}', flush=True)
    print('定论逻辑：两端（NPU vs GPU）同方向 → NPU 算子不是根因；方向不同 → NPU 算子根因', flush=True)
    print('flipped 若≈闭环负 z → 模型对闭环分布输入确实输出负 z（根因在模型）；', flush=True)
    print('flipped 若也偏离 → 闭环 state 构造链（P4.8 perm+signs 路径）重新存疑。', flush=True)

    # 4. 存结果 JSON
    result = {
        'device': DEV,
        'dtype': args.dtype,
        'ckpt': CKPT,
        'task': str(npz['task_lang']),
        'state_schema': 'pos3+axis_angle3+gripper_qpos2 (P4.7/P4.8, 与 server_v2.py 对齐)',
        'processor': 'make_pre_post_processors(config, ckpt) 含 norm_stats（P4.28b 对齐）',
        'chunk_size': policy.config.chunk_size,
        'n_action_steps': policy.config.n_action_steps,
        'two_pass': results_2pass,
        'delta_pos_first3': results_2pass.get('unflipped', {}).get('delta_pos_first3'),
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'=== 结果已存 {args.out}，两端互相对比 ===', flush=True)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['grab', 'infer', 'auto'], default='auto',
                        help='grab=抓obs存npz；infer=读npz推理；auto=单进程一次跑完（GPU/CPU 用，NPU 必须两阶段）')
    parser.add_argument('--ckpt', default=os.environ.get('PI05_CKPT', os.path.expanduser('~/work/lerobot_pi05_libero_official')),
                        help='官方便样 ckpt 路径（lerobot/pi05-libero）')
    parser.add_argument('--device', default='cuda:0', help='推理设备：cuda:0 / npu:0 / cpu')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'bfloat16'])
    parser.add_argument('--obs_npz', default=DEFAULT_OBS_NPZ, help='obs 中转文件（grab 存 / infer 读）')
    parser.add_argument('--out', default=DEFAULT_RESULT, help='结果 JSON 输出路径')
    args = parser.parse_args()

    if args.phase == 'grab':
        phase_grab(args.obs_npz)
        return

    if args.phase == 'infer':
        import numpy as np
        npz = np.load(args.obs_npz, allow_pickle=True)
        phase_infer(args, npz)
        return

    # auto：单进程 grab + infer（GPU/CPU 无 GL/NPU 冲突时用）
    phase_grab(args.obs_npz)
    import numpy as np
    npz = np.load(args.obs_npz, allow_pickle=True)
    phase_infer(args, npz)


if __name__ == '__main__':
    main()
