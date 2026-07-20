"""OpenVLA NPU 优化版闭环验证

性能优化（server_v2.py）：
- 全局预计算 crop 参数（避免每步 np.sqrt + int）
- 固定 seed 只设一次（do_sample=False 时推理确定）
- json_numpy.patch 只调一次
- 移除每步重复的 torch.manual_seed / np.random.seed

验证：3ep × task0，对比优化前 (2.4min/ep, 100% SR)
"""
import os, sys, time, numpy as np

os.environ.update(NUMBA_DISABLE_JIT='1', MUJOCO_GL='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
                   MESA_LOADER_DRIVER_OVERRIDE='swrast', PYOPENGL_PLATFORM='osmesa')
os.environ['LIBGL_DRIVERS_PATH'] = os.path.expanduser('~/render_libs/dri')
os.environ['LD_LIBRARY_PATH'] = os.path.expanduser('~/render_libs') + ':/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH','')

sys.path.insert(0, '${X_VLA_ROOT:?usage: X_VLA_ROOT env var required}/evaluation/libero')
from libero_client import LIBEROEval, ClientModel

SERVER = "127.0.0.1"
PORT = 8011
NUM_EP = 3
SUITE = "libero_spatial"
SEED = 42
ACT_TYPE = "rel"

print(f"=== OpenVLA NPU 优化版闭环验证 ===")
print(f"server: http://{SERVER}:{PORT}/act (优化版 server_v2.py)")

policy = ClientModel(host=SERVER, port=PORT)

# monkey-patch: steps=1 每步推理（delta action 不能 chunk 缓存）
_orig_fmt = policy._format_query
def _fmt_steps1(obs, goal):
    p = _orig_fmt(obs, goal)
    p["steps"] = 1
    return p
policy._format_query = _fmt_steps1

evaluator = LIBEROEval(
    task_suite_name=SUITE,
    num_episodes=NUM_EP,
    init_seed=SEED,
    act_type=ACT_TYPE,
)

t0 = time.time()
successes = []
for ep in range(NUM_EP):
    ep_t0 = time.time()
    env, lang, obs = evaluator._init_env(evaluator.task_suite_list[0], task_id=0, ep=ep)
    dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(10):
        obs, _, _, _ = env.step(dummy)
    policy.reset()
    done_flag = False
    steps = 0
    for h in range(evaluator.eval_horizon):
        robo_ori = evaluator.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
        robo_pos = env.env.robots[0].controller.ee_pos
        obs['robo_pos'] = robo_pos
        obs['robo_ori'] = robo_ori
        action = policy.step(obs, lang)
        obs, reward, done, info = env.step(action)
        steps += 1
        if done:
            done_flag = True
            break
    succ = 1.0 if done_flag else 0.0
    successes.append(succ)
    print(f"  ep{ep}: success={succ}, steps={steps}, 耗时={(time.time()-ep_t0)/60:.1f}分钟", flush=True)
    env.close()

print(f"\n=== 优化版汇总 ===")
print(f"成功率: {sum(successes)/NUM_EP:.0%} ({sum(successes)}/{NUM_EP})")
print(f"总耗时: {(time.time()-t0)/60:.1f}分钟")
print(f"对比优化前: 2.4min/ep, 100% SR (3ep task0)")
