"""OpenVLA NPU 闭环验证 v2：修复 pos 参考系根因

根因（阶段12定位）：
- OpenVLA 输出 delta action（end-effector deltas，见 openvla.py:47）
- 但 X-VLA client 默认 act_type="abs"，强制 robot.controller.use_delta=False
- use_delta=False 时 env.step 把 action 的 [pos3] 当**绝对目标坐标**解释
- OpenVLA 输出 delta pos=[0.096,0.035,-0.003] 被当绝对目标坐标
  → 机器人瞬间被指令拉到工作空间外的位置 [0.096,0.035,-0.003]
  → 当前 ee_pos=[-0.211,-0.011,1.174] 离目标很远，且 z=-0.003 在 workspace 下方
  → 永远 done=False，闭环 0%

修复方案（路径A，与官方 run_libero_eval.py:228 等价）：
- 让 client 走 act_type="rel" 路径，env 保持默认 use_delta=True
- server 输出的 7 维 delta action 直接走 env.step(action)
- pos=[0.096,0.035,-0.003] 被正确解释为"末端相对移动 (0.096, 0.035, -0.003)"

验证：
- 3ep × task0 闭环验证，预期成功率显著 >0%（官方基准 spatial 84.7%）
"""
import os, sys, time, numpy as np

# 渲染环境
os.environ.update(NUMBA_DISABLE_JIT='1', MUJOCO_GL='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
                   MESA_LOADER_DRIVER_OVERRIDE='swrast', PYOPENGL_PLATFORM='osmesa')
os.environ['LIBGL_DRIVERS_PATH'] = os.path.expanduser('~/render_libs/dri')
os.environ['LD_LIBRARY_PATH'] = os.path.expanduser('~/render_libs') + ':/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH','')

sys.path.insert(0, os.environ.get('X_VLA_ROOT', os.path.expanduser('~/work/X-VLA')) + '/evaluation/libero')
from libero_client import LIBEROEval, ClientModel

SERVER = "127.0.0.1"
PORT = 8011
NUM_EP = 3
SUITE = "libero_spatial"
SEED = 42
ACT_TYPE = "rel"   # === 关键修复：用 rel 路径让 env 保持 use_delta=True ===

print(f"=== OpenVLA NPU 闭环验证 v2: {SUITE} task0, {NUM_EP}ep, seed{SEED}, act_type={ACT_TYPE} ===")
print(f"server: http://{SERVER}:{PORT}/act")
print(f"根因修复：act_type=rel → use_delta=True → server 的 delta pos 被正确解释为相对移动")

policy = ClientModel(host=SERVER, port=PORT)

# === monkey-patch: 强制 steps=1 让 client 每步都推理 ===
# OpenVLA 输出绝对 action 不能缓存，每步都该推理
# 注意：delta 模式下每步推理也是正确的（OpenVLA 训练时每步预测 delta）
_orig_fmt = policy._format_query
def _fmt_steps1(obs, goal):
    p = _orig_fmt(obs, goal)
    p["steps"] = 1
    return p
policy._format_query = _fmt_steps1
# === patch end ===

evaluator = LIBEROEval(
    task_suite_name=SUITE,
    num_episodes=NUM_EP,
    init_seed=SEED,
    act_type=ACT_TYPE,   # === 关键修复：rel 路径 ===
)

# 跑 task0 的 3 个 episode（看 success 判定闭环）
t0 = time.time()
successes = []
for ep in range(NUM_EP):
    ep_t0 = time.time()
    env, lang, obs = evaluator._init_env(evaluator.task_suite_list[0], task_id=0, ep=ep)
    # 官方必须：前 10 步用 dummy action 让物体稳定
    dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(10):
        obs, _, _, _ = env.step(dummy)
    policy.reset()
    done_flag = False
    steps = 0
    # 打印首步状态供调试
    first_pos = env.env.robots[0].controller.ee_pos.copy()
    print(f"  ep{ep}: 起始 ee_pos={first_pos}")
    for h in range(evaluator.eval_horizon):
        robo_ori = evaluator.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
        robo_pos = env.env.robots[0].controller.ee_pos
        obs['robo_pos'] = robo_pos
        obs['robo_ori'] = robo_ori
        action = policy.step(obs, lang)
        if h == 0:
            print(f"  ep{ep}: 首步 action(7维 delta)={action}")
        obs, reward, done, info = env.step(action)
        steps += 1
        if done:
            done_flag = True
            break
    succ = 1.0 if done_flag else 0.0
    successes.append(succ)
    final_pos = env.env.robots[0].controller.ee_pos.copy()
    print(f"  ep{ep}: success={succ}, steps={steps}, 末态ee_pos={final_pos}, 耗时{(time.time()-ep_t0)/60:.1f}分钟")
    env.close()

print(f"\n=== 汇总 ===")
print(f"成功率: {sum(successes)/NUM_EP:.0%} ({sum(successes)}/{NUM_EP})")
print(f"总耗时: {(time.time()-t0)/60:.1f}分钟")
print(f"对比官方基准 spatial 84.7%")
