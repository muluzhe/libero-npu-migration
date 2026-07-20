"""OpenVLA NPU 完整 spatial suite 验证

10 task × N ep，对比官方基准 spatial 84.7 ± 0.9%
官方协议：50 trial/task × 1 seed，我们用 5 ep/task 平衡统计意义与耗时

修复要点（阶段12-13定位）：
- OpenVLA 输出 delta action（end-effector deltas）
- 让 client 走 act_type="rel" 路径，env 保持默认 use_delta=True
- server 输出的 7 维 delta action 直接 env.step(action)
"""
import os, sys, time, json, numpy as np
from pathlib import Path

# 渲染环境
os.environ.update(NUMBA_DISABLE_JIT='1', MUJOCO_GL='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
                   MESA_LOADER_DRIVER_OVERRIDE='swrast', PYOPENGL_PLATFORM='osmesa')
os.environ['LIBGL_DRIVERS_PATH'] = os.path.expanduser('~/render_libs/dri')
os.environ['LD_LIBRARY_PATH'] = os.path.expanduser('~/render_libs') + ':/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH','')

sys.path.insert(0, '${X_VLA_ROOT:?usage: X_VLA_ROOT env var required}/evaluation/libero')
from libero_client import LIBEROEval, ClientModel

SERVER = "127.0.0.1"
PORT = 8011
SUITE = "libero_spatial"
SEED = 42
NUM_EP_PER_TASK = 5
EVAL_HORIZON = 220  # 官方 spatial 用 220 步上限（最长训练 demo 193 步）
ACT_TYPE = "rel"    # === 关键修复：rel 路径让 env 保持 use_delta=True ===

# 输出
OUT_DIR = Path('${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}/results')
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_DIR / 'spatial_full_results.json'
LOG_FILE = OUT_DIR / 'spatial_full.log'

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + "\n")

log(f"=== OpenVLA NPU 完整 spatial suite 验证 ===")
log(f"server: http://{SERVER}:{PORT}/act")
log(f"配置: {SUITE}, {NUM_EP_PER_TASK}ep/task, horizon={EVAL_HORIZON}, act_type={ACT_TYPE}")
log(f"官方基准: spatial 84.7 ± 0.9%")

policy = ClientModel(host=SERVER, port=PORT)

# === monkey-patch: 强制 steps=1 让 client 每步都推理 ===
_orig_fmt = policy._format_query
def _fmt_steps1(obs, goal):
    p = _orig_fmt(obs, goal)
    p["steps"] = 1
    return p
policy._format_query = _fmt_steps1

evaluator = LIBEROEval(
    task_suite_name=SUITE,
    eval_horizon=EVAL_HORIZON,
    num_episodes=NUM_EP_PER_TASK,
    init_seed=SEED,
    act_type=ACT_TYPE,
)

task_suite = evaluator.task_suite_list[0]
num_tasks = len(task_suite.tasks)
log(f"spatial suite: {num_tasks} tasks, {NUM_EP_PER_TASK} ep each = {num_tasks*NUM_EP_PER_TASK} rollouts total")

# 逐 task × ep 跑闭环
results = {}  # task_id -> [success per ep]
t_start = time.time()
total_rollouts = 0
total_successes = 0

for task_id in range(num_tasks):
    task = task_suite.get_task(task_id)
    task_lang = task.language
    task_succs = []
    log(f"\n--- task{task_id}: {task_lang} ---")

    for ep in range(NUM_EP_PER_TASK):
        ep_t0 = time.time()
        try:
            env, lang, obs = evaluator._init_env(task_suite, task_id=task_id, ep=ep)
            # 前 10 步 dummy action 让物体稳定
            dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
            for _ in range(10):
                obs, _, _, _ = env.step(dummy)
            policy.reset()
            done_flag = False
            steps = 0
            for h in range(EVAL_HORIZON):
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
            task_succs.append(succ)
            total_rollouts += 1
            total_successes += int(done_flag)
            log(f"  task{task_id} ep{ep}: success={int(done_flag)}, steps={steps}, 耗时={(time.time()-ep_t0)/60:.1f}min")
        except Exception as e:
            import traceback
            log(f"  task{task_id} ep{ep}: EXCEPTION {e}")
            log(traceback.format_exc()[-300:])
            task_succs.append(0.0)
        finally:
            try: env.close()
            except: pass

    task_sr = sum(task_succs) / max(len(task_succs), 1)
    results[task_id] = {
        'language': task_lang,
        'successes': task_succs,
        'task_success_rate': task_sr,
    }
    log(f"  >>> task{task_id} SR = {task_sr:.0%} ({sum(task_succs)}/{len(task_succs)})")

    # 增量保存（防止中断丢失）
    with open(RESULTS_JSON, 'w') as f:
        json.dump({
            'suite': SUITE,
            'config': {
                'num_ep_per_task': NUM_EP_PER_TASK,
                'eval_horizon': EVAL_HORIZON,
                'act_type': ACT_TYPE,
                'seed': SEED,
            },
            'running_success_rate': total_successes / max(total_rollouts, 1),
            'total_rollouts': total_rollouts,
            'total_successes': total_successes,
            'tasks': results,
        }, f, indent=2)

# 汇总
overall_sr = total_successes / max(total_rollouts, 1)
log(f"\n=== 汇总 ===")
log(f"总 rollouts: {total_rollouts}")
log(f"总 successes: {total_successes}")
log(f"Overall SR: {overall_sr:.1%}")
log(f"对比官方基准 spatial 84.7 ± 0.9%")
log(f"总耗时: {(time.time()-t_start)/60:.1f} 分钟")

# 逐 task SR 表
log(f"\n=== 逐 task SR ===")
for tid, r in results.items():
    log(f"  task{tid} ({r['language'][:50]}...): {r['task_success_rate']:.0%}")

# 最终保存
final = {
    'suite': SUITE,
    'config': {
        'num_ep_per_task': NUM_EP_PER_TASK,
        'eval_horizon': EVAL_HORIZON,
        'act_type': ACT_TYPE,
        'seed': SEED,
    },
    'overall_success_rate': overall_sr,
    'total_rollouts': total_rollouts,
    'total_successes': total_successes,
    'total_time_min': (time.time() - t_start) / 60,
    'official_baseline': 0.847,
    'tasks': results,
}
with open(RESULTS_JSON, 'w') as f:
    json.dump(final, f, indent=2)
log(f"结果已保存: {RESULTS_JSON}")
