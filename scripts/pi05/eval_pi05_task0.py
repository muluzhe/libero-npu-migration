#!/usr/bin/env python3
"""
历史弃用入口：当前 PI0.5 server 接口与本脚本不兼容，请使用 eval_pi05_spatial.py。

PI0.5 LIBERO 单任务闭环验证脚本（阶段4）
- act_type="rel"：pi0.5 输出 delta action，走 OpenVLA 验证过的 rel 路径（use_delta=True）
- task0 + 3 episodes，验证推理链路 + 闭环跑通 + success 判定
- 与 X-VLA client 的 libero_client.py LIBEROEval 接口对接
- 不追求成功率量级，只验证 NPU 推理 + 闭环能跑通
"""
import os
# === 环境变量必须在任何 import 之前设置（libero/robosuite/mujoco 加载时就要用）===
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["LIBGL_DRIVERS_PATH"] = os.path.expanduser("~/render_libs/dri")
os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/render_libs") + ":/usr/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import sys
import time
import warnings
import argparse

warnings.filterwarnings("ignore")

# X-VLA libero_client 路径
X_VLA_ROOT = os.environ.get("X_VLA_ROOT", os.path.expanduser("~/work/X-VLA"))
sys.path.insert(0, os.path.join(X_VLA_ROOT, "evaluation/libero"))

from libero_client import LIBEROEval  # noqa: E402


def main():
    raise SystemExit("eval_pi05_task0.py 已弃用且不适配当前 server；请使用 scripts/pi05/eval_pi05_spatial.py")

    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8012)
    parser.add_argument("--task_suite", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--eval_horizon", type=int, default=220,
                        help="官方 spatial 用 220 步上限")
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--act_type", default="rel",
                        choices=["abs", "rel"],
                        help="pi0.5 输出 delta action，走 rel 路径让 env 保持 use_delta=True")
    parser.add_argument("--output_dir", default=os.path.expanduser("~/work/pi05_task0_results"))
    args = parser.parse_args()

    t0 = time.time()
    print(f"=== PI0.5 LIBERO 单任务闭环验证 ===")
    print(f"server: {args.server_ip}:{args.server_port}")
    print(f"suite: {args.task_suite} | task_id: {args.task_id} | episodes: {args.num_episodes}")
    print(f"act_type: {args.act_type} | horizon: {args.eval_horizon} | seed: {args.init_seed}")

    evaluator = LIBEROEval(
        task_suite_name=args.task_suite,
        eval_horizon=args.eval_horizon,
        act_type=args.act_type,
        num_episodes=args.num_episodes,
        init_seed=args.init_seed,
    )

    # 单 task × N episodes
    task_suite = evaluator.task_suite_list[0]
    from libero_client import ClientModel
    client = ClientModel(args.server_ip, args.server_port)

    os.makedirs(args.output_dir, exist_ok=True)
    successes = []
    goal = ""
    for ep in range(args.num_episodes):
        ep_t0 = time.time()
        # _init_env 已做 env.reset + set_init_state + 10步settle + act_type 设置，返回 (env, lang, obs)
        env, lang, obs = evaluator._init_env(task_suite, task_id=args.task_id, ep=ep)
        goal = lang
        print(f"\n[ep {ep}] task: {goal}")
        client.reset()
        done = False
        step = 0
        success = 0.0
        while step < args.eval_horizon and not done:
            # 与官方 _rollout 一致：每步从 controller 注入 robo_ori/robo_pos（client._format_query 要）
            robo_ori = evaluator.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
            obs['robo_ori'] = robo_ori
            obs['robo_pos'] = env.env.robots[0].controller.ee_pos
            action = client.step(obs, goal)
            obs, _, done, info = env.step(action)
            step += 1
            if done:
                success = 1.0 if env.check_success() else 0.0
                break
        if not done:
            success = 1.0 if env.check_success() else 0.0
        successes.append(success)
        print(f"[ep {ep}] steps={step} success={success} 耗时={time.time()-ep_t0:.1f}s")
        try: env.close()
        except: pass

    sr = sum(successes) / len(successes)
    print(f"\n=== 完成 ===")
    print(f"成功率: {sr*100:.1f}% ({int(sum(successes))}/{len(successes)})")
    print(f"总耗时: {time.time()-t0:.1f}s")
    # 保存结果
    import json
    out = {
        "model": "pi0.5_lerobot_pi05-libero",
        "suite": args.task_suite,
        "task_id": args.task_id,
        "task_language": goal,
        "act_type": args.act_type,
        "init_seed": args.init_seed,
        "num_episodes": args.num_episodes,
        "success_rate": sr,
        "successes": successes,
        "total_time_s": time.time() - t0,
    }
    out_p = os.path.join(args.output_dir, "task0_results.json")
    json.dump(out, open(out_p, "w"), indent=2)
    print(f"结果保存: {out_p}")


if __name__ == "__main__":
    main()
