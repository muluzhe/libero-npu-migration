#!/usr/bin/env python3
"""
PI0.5 LIBERO 完整 spatial suite 验证脚本（阶段5；P4.29 修复版）
- libero_spatial: 10 task × N episodes/task
- act_type="rel"：pi0.5 输出 delta action，走 OpenVLA 验证过的 rel 路径
- 与 openpi 官方基准 96.85%（spatial 98.8）对比
- 复用 libero_client.py 的 LIBEROEval._init_env（官方便样，零改动）

P4.29 输入构造修复（闭环 0% 真根因，配套 models/pi0/server_v2.py）：
1) 每步重置 client.proprio = None → 每次查询携带 fresh closed_loop_proprio
   （旧链路 client.proprio 冻结在首帧，server 以为机器人从未移动）
2) monkey-patch _format_query 注入 env 真值 robot0_eef_quat（robosuite (x,y,z,w)，
   训练 state 同源）+ robot0_gripper_qpos —— client 文件保持官方原样零改动
- 进度实时写 results/pi05_spatial/progress.log，结果写 spatial_results.json
"""
import os
# === 环境变量必须在任何 import 之前设置 ===
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "swrast"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LIBGL_DRIVERS_PATH"] = os.path.expanduser("~/render_libs/dri")
os.environ["LD_LIBRARY_PATH"] = os.path.expanduser("~/render_libs") + ":/usr/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

import sys
import time
import json
import warnings
import argparse
import atexit
from datetime import datetime
import shlex
import socket
import subprocess
import traceback

import numpy as np
import imageio
import json_numpy
json_numpy.patch()

warnings.filterwarnings("ignore")

X_VLA_ROOT = os.environ.get("X_VLA_ROOT", os.path.expanduser("~/work/X-VLA"))
sys.path.insert(0, os.path.join(X_VLA_ROOT, "evaluation/libero"))

from libero_client import LIBEROEval, ClientModel, _flip_agentview  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_ip", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8012)
    parser.add_argument("--task_suite", default="libero_spatial")
    parser.add_argument("--num_tasks", type=int, default=10,
                        help="spatial suite 共 10 task")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="每 task 10 episodes（官方 spatial 用 20ep，NPU 资源受限缩到 10ep）")
    parser.add_argument("--eval_horizon", type=int, default=220,
                        help="官方 spatial 用 220 步上限")
    parser.add_argument("--init_seed", type=int, default=42)
    parser.add_argument("--act_type", default="rel")
    parser.add_argument("--output_dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "pi05_spatial"),
                        help="历史结果所在的父目录；每次运行在其下创建独立时间目录")
    parser.add_argument("--server_cmd", required=True,
                        help="启动本次 PI0.5 server 的命令；输出写入本次运行的 server.log")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    # 不清理或覆盖父目录中的历史 JSON、日志及视频。
    args.output_dir = os.path.join(
        args.output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}")
    os.makedirs(args.output_dir, exist_ok=False)
    video_dir = os.path.join(args.output_dir, "videos", args.task_suite)
    os.makedirs(video_dir, exist_ok=True)
    progress_log = os.path.join(args.output_dir, "progress.log")
    eval_log = os.path.join(args.output_dir, "eval.log")
    server_log = os.path.join(args.output_dir, "server.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
        for path in (progress_log, eval_log):
            with open(path, "a") as f:
                f.write(line)
        print(msg, flush=True)

    def log_exception(exc_type, exc, tb):
        with open(eval_log, "a") as f:
            traceback.print_exception(exc_type, exc, tb, file=f)
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = log_exception

    # 本次运行自行启动 server，确保日志确实来自本次验证，而非既有进程。
    # NPU1 仍由 OpenVLA 占用；只向子进程暴露空闲的 NPU0。
    server_env = os.environ.copy()
    server_env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    server_stream = open(server_log, "w")
    try:
        server = subprocess.Popen(shlex.split(args.server_cmd),
                                  stdout=server_stream, stderr=subprocess.STDOUT,
                                  env=server_env)
    except BaseException:
        server_stream.close()
        raise
    server_stream.close()

    def stop_server():
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    atexit.register(stop_server)
    deadline = time.monotonic() + 600
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"PI0.5 server 已退出（code={server.returncode}）；查看 {server_log}")
        try:
            with socket.create_connection((args.server_ip, args.server_port), timeout=1):
                break
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待 PI0.5 server 超时；查看 {server_log}")
            time.sleep(1)
    log(f"本次运行目录: {args.output_dir} | server 日志: {server_log} | eval 日志: {eval_log}")

    t0 = time.time()
    log(f"=== PI0.5 LIBERO 完整 {args.task_suite} 验证 ===")
    log(f"server: {args.server_ip}:{args.server_port} | tasks: {args.num_tasks} | ep/task: {args.num_episodes} | horizon: {args.eval_horizon} | seed: {args.init_seed} | act_type: {args.act_type}")

    evaluator = LIBEROEval(
        task_suite_name=args.task_suite,
        eval_horizon=args.eval_horizon,
        act_type=args.act_type,
        num_episodes=args.num_episodes,
        init_seed=args.init_seed,
    )
    client = ClientModel(args.server_ip, args.server_port)

    # === P4.29 修复 2：注入 env 真值 quat/gripper_qpos（client 文件零改动） ===
    # server 用 robot0_eef_quat 按 (x,y,z,w) 构造训练同源 state_8 的 aa 维
    _orig_fmt = client._format_query

    def _fmt_pi05(obs, goal):
        p = _orig_fmt(obs, goal)
        p["robot0_eef_quat"] = json_numpy.dumps(
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
        p["robot0_gripper_qpos"] = json_numpy.dumps(
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))
        return p

    client._format_query = _fmt_pi05

    task_suite = evaluator.task_suite_list[0]
    total_eps = args.num_tasks * args.num_episodes
    task_results = []
    all_successes = []
    done_eps = 0

    for tid in range(args.num_tasks):
        task_t0 = time.time()
        task_lang = task_suite.get_task(tid).language
        log(f"\n>>> [task {tid+1}/{args.num_tasks}] {task_lang}")
        ep_succ = []
        episode_metadata = []
        for ep in range(args.num_episodes):
            env, lang, obs = evaluator._init_env(task_suite, task_id=tid, ep=ep)
            client.reset()
            done = False
            step = 0
            success = 0.0
            frames = []
            while step < args.eval_horizon and not done:
                robo_ori = evaluator.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
                obs["robo_ori"] = robo_ori
                obs["robo_pos"] = env.env.robots[0].controller.ee_pos
                # P4.29 修复 1：每步重置 client.proprio → 每次查询发 fresh env 真值
                # （client.proprio 旧链路冻结在首帧；reset 后 _format_query 从
                #   obs['robo_pos'/'robo_ori'] 重新初始化，chunk 缓存期间不触发查询无副作用）
                client.proprio = None
                action = client.step(obs, lang)
                frames.append(_flip_agentview(np.asarray(obs["agentview_image"])).copy())
                obs, _, done, info = env.step(action)
                step += 1
                frames.append(_flip_agentview(np.asarray(obs["agentview_image"])).copy())
                if done:
                    success = 1.0 if env.check_success() else 0.0
                    break
            if not done:
                success = 1.0 if env.check_success() else 0.0
            video_path = os.path.join(video_dir, f"task{tid:02d}_ep{ep:02d}_success{int(success)}.mp4")
            if os.path.exists(video_path):
                raise RuntimeError(f"视频已存在，拒绝覆盖: {video_path}")
            try:
                imageio.mimsave(video_path, frames, fps=30)
            except Exception as e:
                raise RuntimeError(f"视频编码失败: {video_path}: {e}") from e
            # 编码完成后立即实际解码整段视频，而非仅检查文件是否存在。
            decoded_frames = 0
            try:
                reader = imageio.get_reader(video_path)
                try:
                    for frame in reader:
                        if frame.shape[:2] != frames[0].shape[:2]:
                            raise ValueError(f"帧尺寸不匹配: {frame.shape[:2]} != {frames[0].shape[:2]}")
                        decoded_frames += 1
                finally:
                    reader.close()
                if decoded_frames != len(frames):
                    raise ValueError(f"帧数不匹配: {decoded_frames} != {len(frames)}")
                if len(frames) < 2 or np.array_equal(frames[0], frames[-1]):
                    raise ValueError(f"视频帧为空或静态: {video_path}")
            except Exception as e:
                raise RuntimeError(f"视频解码核验失败: {video_path}: {e}") from e
            ep_succ.append(success)
            episode_metadata.append({"episode": ep, "steps": step, "success": bool(success),
                                    "video_path": video_path,
                                    "video_verified": True,
                                    "video_decoded_frames": decoded_frames})
            all_successes.append(success)
            done_eps += 1
            try: env.close()
            except: pass
            log(f"  [task{tid} ep{ep}] steps={step} success={success} | 全局进度 {done_eps}/{total_eps}")
        sr_task = sum(ep_succ) / len(ep_succ) if ep_succ else 0.0
        task_results.append({"task_id": tid, "language": task_lang, "success_rate": sr_task, "episodes": ep_succ,
                             "episode_metadata": episode_metadata})
        log(f"<<< [task {tid+1}] SR={sr_task*100:.1f}% ({int(sum(ep_succ))}/{len(ep_succ)}) | 耗时 {time.time()-task_t0:.1f}s")
        # 每 task 完后即时存中间结果（防崩溃丢数据）
        json.dump({"model": "pi0.5_lerobot_pi05-libero", "suite": args.task_suite,
                   "act_type": args.act_type, "init_seed": args.init_seed,
                   "num_episodes": args.num_episodes, "eval_horizon": args.eval_horizon,
                   "video_dir": video_dir,
                   "tasks_done": tid + 1, "task_results": task_results,
                   "current_overall_sr": sum(all_successes)/len(all_successes) if all_successes else 0.0},
                  open(os.path.join(args.output_dir, "spatial_results_partial.json"), "w"), indent=2)

    overall_sr = sum(all_successes) / len(all_successes) if all_successes else 0.0
    log(f"\n=== 完成 ===")
    log(f"总成功率: {overall_sr*100:.1f}% ({int(sum(all_successes))}/{len(all_successes)})")
    log(f"总耗时: {time.time()-t0:.1f}s")
    final = {
        "model": "pi0.5_lerobot_pi05-libero",
        "suite": args.task_suite,
        "checkpoint": args.ckpt,
        "device": args.device,
        "dtype": args.dtype,
        "run_dir": args.output_dir,
        "act_type": args.act_type,
        "init_seed": args.init_seed,
        "num_episodes_per_task": args.num_episodes,
        "eval_horizon": args.eval_horizon,
        "video_dir": video_dir,
        "total_episodes": total_eps,
        "overall_success_rate": overall_sr,
        "task_results": task_results,
        "official_baseline": {"openpi_30k": 0.9685, "lerobot_repro": 0.975, "openpi_spatial_only": 0.988},
        "total_time_s": time.time() - t0,
    }
    json.dump(final, open(os.path.join(args.output_dir, "spatial_results.json"), "w"), indent=2)
    log(f"最终结果保存: {os.path.join(args.output_dir, 'spatial_results.json')}")


if __name__ == "__main__":
    main()
