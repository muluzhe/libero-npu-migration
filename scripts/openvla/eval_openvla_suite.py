"""OpenVLA NPU 任意 LIBERO suite 完整验证（原 eval_spatial_full.py 泛化版）

支持 4 个 suite（对齐 OpenVLA 论文协议，官方 run_libero_eval.py:173-182）：
- libero_spatial: max_steps=220, 官方基准 84.7 ± 0.9%
- libero_object:  max_steps=280, 官方基准 88.4 ± 0.8%
- libero_goal:    max_steps=300, 官方基准 79.2 ± 1.0%
- libero_10:      max_steps=520, 官方基准 53.7 ± 1.3%

协议说明：官方 50 trial/task × 3 seed（A100）；我们默认 10 ep/task × 单 seed 42
（与 X-VLA 验证的 10ep 约定一致，平衡统计意义与耗时）。

关键修复（阶段12-13定位，详见 docs/openvla/OPENVLA_HANDOVER.md）：
- OpenVLA 输出 delta action（end-effector deltas）
- client 走 act_type="rel" 路径，env 保持默认 use_delta=True
- monkey-patch steps=1 让 client 每步都推理（OpenVLA 每步预测，不吃 chunk 缓存）

用法：
  python eval_openvla_suite.py --suite libero_spatial --episodes 10 --port 8011
"""
import os, sys, time, json, argparse, traceback
import numpy as np
import imageio
from pathlib import Path

# 渲染环境（必须在 import libero/mujoco 前）
os.environ.update(NUMBA_DISABLE_JIT='1', MUJOCO_GL='osmesa', LIBGL_ALWAYS_SOFTWARE='1',
                   MESA_LOADER_DRIVER_OVERRIDE='swrast', PYOPENGL_PLATFORM='osmesa')
os.environ['LIBGL_DRIVERS_PATH'] = os.path.expanduser('~/render_libs/dri')
os.environ['LD_LIBRARY_PATH'] = os.path.expanduser('~/render_libs') + ':/usr/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# X-VLA client 路径（官方原样零改动，含 LiberoAbsActionProcessor 同源转换链）
X_VLA_ROOT = os.environ.get('X_VLA_ROOT', os.path.expanduser('~/work/X-VLA'))
sys.path.insert(0, os.path.join(X_VLA_ROOT, 'evaluation', 'libero'))
from libero_client import LIBEROEval, ClientModel, _flip_agentview

# 各 suite 官方协议（max_steps 来自官方 run_libero_eval.py，基准来自 OpenVLA 论文 Table 2）
class VideoError(RuntimeError):
    pass


SUITE_CONFIG = {
    'libero_spatial': dict(max_steps=220, official=0.847, official_str='84.7 ± 0.9%'),
    'libero_object':  dict(max_steps=280, official=0.884, official_str='88.4 ± 0.8%'),
    'libero_goal':    dict(max_steps=300, official=0.792, official_str='79.2 ± 1.0%'),
    'libero_10':      dict(max_steps=520, official=0.537, official_str='53.7 ± 1.3%'),
}


def parse_args():
    p = argparse.ArgumentParser(description='OpenVLA NPU LIBERO suite 验证')
    p.add_argument('--suite', required=True, choices=list(SUITE_CONFIG.keys()))
    p.add_argument('--episodes', type=int, default=10, help='每 task episode 数（官方 50）')
    p.add_argument('--port', type=int, default=8011, help='推理 server 端口')
    p.add_argument('--server', default='127.0.0.1')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--act_type', default='rel', help='rel=delta action 路径（OpenVLA 正确语义）')
    p.add_argument('--out_dir', default=None, help='默认 $PROJECT_ROOT/results/openvla_full/')
    p.add_argument('--video_dir', default=None,
                   help='每个 rollout 保存实际 NPU 闭环视频的目录；默认 $OUT_DIR/videos/<suite>')
    p.add_argument('--resume', action='store_true',
                   help='跳过 results JSON 中已完成的 task（断点续跑）。'
                        '安全性：_init_env 每 episode 独立重建 env 且 seed=init_seed+ep+100，'
                        '与 task 顺序无关，续跑结果与全新跑完全等价')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SUITE_CONFIG[args.suite]
    EVAL_HORIZON = cfg['max_steps']

    PROJECT_ROOT = os.environ.get('PROJECT_ROOT', str(Path(__file__).resolve().parents[2]))
    OUT_DIR = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT) / 'results' / 'openvla_full'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON = OUT_DIR / f'{args.suite}_results.json'
    LOG_FILE = OUT_DIR / f'{args.suite}.log'
    VIDEO_DIR = Path(args.video_dir) if args.video_dir else OUT_DIR / 'videos' / args.suite
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(LOG_FILE, 'a') as f:
            f.write(line + "\n")

    log(f"=== OpenVLA NPU {args.suite} 完整验证 ===")
    log(f"server: http://{args.server}:{args.port}/act")
    log(f"配置: {args.episodes}ep/task, horizon={EVAL_HORIZON} (官方), act_type={args.act_type}, seed={args.seed}")
    log(f"官方基准: {cfg['official_str']} (A100, 50 trial/task × 3 seed)")

    # === 断点续跑：载入已完成 task（仅当配置一致时才复用） ===
    prev_results = {}
    if args.resume and RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON) as f:
                prev = json.load(f)
            prev_cfg = prev.get('config', {})
            if (prev.get('suite') == args.suite
                    and prev_cfg.get('num_ep_per_task') == args.episodes
                    and prev_cfg.get('eval_horizon') == EVAL_HORIZON
                    and prev_cfg.get('seed') == args.seed):
                prev_results = {int(k): v for k, v in prev.get('tasks', {}).items()}
                done = [k for k, v in prev_results.items()
                        if len(v.get('successes', [])) == args.episodes]
                missing_video = [k for k in done if len(v.get('episodes', [])) != args.episodes
                                 or any(not ep.get('video_path') for ep in v.get('episodes', []))]
                if missing_video:
                    raise VideoError(
                        f"resume 结果 task{missing_video} 缺少视频 metadata；旧无视频结果不会补齐或覆盖，请使用新 out_dir")
                log(f"resume: 载入已有结果，{len(done)} 个 task 已完成将跳过: {sorted(done)}")
            else:
                log("resume: 已有结果配置不一致，忽略并从头跑")
        except VideoError:
            raise
        except Exception as e:
            log(f"resume: 载入已有结果失败（{e}），从头跑")

    policy = ClientModel(host=args.server, port=args.port)

    # === monkey-patch: 强制 steps=1 让 client 每步都推理 ===
    _orig_fmt = policy._format_query
    def _fmt_steps1(obs, goal):
        p = _orig_fmt(obs, goal)
        p["steps"] = 1
        return p
    policy._format_query = _fmt_steps1

    evaluator = LIBEROEval(
        task_suite_name=args.suite,
        eval_horizon=EVAL_HORIZON,
        num_episodes=args.episodes,
        init_seed=args.seed,
        act_type=args.act_type,
    )

    task_suite = evaluator.task_suite_list[0]
    num_tasks = len(task_suite.tasks)
    log(f"{args.suite}: {num_tasks} tasks × {args.episodes} ep = {num_tasks * args.episodes} rollouts total")

    results = {}
    t_start = time.time()
    total_rollouts = 0
    total_successes = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_lang = task.language

        # resume: 已完成的 task 直接复用结果（seeding 与顺序无关，统计等价于全新跑）
        if task_id in prev_results and len(prev_results[task_id].get('successes', [])) == args.episodes:
            entry = prev_results[task_id]
            results[task_id] = entry
            total_rollouts += args.episodes
            total_successes += int(sum(entry['successes']))
            log(f"--- task{task_id}: {task_lang} --- [resume 跳过，SR={entry['task_success_rate']:.0%}]")
            continue

        task_succs = []
        episode_metadata = []
        log(f"--- task{task_id}: {task_lang} ---")

        for ep in range(args.episodes):
            ep_t0 = time.time()
            env = None
            try:
                env, lang, obs = evaluator._init_env(task_suite, task_id=task_id, ep=ep)
                # 前 10 步 dummy action 让物体稳定（官方 num_steps_wait=10）
                dummy = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                for _ in range(10):
                    obs, _, _, _ = env.step(dummy)
                policy.reset()
                done_flag = False
                steps = 0
                frames = []
                for h in range(EVAL_HORIZON):
                    robo_ori = evaluator.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
                    robo_pos = env.env.robots[0].controller.ee_pos
                    obs['robo_pos'] = robo_pos
                    obs['robo_ori'] = robo_ori
                    action = policy.step(obs, lang)
                    frames.append(_flip_agentview(np.asarray(obs['agentview_image'])).copy())
                    obs, reward, done, info = env.step(action)
                    steps += 1
                    frames.append(_flip_agentview(np.asarray(obs['agentview_image'])).copy())
                    if done:
                        done_flag = True
                        break
                video_path = VIDEO_DIR / f'task{task_id:02d}_ep{ep:02d}_success{int(done_flag)}.mp4'
                if video_path.exists():
                    raise VideoError(f"视频已存在，拒绝覆盖: {video_path}")
                try:
                    imageio.mimsave(video_path, frames, fps=30)
                except Exception as e:
                    raise VideoError(f"视频编码失败: {video_path}: {e}") from e
                task_succs.append(1.0 if done_flag else 0.0)
                episode_metadata.append({'episode': ep, 'steps': steps, 'success': bool(done_flag),
                                        'video_path': str(video_path)})
                total_rollouts += 1
                total_successes += int(done_flag)
                log(f"  task{task_id} ep{ep}: success={int(done_flag)}, steps={steps}, 耗时={(time.time() - ep_t0) / 60:.1f}min")
            except VideoError:
                raise
            except Exception as e:
                log(f"  task{task_id} ep{ep}: EXCEPTION {e}")
                log(traceback.format_exc()[-300:])
                task_succs.append(0.0)
                total_rollouts += 1
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

        task_sr = sum(task_succs) / max(len(task_succs), 1)
        results[task_id] = {
            'language': task_lang,
            'successes': task_succs,
            'task_success_rate': task_sr,
            'episodes': episode_metadata,
        }
        log(f"  >>> task{task_id} SR = {task_sr:.0%} ({sum(task_succs)}/{len(task_succs)})")

        # 增量保存（防止中断丢失）
        with open(RESULTS_JSON, 'w') as f:
            json.dump({
                'suite': args.suite,
                'config': {
                    'num_ep_per_task': args.episodes,
                    'eval_horizon': EVAL_HORIZON,
                    'act_type': args.act_type,
                    'seed': args.seed,
                    'video_dir': str(VIDEO_DIR),
                },
                'running_success_rate': total_successes / max(total_rollouts, 1),
                'total_rollouts': total_rollouts,
                'total_successes': total_successes,
                'tasks': results,
            }, f, indent=2)

    overall_sr = total_successes / max(total_rollouts, 1)
    log(f"=== 汇总 ===")
    log(f"总 rollouts: {total_rollouts}, 总 successes: {total_successes}")
    log(f"Overall SR: {overall_sr:.1%} vs 官方 {cfg['official_str']}")
    log(f"总耗时: {(time.time() - t_start) / 60:.1f} 分钟")

    final = {
        'suite': args.suite,
        'config': {
            'num_ep_per_task': args.episodes,
            'eval_horizon': EVAL_HORIZON,
            'act_type': args.act_type,
            'seed': args.seed,
            'video_dir': str(VIDEO_DIR),
        },
        'overall_success_rate': overall_sr,
        'total_rollouts': total_rollouts,
        'total_successes': total_successes,
        'total_time_min': (time.time() - t_start) / 60,
        'official_baseline': cfg['official'],
        'tasks': results,
    }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(final, f, indent=2)
    log(f"结果已保存: {RESULTS_JSON}")


if __name__ == '__main__':
    main()
