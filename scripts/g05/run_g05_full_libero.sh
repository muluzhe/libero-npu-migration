#!/usr/bin/env bash
# G0.5 LIBERO 四 suite NPU 串行编排器（对齐官方协议：50 trials/task，horizon 220/280/300/520）
# server（GalaxeaVLA serve_policy_batched.py，NPU fp32）全 suite 复用；客户端串行跑 4 suite。
# 结果与视频输出到 results/g05/full_libero/<run>/<suite>/。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$SCRIPT_DIR/../eval_defaults.sh"
G05_ROOT=${G05_ROOT:-"$HOME/work/GalaxeaVLA"}
CKPT=${G05_CKPT:-"$G05_ROOT/checkpoints/g05-libero/model.pt"}
PYTHON=${G05_PYTHON:-"$HOME/anaconda3/envs/g05/bin/python"}
PORT=${G05_PORT:-8765}
TRIALS=${G05_TRIALS:-$LIBERO_TRIALS_PER_TASK}
PARALLEL=${G05_PARALLEL:-$LIBERO_NUM_PARALLEL}
SEED=${G05_SEED:-$LIBERO_SEED}
STEPS_WAIT=${G05_STEPS_WAIT:-$LIBERO_STEPS_WAIT}
ENV_RESOLUTION=${G05_ENV_RESOLUTION:-$LIBERO_ENV_RESOLUTION}
VISIBLE=${G05_VISIBLE_DEVICE:-0}
RUNS_PARENT=${G05_FULL_OUT_ROOT:-"$PROJECT_ROOT/results/g05/full_libero"}
RUN_ROOT="$RUNS_PARENT/$(date +%Y%m%d_%H%M%S)_$$"

[[ -f "$CKPT" ]] || { echo "missing ckpt: $CKPT" >&2; exit 2; }
[[ -f "$G05_ROOT/scripts/serve_policy_batched.py" ]] || { echo "missing server script" >&2; exit 2; }

source "${ASCEND_ENV_SH:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" 2>/dev/null || true
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE" NPU_VISIBLE_DEVICE="$VISIBLE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=swrast PYOPENGL_PLATFORM=osmesa
export LIBGL_DRIVERS_PATH="$HOME/render_libs/dri"
export LD_LIBRARY_PATH="$HOME/render_libs:/usr/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

# hydra config 内的相对路径（checkpoints/...）按 cwd 解析，必须以 GalaxeaVLA 为工作目录
cd "$G05_ROOT"

mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "model": "g05",
  "checkpoint": "$CKPT",
  "device": "npu:0",
  "dtype": "float32",
  "seed": $SEED,
  "trials_per_task": $TRIALS,
  "num_parallel_envs": $PARALLEL,
  "action_steps": 10,
  "num_steps_wait": $STEPS_WAIT,
  "env_resolution": $ENV_RESOLUTION,
  "video": "forced (--save_videos)",
  "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
  "horizons": {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520},
  "official_baseline_libero_avg": 0.989
}
EOF

echo "[g05] run root: $RUN_ROOT"
echo "[g05] ckpt: $CKPT | trials/task: $TRIALS | parallel: $PARALLEL | seed: $SEED"

# ── server（全 suite 复用）──
"$PYTHON" "$G05_ROOT/scripts/serve_policy_batched.py" \
  --ckpt_path "$CKPT" --host 0.0.0.0 --port "$PORT" \
  eval_embodiment=libero --action_steps 10 --device npu:0 --no-bf16 \
  --max_batch_size "$PARALLEL" --max_wait_ms 500 \
  > "$RUN_ROOT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$RUN_ROOT/server.pid"
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT INT TERM

# 等就绪：端口监听或 600s 超时
ready=0
for _ in $(seq 1 300); do
  if ss -tln 2>/dev/null | grep -q ":$PORT "; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[g05] server died during startup; see $RUN_ROOT/server.log" >&2; exit 1
  fi
  sleep 2
done
if (( ready != 1 )); then echo "[g05] server not ready in 600s" >&2; exit 1; fi
echo "[g05] server ready on port $PORT (pid $SERVER_PID)"

# ── 四 suite 串行 ──
suites=(libero_spatial libero_object libero_goal libero_10)
for suite in "${suites[@]}"; do
  suite_root="$RUN_ROOT/$suite"
  mkdir -p "$suite_root"
  echo "[g05] starting $suite ..."
  set +e
  "$PYTHON" "$G05_ROOT/experiments/libero/eval_libero_parallel.py" \
    --server_uri "ws://127.0.0.1:$PORT" \
    --task_suite_name "$suite" \
    --num_trials_per_task "$TRIALS" \
    --num_parallel "$PARALLEL" \
    --num_steps_wait "$STEPS_WAIT" \
    --env_resolution "$ENV_RESOLUTION" \
    --seed "$SEED" \
    --output_dir "$suite_root" \
    --save_videos \
    > "$suite_root/client.log" 2>&1
  status=$?
  set -e
  if (( status != 0 )); then
    echo "[g05] $suite FAILED (status=$status); see $suite_root/client.log" >&2
    exit "$status"
  fi
  echo "[g05] completed $suite"
done

echo "[g05] ALL DONE; results under $RUN_ROOT"
