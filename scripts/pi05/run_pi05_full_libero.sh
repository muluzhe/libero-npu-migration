#!/usr/bin/env bash
# PI0.5 LIBERO 四 suite NPU 串行编排器。
# 每个 suite 独立 run 目录、server/eval 日志和视频；不修改已有结果。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
EVAL="$SCRIPT_DIR/eval_pi05_spatial.py"
SERVER="$PROJECT_ROOT/models/pi0/server_v2.py"

# 不 source 整个 ~/.bashrc：ModelArts/交互初始化可能在非交互 shell 中提前退出。
if [[ -n "${PI05_PYTHON:-}" ]]; then
  PYTHON=("$PI05_PYTHON")
else
  CONDA_SH=${CONDA_SH:-"$HOME/anaconda3/etc/profile.d/conda.sh"}
  if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate "${PI05_CONDA_ENV:-lerobot-pi05}"
  fi
  PYTHON=(python)
fi
source "${ASCEND_ENV_SH:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" 2>/dev/null || true

# 统一评测默认参数（seed 42、trials/task 50 官方全量、horizon 220/280/300/520、强制视频）
# 详见 scripts/eval_defaults.sh 与 docs/EVAL_PROTOCOL.md；缩减验证用 PI05_EPISODES=10 且结果须标注协议
source "$SCRIPT_DIR/../eval_defaults.sh"
CKPT=${PI05_CKPT:-"$HOME/work/lerobot_pi05_libero_official"}
RUNS_PARENT=${PI05_FULL_OUT_ROOT:-"$PROJECT_ROOT/results/pi05/full_libero"}
RUN_ROOT="$RUNS_PARENT/$(date +%Y%m%d_%H%M%S)_$$"
DEVICE=${PI05_DEVICE:-npu:0}
VISIBLE_DEVICE=${NPU_VISIBLE_DEVICE:-0}
PORT_BASE=${PI05_FULL_PORT_BASE:-8112}
EPISODES=${PI05_EPISODES:-$LIBERO_TRIALS_PER_TASK}
SEED=${PI05_SEED:-$LIBERO_SEED}

export NPU_VISIBLE_DEVICE="$VISIBLE_DEVICE"
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICE"
export LEROBOT_PI05_ROOT=${LEROBOT_PI05_ROOT:-"$HOME/work/lerobot_pi05"}
export X_VLA_ROOT=${X_VLA_ROOT:-"$HOME/work/X-VLA"}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NUMBA_DISABLE_JIT=1 MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast PYOPENGL_PLATFORM=osmesa
export LIBGL_DRIVERS_PATH="$HOME/render_libs/dri"
export LD_LIBRARY_PATH="$HOME/render_libs:/usr/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"

[[ -f "$EVAL" ]] || { echo "missing evaluator: $EVAL" >&2; exit 2; }
[[ -f "$SERVER" ]] || { echo "missing server: $SERVER" >&2; exit 2; }
[[ -d "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "model": "pi0.5",
  "checkpoint": "$CKPT",
  "device": "$DEVICE",
  "npu_visible_device": "$VISIBLE_DEVICE",
  "dtype": "float32",
  "seed": $SEED,
  "episodes_per_task": $EPISODES,
  "video": "forced",
  "suites": [
    {"name": "libero_spatial", "horizon": $LIBERO_HORIZON_SPATIAL},
    {"name": "libero_object", "horizon": $LIBERO_HORIZON_OBJECT},
    {"name": "libero_goal", "horizon": $LIBERO_HORIZON_GOAL},
    {"name": "libero_10", "horizon": $LIBERO_HORIZON_10}
  ]
}
EOF

# server 生命周期由 eval_pi05_spatial.py 内建管理（--server_cmd 启动子进程、
# 600s 就绪等待、atexit 清理、server.log 落盘），编排器只负责四 suite 串行调度。
suites=(libero_spatial libero_object libero_goal libero_10)
horizons=("$LIBERO_HORIZON_SPATIAL" "$LIBERO_HORIZON_OBJECT" "$LIBERO_HORIZON_GOAL" "$LIBERO_HORIZON_10")  # 与 eval_defaults.sh 一致
echo "[full] run root: $RUN_ROOT"
echo "[full] checkpoint: $CKPT"
echo "[full] device: $DEVICE (NPU_VISIBLE_DEVICE=$VISIBLE_DEVICE)"

for i in "${!suites[@]}"; do
  suite=${suites[$i]}
  horizon=${horizons[$i]}
  port=$((PORT_BASE + i))
  suite_root="$RUN_ROOT/$suite"
  mkdir -p "$suite_root"
  eval_log="$suite_root/eval.log"
  printf '%s\n' "$port" > "$suite_root/server.port"

  # eval 脚本以此为子进程命令自行启动/等待/停止 server（shlex 解析，路径勿含空格）
  server_cmd="${PYTHON[*]} $SERVER --model_path $CKPT --port $port --device $DEVICE --no-bf16"
  echo "[full] starting $suite on port $port (horizon=$horizon)"

  set +e
  "${PYTHON[@]}" "$EVAL" \
    --server_ip 127.0.0.1 --server_port "$port" \
    --task_suite "$suite" --num_tasks 10 --num_episodes "$EPISODES" \
    --eval_horizon "$horizon" --init_seed "$SEED" --act_type rel \
    --output_dir "$suite_root" \
    --server_cmd "$server_cmd" --ckpt "$CKPT" --device "$DEVICE" --dtype float32 \
    >"$eval_log" 2>&1
  eval_status=$?
  set -e

  if (( eval_status != 0 )); then
    echo "[full] $suite failed (status=$eval_status); see $eval_log" >&2
    exit "$eval_status"
  fi
  echo "[full] completed $suite"
done

echo "[full] completed; manifest and per-suite results are under $RUN_ROOT"
