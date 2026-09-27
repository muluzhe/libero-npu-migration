#!/usr/bin/env bash
# PI0.5 LIBERO 四 suite NPU 编排器。
# 每个 suite 独立 run 目录/日志/视频；不清理历史结果，不复用活跃 spatial 端口。
set -euo pipefail

source ~/.bashrc 2>/dev/null || true
CONDA_SH=${CONDA_SH:-"$HOME/anaconda3/etc/profile.d/conda.sh"}
source "$CONDA_SH"
conda activate lerobot-pi05
source "${ASCEND_ENV_SH:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" 2>/dev/null || true

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL=${PROJECT_ROOT}/scripts/pi05/eval_pi05_spatial.py
CKPT=${PI05_CKPT:-"$HOME/work/lerobot_pi05_libero_official"}
OUT_ROOT=${PI05_FULL_OUT_ROOT:-${PROJECT_ROOT}/results/pi05/full_libero}
DEVICE=${PI05_DEVICE:-npu:0}
PORT_BASE=${PI05_FULL_PORT_BASE:-8112}
EPISODES=${PI05_EPISODES:-10}
SEED=${PI05_SEED:-42}
DTYPE=${PI05_DTYPE:-float32}

export ASCEND_RT_VISIBLE_DEVICES=${PI05_VISIBLE_DEVICE:-0}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NUMBA_DISABLE_JIT=1 MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast PYOPENGL_PLATFORM=osmesa
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}
export X_VLA_ROOT=${X_VLA_ROOT:-"$HOME/work/X-VLA"}

[[ -f "$EVAL" ]] || { echo "missing evaluator: $EVAL" >&2; exit 2; }
[[ -d "$CKPT" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 2; }
mkdir -p "$OUT_ROOT"
RUN_ROOT="$OUT_ROOT/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_ROOT"
cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "model": "pi0.5",
  "checkpoint": "${CKPT}",
  "device": "${DEVICE}",
  "visible_device": "${ASCEND_RT_VISIBLE_DEVICES}",
  "dtype": "${DTYPE}",
  "seed": ${SEED},
  "episodes_per_task": ${EPISODES},
  "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
  "protocol_status": "local_reproduction: 10 episodes per task; official LIBERO comparisons commonly use 50 trials per task and multiple seeds",
  "reference_status": "paper_pi05_does_not_directly_report_LIBERO; openpi_and_lerobot_followup_references_are_separate"
}
EOF

echo "[full] run root: $RUN_ROOT"
echo "[full] checkpoint: $CKPT"
echo "[full] device: $DEVICE (ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES)"

suites=(libero_spatial libero_object libero_goal libero_10)
for i in "${!suites[@]}"; do
  suite=${suites[$i]}
  port=$((PORT_BASE + i))
  case "$suite" in
    libero_spatial) horizon=220 ;;
    libero_object) horizon=280 ;;
    libero_goal) horizon=300 ;;
    libero_10) horizon=520 ;;
  esac
  echo "[full] starting $suite on port $port (horizon=$horizon)"
  python "$EVAL" \
    --server_ip 127.0.0.1 --server_port "$port" \
    --task_suite "$suite" --num_episodes "$EPISODES" \
    --eval_horizon "$horizon" --init_seed "$SEED" --act_type rel \
    --output_dir "$RUN_ROOT/$suite" \
    --ckpt "$CKPT" --device "$DEVICE" --dtype "$DTYPE" \
    --server_cmd "${PI05_PYTHON:-python} ${PROJECT_ROOT}/models/pi0/server_v2.py --model_path ${CKPT} --port ${port} --device ${DEVICE}"
done

echo "[full] completed; manifest and per-suite results are under $RUN_ROOT"
