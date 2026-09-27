#!/usr/bin/env bash
# PI0.5 LIBERO 完整 spatial suite 一键验证脚本（阶段5；P4.29 修复版）
# - 启动 pi0.5 NPU 推理 server（后台，fp32：F4 验证结论 bf16 有精度风险）
# - 启动 eval_pi05_spatial.py 跑 10 task × 10 ep（P4.29 修复：fresh proprio + env 真值注入）
# - 不影响其他模型文件 / env / 结果
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(dirname "$PROJECT_ROOT")"

CONDA_SH=${CONDA_SH:-"$HOME/anaconda3/etc/profile.d/conda.sh"}
source "$CONDA_SH"
conda activate lerobot-pi05

# CANN/驱动环境（后台 NPU 任务必须显式 source，否则 torch.npu.is_available()=False，教训#33）
source "${ASCEND_ENV_SH:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" 2>/dev/null || true
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}

export ASCEND_RT_VISIBLE_DEVICES=${NPU_DEVICE:-0} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LEROBOT_PI05_ROOT=${LEROBOT_PI05_ROOT:-"$HOME/work/lerobot_pi05"} X_VLA_ROOT=${X_VLA_ROOT:-"$HOME/work/X-VLA"}
export NUMBA_DISABLE_JIT=1 MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=swrast PYOPENGL_PLATFORM=osmesa
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:${LD_LIBRARY_PATH:-}

CKPT=${PI05_CKPT:-"$HOME/work/lerobot_pi05_libero_official"}
PORT=${PI05_PORT:-8012}

# === 启动 pi0.5 server（若未监听）===
if ! ss -tlnp 2>/dev/null | grep -q ":$PORT"; then
  echo "[run] 启动 pi0.5 NPU server（fp32, ckpt=$CKPT）..."
  rm -f /tmp/pi05_srv.log
  nohup setsid python "$PROJECT_ROOT/models/pi0/server_v2.py" \
    --model_path "$CKPT" \
    --port "$PORT" --device npu:0 > /tmp/pi05_srv.log 2>&1 &
  echo "[run] server pid=$!"
  disown 2>/dev/null
  for i in $(seq 1 36); do
    if ss -tlnp 2>/dev/null | grep -q ":$PORT"; then echo "[run] server 就绪"; break; fi
    sleep 10
  done
else
  echo "[run] pi0.5 server 已在监听，复用"
fi

# === 启动完整 spatial suite 验证（后台）===
mkdir -p "$PROJECT_ROOT/results/pi05_spatial"
rm -f /tmp/pi05_spatial_eval.log
nohup setsid python "$PROJECT_ROOT/scripts/pi05/eval_pi05_spatial.py" \
  --server_ip 127.0.0.1 --server_port "$PORT" \
  --task_suite libero_spatial --num_tasks 10 --num_episodes 10 \
  --eval_horizon 220 --init_seed 42 --act_type rel \
  --output_dir "$PROJECT_ROOT/results/pi05_spatial" \
  --progress_log "$PROJECT_ROOT/results/pi05_spatial/progress.log" \
  > /tmp/pi05_spatial_eval.log 2>&1 &
EVAL_PID=$!
echo "[run] eval pid=$EVAL_PID"
disown 2>/dev/null
echo "[run] 进度日志: $PROJECT_ROOT/results/pi05_spatial/progress.log"
echo "[run] eval stdout: /tmp/pi05_spatial_eval.log"
echo "[run] server log: /tmp/pi05_srv.log"
echo "[run] 后台跑中，用以下命令看进度：tail -f $PROJECT_ROOT/results/pi05_spatial/progress.log"
