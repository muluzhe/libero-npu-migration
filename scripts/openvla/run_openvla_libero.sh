#!/bin/bash
# ⚠️ DEPRECATED（2026-09-26）：路径B（官方 run_libero_eval.py）因 prismatic→dlimp→tensorflow
# import 链在 OSMesa+NPU 环境 segfault 已弃用（详见 docs/openvla/OPENVLA_HANDOVER.md）。
# 正确路径：bash scripts/openvla/run_openvla_full_validation.sh <npu_device> <port> <suite>...
#
# OpenVLA LIBERO NPU 验证（路径B：官官eval脚本 + NPU适配）
# 用法: bash run_openvla_libero.sh <suite> <num_episodes> <seed>
# suite: libero_spatial | libero_object | libero_goal | libero_10
set -e

SUITE=${1:-libero_spatial}
EP=${2:-10}
SEED=${3:-42}
PY=python
CKPT_BASE=${CKPT_BASE:?usage: CKPT_BASE env var required}
PROJECT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}

# checkpoint 映射（每suite一个独立微调权重）
declare -A CKPT_MAP=(
  [libero_spatial]=libero-spatial
  [libero_object]=libero-object
  [libero_goal]=libero-goal
  [libero_10]=libero-10
)
CKPT=$CKPT_BASE/${CKPT_MAP[$SUITE]}
[ -d "$CKPT" ] || { echo "❌ checkpoint $CKPT 不存在，先下载"; exit 1; }

# OSMesa 渲染环境（复用X-VLA那套）
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=swrast
export PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0

# 确认渲染库在
[ -f "$HOME/render_libs/libOSMesa.so.8.0.0" ] || { echo "❌ 渲染库缺，先重建"; exit 1; }

# 确认robosuite patch在（openvla用OffScreenRenderEnv走robosuite）
grep -q "np.flip(img, 0).copy()" $(python -c "import robosuite,os;print(os.path.dirname(robosuite.__file__))")/utils/binding_utils.py 2>/dev/null || {
  echo "⚠️ read_pixels patch缺，重新应用..."
  $PY $PROJECT/scripts/apply_patches.py 2>&1 | grep -vE "owner|Warning"
}

echo "=== OpenVLA LIBERO NPU 验证 ==="
echo "suite: $SUITE | episodes: $EP | seed: $SEED"
echo "checkpoint: $CKPT"
echo "输出: $PROJECT/results/openvla_npu_${SUITE}_seed${SEED}_${EP}ep/"

cd ${OPENVLA_ROOT:?usage: OPENVLA_ROOT env var required}
export PYTHONPATH=${OPENVLA_ROOT:?usage: OPENVLA_ROOT env var required}:$PYTHONPATH
$PY experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "$CKPT" \
  --task_suite_name "$SUITE" \
  --center_crop True \
  --num_trials_per_task "$EP" \
  --seed "$SEED" \
  --use_wandb False \
  2>&1 | tee $PROJECT/results/openvla_npu_${SUITE}_seed${SEED}_${EP}ep.log
