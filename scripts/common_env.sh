#!/usr/bin/env bash
# LIBERO NPU 通用运行环境
# 所有模型验证脚本可 source 本文件，再设置模型专用变量。
set -u

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export X_VLA_ROOT="${X_VLA_ROOT:-$HOME/work/X-VLA}"
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast
export PYOPENGL_PLATFORM=osmesa
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-$HOME/render_libs/dri}"
export LD_LIBRARY_PATH="${HOME}/render_libs:/usr/lib64:${LD_LIBRARY_PATH:-}"

# 后台 NPU 任务需要 CANN/驱动路径；已加载时保持现有环境。
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
fi
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-unset}"
  echo "MUJOCO_GL=$MUJOCO_GL"
fi
