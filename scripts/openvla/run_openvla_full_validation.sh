#!/bin/bash
# OpenVLA NPU 全量验证编排器：按顺序跑多个 suite，每个 suite 自动起/停对应推理 server
#
# 用法: bash run_openvla_full_validation.sh <npu_visible_device> <port> <suite> [<suite> ...]
#   <npu_visible_device>  ASCEND_RT_VISIBLE_DEVICES 值（0 或 1，指定用哪张 910B4）
#   <port>                推理 server 端口（两条并行流用不同端口，如 8011 / 8021）
#   <suite>               libero_spatial / libero_object / libero_goal / libero_10
#
# 例（双卡并行，两终端各跑一条流）:
#   bash run_openvla_full_validation.sh 0 8011 libero_spatial libero_goal
#   bash run_openvla_full_validation.sh 1 8021 libero_object libero_10
#
# 环境变量:
#   EPISODES      每 task episode 数（默认统一协议 50；可显式缩减并在结果中标注）
#   VIDEO_DIR     视频输出根目录（默认 $PROJECT_ROOT/results/openvla_full/videos/<suite>）
#   OPENVLA_ROOT  openvla 仓库路径（默认 ~/work/openvla，server 需其 prismatic 模块）
#   OPENVLA_CKPTS checkpoint 根目录（默认 ~/work/openvla_checkpoints，子目录 libero-spatial 等）
#
# checkpoint 来源：官方 msharma11/openvla-7b-13b-libero-* 的 modelscope 镜像
#（superpeach/*，已验证 spatial 镜像与官方逐字节同大小）。
set -u

DEVICE_ID=${1:?usage: run_openvla_full_validation.sh <npu_device> <port> <suite>...}
PORT=${2:?usage: run_openvla_full_validation.sh <npu_device> <port> <suite>...}
shift 2
SUITES=("$@")
[ ${#SUITES[@]} -ge 1 ] || { echo "至少需要一个 suite 参数"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# 统一评测默认参数（seed 42、trials/task 50 官方全量、官方固定 horizon、强制视频）
# 详见 scripts/eval_defaults.sh 与 docs/EVAL_PROTOCOL.md；缩减验证用 EPISODES=10 且结果须标注协议
source "$SCRIPT_DIR/../eval_defaults.sh"
EPISODES=${EPISODES:-$LIBERO_TRIALS_PER_TASK}
SEED=${SEED:-$LIBERO_SEED}
VIDEO_DIR=${VIDEO_DIR:-}
RESUME=${RESUME:-1}   # 1=断点续跑（跳过已完成的 task，seeding 与顺序无关故统计等价）；0=强制从头
RESUME_FLAG=""
[ "$RESUME" = "1" ] && RESUME_FLAG="--resume"
OPENVLA_ROOT=${OPENVLA_ROOT:-$HOME/work/openvla}
OPENVLA_CKPTS=${OPENVLA_CKPTS:-$HOME/work/openvla_checkpoints}
X_VLA_ROOT=${X_VLA_ROOT:-$HOME/work/X-VLA}
PY=$(which python)

# CANN/驱动环境（非交互 shell 不一定加载 ~/.bashrc，直接 source Ascend 官方脚本保证齐全；
# 曾出现后台任务 halGetDeviceInfo drvRet=4 / npu 不可用，根因是驱动库路径缺失，见 PI05_RECORD 教训#33）
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}

# OSMesa 渲染环境（必须在 python 启动前 export，动态链接器只读一次）
export NUMBA_DISABLE_JIT=1 MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast PYOPENGL_PLATFORM=osmesa
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=== OpenVLA NPU 全量验证编排 ==="
echo "device=$DEVICE_ID port=$PORT episodes=$EPISODES seed=$SEED suites=${SUITES[*]}"

for SUITE in "${SUITES[@]}"; do
  # libero_spatial -> libero-spatial（checkpoint 子目录命名：下划线换连字符）
  CKPT="$OPENVLA_CKPTS/${SUITE/_/-}"
  # 分片完整性校验（官方 4 分片固定大小，防下载中的残缺 checkpoint 被误用）
  CKPT_OK=1
  for pair in "model-00001-of-00004.safetensors:4925122448" \
              "model-00002-of-00004.safetensors:4947392496" \
              "model-00003-of-00004.safetensors:4947417456" \
              "model-00004-of-00004.safetensors:262668432"; do
    f="${pair%%:*}"; want="${pair##*:}"
    have=$(stat -c %s "$CKPT/$f" 2>/dev/null || echo 0)
    if [ "$have" != "$want" ]; then
      echo "[WAIT] $SUITE checkpoint 不完整: $f ($have/$want bytes)，跳过"
      CKPT_OK=0; break
    fi
  done
  [ $CKPT_OK -eq 1 ] || continue

  echo ""
  echo "=== [$SUITE] 启动推理 server（npu:$DEVICE_ID, port=$PORT, ckpt=$CKPT）==="
  pkill -f "server_v2.py --model_path $CKPT" 2>/dev/null && sleep 2
  ASCEND_RT_VISIBLE_DEVICES=$DEVICE_ID OPENVLA_ATTN=sdpa \
    X_VLA_ROOT=$X_VLA_ROOT PYTHONPATH=$OPENVLA_ROOT:$PYTHONPATH \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    nohup setsid $PY "$PROJECT_ROOT/models/openvla/server_v2.py" \
      --model_path "$CKPT" --port $PORT --unnorm_key $SUITE --device auto --bf16 \
      > /tmp/openvla_server_${SUITE}.log 2>&1 &
  SERVER_PID=$!
  echo "server PID=$SERVER_PID, log=/tmp/openvla_server_${SUITE}.log"

  # 等 server 就绪（模型加载约 1-3 分钟）
  READY=0
  for i in $(seq 1 90); do
    if curl -s -o /dev/null --connect-timeout 2 "http://127.0.0.1:$PORT/" 2>/dev/null; then
      READY=1; break
    fi
    # server 进程死掉立即退出等待
    if ! kill -0 $SERVER_PID 2>/dev/null; then
      echo "[ERROR] server 进程退出，log 尾部："; tail -20 /tmp/openvla_server_${SUITE}.log; break
    fi
    sleep 10
  done
  if [ $READY -ne 1 ]; then
    echo "[ERROR] server 15 分钟未就绪，跳过 $SUITE"; tail -20 /tmp/openvla_server_${SUITE}.log
    continue
  fi
  echo "[$SUITE] server 就绪"

  echo "=== [$SUITE] 跑完整验证（${EPISODES}ep/task, resume=$RESUME）==="
  VIDEO_ARGS=()
  [ -n "$VIDEO_DIR" ] && VIDEO_ARGS+=(--video_dir "$VIDEO_DIR/$SUITE")
  X_VLA_ROOT=$X_VLA_ROOT PROJECT_ROOT=$PROJECT_ROOT \
    $PY "$PROJECT_ROOT/scripts/openvla/eval_openvla_suite.py" \
      --suite $SUITE --episodes $EPISODES --seed $SEED --port $PORT $RESUME_FLAG "${VIDEO_ARGS[@]}"
  EVAL_RC=$?
  echo "[$SUITE] eval 退出码=$EVAL_RC"

  # 停 server 释放 HBM（下一个 suite 换 checkpoint 需重启）
  pkill -f "server_v2.py --model_path $CKPT" 2>/dev/null
  sleep 5
  echo "[$SUITE] server 已停止"
done

echo ""
echo "=== 编排完成，结果在 $PROJECT_ROOT/results/openvla_full/ ==="
