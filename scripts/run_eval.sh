#!/bin/bash
# LIBERO 仿真验证运行脚本（X-VLA 一键路径）
# 用法: bash run_eval.sh <VLA模型> <模型路径> <输出目录> <episodes数> [act_type]
# VLA模型: xvla | openvla | pi0 | smolvla | act | diffusion_policy
#
# ⚠️ 模型支持说明（2026-09-26 更新，与实际验证路径对齐）：
#   - xvla   ：本脚本一键可跑（官方 deploy 链路，act_type=abs，已验证 95.8%）
#   - openvla：本脚本不可直接跑（需 server_v2 + act_type=rel + 每 suite 独立 ckpt/unnorm_key
#              + steps=1 monkey-patch），请用专用路径：
#              bash scripts/openvla/run_openvla_full_validation.sh <npu_device> <port> <suite>...
#   - pi0    ：本脚本不可直接跑（需独立 conda env lerobot-pi05 + server_v2），见
#              scripts/pi05/run_pi05_spatial.sh 与 docs/pi05/PI05_TRACKING.md
#   - smolvla / act / diffusion_policy：server 为未实测骨架，跑通仅验证链路
set -e

MODEL_TYPE=${1:-xvla}
MODEL_PATH=${2:-${MODEL_PATH:?usage: MODEL_PATH env var required}}
OUTPUT_DIR=${3:-./libero_eval_results}
NUM_EPISODES=${4:-10}
ACT_TYPE=${5:-abs}
PORT=8010

if [ "$MODEL_TYPE" = "openvla" ] || [ "$MODEL_TYPE" = "pi0" ]; then
  echo "[ERROR] $MODEL_TYPE 不能用本脚本一键跑（动作语义/依赖与 X-VLA client 不同），见头部说明"
  exit 1
fi

# 关键环境变量（OSMesa 软件渲染 + NPU + numba禁用）
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=swrast
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
X_VLA_ROOT=${X_VLA_ROOT:-$HOME/work/X-VLA}

echo "=== LIBERO 仿真验证 ($MODEL_TYPE) ==="
echo "模型: $MODEL_PATH | 输出: $OUTPUT_DIR | Episodes: $NUM_EPISODES | act_type: $ACT_TYPE"

# 1. 启动 VLA 推理服务器（先清本端口旧进程，避免误杀无关进程）
echo "[1/2] 启动 $MODEL_TYPE NPU 推理服务器 (端口 $PORT)..."
OLD_PID=$(ss -tlnp 2>/dev/null | grep ":$PORT" | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2)
[ -n "$OLD_PID" ] && { echo "端口 $PORT 被 PID $OLD_PID 占用，先清理"; kill -9 $OLD_PID 2>/dev/null || true; sleep 2; }
nohup setsid python3 "$PROJECT_DIR/models/${MODEL_TYPE}/server.py" \
  --model_path "$MODEL_PATH" --port $PORT > /tmp/${MODEL_TYPE}_srv.log 2>&1 &
disown 2>/dev/null
SRV_PID=$!
echo "等待模型加载到 NPU..."
READY=0
for i in $(seq 1 60); do
  ss -tlnp 2>/dev/null | grep -q ":$PORT" && { READY=1; echo "服务器就绪"; break; }
  kill -0 $SRV_PID 2>/dev/null || { echo "[ERROR] server 进程退出："; tail -20 /tmp/${MODEL_TYPE}_srv.log; exit 1; }
  sleep 5
done
if [ $READY -ne 1 ]; then
  echo "[ERROR] server 5 分钟未就绪，退出（log: /tmp/${MODEL_TYPE}_srv.log）"
  exit 1
fi

# 2. 跑 LIBERO 仿真验证（统一客户端，各模型共用）
echo "[2/2] 启动 LIBERO 仿真验证..."
cd "${X_VLA_ROOT:?usage: X_VLA_ROOT env var required}/evaluation/libero"
mkdir -p "$OUTPUT_DIR"
python3 libero_client.py \
  --server_ip 127.0.0.1 --server_port $PORT \
  --task_suites libero_spatial libero_goal libero_object libero_10 \
  --eval_time $NUM_EPISODES \
  --output_dir "$OUTPUT_DIR" \
  --init_seed 42 --act_type "$ACT_TYPE"

echo "=== 完成 ==="
echo "结果: $OUTPUT_DIR/*/results.json"
echo "视频: $OUTPUT_DIR/*/*.mp4"
