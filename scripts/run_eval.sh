#!/bin/bash
# LIBERO 仿真验证运行脚本
# 用法: bash run_eval.sh <VLA模型> <模型路径> <输出目录> <episodes数>
# VLA模型: xvla | openvla | pi0 | smolvla | act | diffusion_policy
set -e

MODEL_TYPE=${1:-xvla}
MODEL_PATH=${2:-${MODEL_PATH:?usage: MODEL_PATH env var required}}
OUTPUT_DIR=${3:-./libero_eval_results}
NUM_EPISODES=${4:-10}
PORT=8010

# 关键环境变量（OSMesa 软件渲染 + NPU + numba禁用）
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=swrast
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== LIBERO 仿真验证 ($MODEL_TYPE) ==="
echo "模型: $MODEL_PATH | 输出: $OUTPUT_DIR | Episodes: $NUM_EPISODES"

# 1. 启动 VLA 推理服务器
echo "[1/2] 启动 $MODEL_TYPE NPU 推理服务器 (端口 $PORT)..."
ps -ef | grep "deploy\|model_server" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null; sleep 2
nohup setsid python3 "$PROJECT_DIR/models/${MODEL_TYPE}/server.py" \
  --model_path "$MODEL_PATH" --port $PORT > /tmp/${MODEL_TYPE}_srv.log 2>&1 &
disown 2>/dev/null
echo "等待模型加载到 NPU..."
for i in $(seq 1 60); do
  ss -tlnp 2>/dev/null | grep -q ":$PORT" && { echo "服务器就绪"; break; }
  sleep 5
done

# 2. 跑 LIBERO 仿真验证（统一客户端，各模型共用）
echo "[2/2] 启动 LIBERO 仿真验证..."
cd ${X_VLA_ROOT:?usage: X_VLA_ROOT env var required}/evaluation/libero
mkdir -p "$OUTPUT_DIR"
python3 libero_client.py \
  --server_ip 127.0.0.1 --server_port $PORT \
  --task_suites libero_spatial libero_goal libero_object libero_10 \
  --eval_time $NUM_EPISODES \
  --output_dir "$OUTPUT_DIR" \
  --init_seed 42 --act_type abs

echo "=== 完成 ==="
echo "结果: $OUTPUT_DIR/*/results.json"
echo "视频: $OUTPUT_DIR/*/*.mp4"
