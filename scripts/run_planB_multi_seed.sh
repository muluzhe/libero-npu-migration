#!/bin/bash
# 方案B：多seed验证（5个seed×10ep×4suite）
# 每个seed跑一轮完整4suite验证，输出到 results/xvla_npu_seed{S}_10ep/
# 完成后聚合5个seed的结果算平均±方差，与论文GPU基准对比
set -e

PY=python
PROJECT=${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
SEEDS=(42 123 456 789 2024)
SUITES="libero_spatial libero_goal libero_object libero_10"
EP=10
PORT=8010

echo "=== 方案B：多seed验证（5个seed×${EP}ep×4suite）==="
echo "seeds: ${SEEDS[*]}"

# 关键环境变量
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=swrast
export PYOPENGL_PLATFORM=osmesa
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0

# 1. 重启服务器（用默认seed，实际seed由XVLA_SEED环境变量控制每轮）
echo "[1] 重启推理服务器..."
ps -ef | grep deploy.py | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null; sleep 2
cd ${X_VLA_ROOT:?usage: X_VLA_ROOT env var required} && rm -rf logs
nohup setsid env ASCEND_RT_VISIBLE_DEVICES=0 $PY deploy.py \
  --model_path ${MODEL_PATH:?usage: MODEL_PATH env var required} --device auto --port $PORT --host 0.0.0.0 --output_dir logs \
  > /tmp/xvla_srv_planB.log 2>&1 &
disown 2>/dev/null
for i in $(seq 1 50); do
  ss -tlnp 2>/dev/null | grep -q ":$PORT" && { echo "  ✅ 服务器就绪"; break; }
  sleep 5
done

# 2. 逐seed跑验证
cd ${X_VLA_ROOT:?usage: X_VLA_ROOT env var required}/evaluation/libero
for S in "${SEEDS[@]}"; do
  OUT=$PROJECT/results/xvla_npu_seed${S}_${EP}ep
  echo ""
  echo "[seed=$S] 输出: $OUT"
  # 跳过已完成的seed（幂等）
  if [ -f "$OUT/results.json" ]; then
    echo "  已存在，跳过"
    continue
  fi
  XVLA_SEED=$S $PY libero_client.py \
    --server_ip 127.0.0.1 --server_port $PORT \
    --task_suites $SUITES \
    --eval_time $EP \
    --output_dir "$OUT" \
    --init_seed 42 --act_type abs \
    > "$OUT.log" 2>&1
  echo "  完成: $(cat $OUT/results.json 2>/dev/null)"
done

# 3. 聚合结果
echo ""
echo "=== 聚合5个seed结果 ==="
$PY << 'PYEOF'
import json, os, statistics
PROJECT='$PROJECT'
seeds=[42,123,456,789,2024]
suites=['libero_spatial','libero_goal','libero_object','libero_10']
print(f"{'seed':<6} {'spatial':<10} {'goal':<8} {'object':<8} {'long':<8} {'avg':<8}")
print('-'*50)
all_rates={s:[] for s in suites}
completed=0
for seed in seeds:
    f=f'{PROJECT}/results/xvla_npu_seed{seed}_10ep/results.json'
    if not os.path.exists(f):
        print(f'{seed:<6} (未完成，跳过)')
        continue
    try:
        d=json.load(open(f))
        # 确认4个suite都齐
        if not all(s in d for s in suites):
            print(f'{seed:<6} (部分完成，跳过)')
            continue
    except Exception:
        print(f'{seed:<6} (results.json损坏，跳过)')
        continue
    rates={k:v*100 for k,v in d.items()}
    avg=sum(rates.values())/4
    for s in suites: all_rates[s].append(rates.get(s,0))
    completed+=1
    print(f'{seed:<6} {rates["libero_spatial"]:>6.1f}%   {rates["libero_goal"]:>5.1f}%   {rates["libero_object"]:>5.1f}%   {rates["libero_10"]:>5.1f}%   {avg:>5.1f}%')
print('-'*50)
if completed==0:
    print('无已完成seed，无法聚合')
elif completed<len(seeds):
    print(f'仅{completed}/{len(seeds)}个seed完成，跳过avg/std聚合（需全部完成）')
else:
    print('avg:   ' + '  '.join(f'{statistics.mean(v):>5.1f}%' for v in all_rates.values()) + f'  {statistics.mean([statistics.mean(v) for v in all_rates.values()]):>5.1f}%')
    print('std:   ' + '  '.join(f'{statistics.stdev(v):>5.1f}%' for v in all_rates.values()) + f'  {statistics.stdev([statistics.mean(v) for v in all_rates.values()]):>5.1f}%')
print()
print('论文GPU基准: spatial 98.2% / goal 97.8% / object 98.6% / long 97.6% / avg 98.1%')
PYEOF
