#!/usr/bin/env bash
# LIBERO 评测统一默认参数（所有模型编排脚本应 source 本文件）
#
# 基准：LIBERO 官方 benchmark 协议（每 task 50 个初始状态）与主流 VLA 论文
# （OpenVLA / G0.5）沿用的一致设置。各模型论文特有的固有参数（动作空间、
# chunk 步数、精度）不在此统一，见 docs/EVAL_PROTOCOL.md 的"模型固有参数"。
#
# 用法：source scripts/eval_defaults.sh
#       覆盖默认值：LIBERO_TRIALS_PER_TASK=10 source scripts/eval_defaults.sh

# ── 评测协议（统一）────────────────────────────────────────────
export LIBERO_SEED="${LIBERO_SEED:-42}"                 # 固定随机种子（可复现）
export LIBERO_TRIALS_PER_TASK="${LIBERO_TRIALS_PER_TASK:-50}"  # 项目推荐/对照官方的默认全量 50；缩减验证用 10 且必须在 manifest 标注
export LIBERO_ENV_RESOLUTION="${LIBERO_ENV_RESOLUTION:-256}"   # 相机分辨率 256x256
export LIBERO_NUM_PARALLEL="${LIBERO_NUM_PARALLEL:-10}"        # 并行仿真环境数（资源受限可降）
export LIBERO_STEPS_WAIT="${LIBERO_STEPS_WAIT:-20}"            # 初始静置步数
export LIBERO_VIDEO=1                                      # 正式验证固定输出视频

# ── 四 suite 与 horizon 上限（官方协议，固定，不提供环境变量覆盖）────
export LIBERO_SUITES="${LIBERO_SUITES:-libero_spatial libero_object libero_goal libero_10}"
export LIBERO_HORIZON_SPATIAL=220
export LIBERO_HORIZON_OBJECT=280
export LIBERO_HORIZON_GOAL=300
export LIBERO_HORIZON_10=520

# ── 结果目录约定（统一）────────────────────────────────────────
# 全量：  results/<model>/full_libero/<YYYYmmdd_HHMMSS_PID>/<suite>/
# 冒烟：  results/<model>/smoke_<date>/
# 每个 run 根目录包含 manifest.json、server.log；每个 <suite>/ 包含 client.log（或 eval.log）、
#   <suite>_results.json（或 spatial_results.json）、videos/*.mp4
export LIBERO_RESULTS_ROOT="${LIBERO_RESULTS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/results}"

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "LIBERO_SEED=$LIBERO_SEED  TRIALS/TASK=$LIBERO_TRIALS_PER_TASK  PARALLEL=$LIBERO_NUM_PARALLEL"
  echo "HORIZONS (fixed official): spatial=$LIBERO_HORIZON_SPATIAL object=$LIBERO_HORIZON_OBJECT goal=$LIBERO_HORIZON_GOAL libero_10=$LIBERO_HORIZON_10"
  echo "SUITES: $LIBERO_SUITES"
  echo "RESULTS_ROOT: $LIBERO_RESULTS_ROOT"
fi
