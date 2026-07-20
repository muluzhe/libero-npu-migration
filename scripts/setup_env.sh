#!/bin/bash
# LIBERO 仿真 NPU 迁移：环境搭建
# 适用：HCE 2.0 (glibc 2.34) + Ascend 910B + torch_npu
set -e
echo "=== LIBERO 仿真 NPU 迁移环境搭建 ==="

# 1. Python 依赖
echo "[1/4] Python 依赖..."
pip install -q mujoco==3.10.0 robosuite==1.4.1 bddl==1.0.1 robomimic==0.2.0 \
  "numpy==1.26.3" "scipy==1.15.0" thop future imageio \
  gymnasium "hydra-core>=1.2,<1.4" wandb termcolor \
  fastapi "uvicorn==0.34.3" "json_numpy==2.1.0" "einops==0.8.1" \
  "timm==1.0.12" "accelerate==1.2.1" "peft==0.17.1" "safetensors==0.4.5" \
  mediapy transformers 2>&1 | tail -1

# 2. LIBERO 仿真库
echo "[2/4] LIBERO 仿真库..."
pip download -q libero==0.1.1 --no-deps -d /tmp/libero_pkg 2>&1 | tail -1
pip install -q /tmp/libero_pkg/libero-0.1.1.tar.gz --no-deps 2>&1 | tail -1
printf 'N\n' | python3 -c "from libero.libero import benchmark" 2>/dev/null
echo "libero 安装完成"

# 3. OSMesa 渲染库（从 HCE 2.0 官方仓库，无需 root）
echo "[3/4] OSMesa 软件渲染库..."
mkdir -p ~/render_libs/dri /tmp/mesa_pkgs && cd /tmp/mesa_pkgs
yumdownloader --destdir /tmp/mesa_pkgs mesa-libOSMesa mesa-dri-drivers llvm-libs 2>&1 | tail -1
for rpm in *.rpm; do bsdtar -xf "$rpm" 2>/dev/null; done
cp -L /tmp/mesa_pkgs/usr/lib64/libOSMesa.so.8* ~/render_libs/
cp -L /tmp/mesa_pkgs/usr/lib64/libLLVM-12.so* ~/render_libs/
cp -L /tmp/mesa_pkgs/usr/lib64/dri/swrast_dri.so ~/render_libs/dri/
cd ~/render_libs && ln -sf libOSMesa.so.8* libOSMesa.so.8 2>/dev/null
ln -sf libOSMesa.so.8 libOSMesa.so 2>/dev/null
ln -sf libLLVM-12.so* libLLVM-12.so 2>/dev/null
echo "渲染库: $(ls ~/render_libs/*.so* ~/render_libs/dri/*.so 2>/dev/null | wc -l) 个"

# 4. LIBERO assets（从 Hugging Face 官方仓库下载）
echo "[4/4] LIBERO mujoco assets..."
rm -rf /tmp/libero-assets
git clone -q --depth 1 https://huggingface.co/jadechoghari/libero-assets /tmp/libero-assets 2>&1 | tail -1
LIBERO_PKG=$(python3 -c "import libero,os;print(os.path.dirname(libero.__file__))" 2>/dev/null)/assets
mkdir -p "$LIBERO_PKG" && cp -r /tmp/libero-assets/* "$LIBERO_PKG/"
cd "$LIBERO_PKG"
BASE="https://huggingface.co/jadechoghari/libero-assets/resolve/main"
grep -rl "oid sha256:" . 2>/dev/null | sed 's#^\./##' | xargs -P 8 -I {} \
  curl -sSL --fail --max-time 60 -o {} "$BASE/{}" 2>/dev/null
echo "LFS 剩余: $(grep -rl 'oid sha256:' . 2>/dev/null | wc -l)"

# 5. 应用 patch
echo "[5/4] 应用 NPU 迁移 patch..."
python3 "$(dirname "$0")/apply_patches.py"

echo ""
echo "=== 环境搭建完成 ==="
echo "运行仿真前设置环境变量："
cat <<'ENV'
export NUMBA_DISABLE_JIT=1
export MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=swrast
export LIBGL_DRIVERS_PATH=$HOME/render_libs/dri
export LD_LIBRARY_PATH=$HOME/render_libs:/usr/lib64:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0
ENV
