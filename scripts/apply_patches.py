#!/usr/bin/env python3
"""
LIBERO 仿真 NPU 迁移：自动应用所有 patch
这是确保仿真在 NPU(无GPU)环境跑通的唯一必要修改。

核心修复点（唯一）：
  robosuite binding_utils.py 的 MjRenderContext.read_pixels()
  → 模拟 GPU 的 GL 坐标（上下翻转），让 OSMesa 渲染图与 GPU 渲染一致

原理：
  GPU 的 mjr_readPixels 返回 GL 坐标(原点左下，图像上下颠倒)
  OSMesa 的 Renderer.render() 返回正常坐标(原点左上)
  LIBERO 客户端的 _flip_agentview 假设输入是 GPU 的颠倒图
  → 必须在 read_pixels 里翻转 OSMesa 图，模拟 GPU 行为
  → 这样客户端零改动，视频方向正确，模型输入与训练数据一致

用法: python3 apply_patches.py
"""
import os, sys, site, shutil
from pathlib import Path

def get_site_packages():
    return Path(site.getsitepackages()[0])

def patch_read_pixels(binding_utils_path):
    """核心修复：在 read_pixels 模拟 GPU 的 GL 坐标翻转"""
    fp = Path(binding_utils_path)
    content = fp.read_text()

    # 已应用检查
    if "模拟 GPU 行为：上下翻转" in content:
        print("✅ [1/3] read_pixels GL坐标修复 已应用")
        return True

    # 注入 _PatchedMjRenderContext 类（替代崩溃的 MjrContext）
    if "_PatchedMjRenderContext" not in content:
        patch_file = Path(__file__).parent.parent / "patches" / "robosuite_osmesa_render.py"
        patched_class = patch_file.read_text()
        content = content.replace(
            "_MjSim_render_lock = Lock()",
            "_MjSim_render_lock = Lock()\n\n" + patched_class, 1)

    # 替换 MjRenderContext 类定义为继承 patch 基类
    old_class = '''class MjRenderContext:
    """
    Class that encapsulates rendering functionality for a
    MuJoCo simulation.

    See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjrendercontext.pyx
    """'''
    new_class = '''class MjRenderContext(_PatchedMjRenderContext):
    """PATCHED: 用 mujoco 3.10 原生 Renderer 替代崩溃的 MjrContext (OSMesa segfault)"""'''
    if old_class in content:
        content = content.replace(old_class, new_class, 1)
    else:
        print("⚠️ [1/3] 未找到 MjRenderContext 原始类定义，可能已 patch")
        return False

    fp.write_text(content)
    print("✅ [1/3] read_pixels GL坐标修复 已应用")
    return True

def patch_mj_fullm(controller_path):
    """mujoco 3.10 API 兼容：mj_fullM 新签名"""
    fp = Path(controller_path)
    content = fp.read_text()
    if "mj_fullM(self.sim.model._model, self.sim.data._data" in content:
        print("✅ [2/3] mj_fullM API 兼容 已应用")
        return True
    old = "mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)"
    new = """try:
                mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix)
            except (TypeError, AttributeError):
                mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)"""
    if old not in content:
        print("⚠️ [2/3] 未找到 mj_fullM 原始调用")
        return False
    content = content.replace(old, new, 1)
    fp.write_text(content)
    print("✅ [2/3] mj_fullM API 兼容 已应用")
    return True

def verify_libero_client(client_path):
    """确认 libero_client.py 是官方原版（无需修改）"""
    fp = Path(client_path)
    if not fp.exists():
        print("⚠️ [3/3] libero_client.py 不存在")
        return False
    content = fp.read_text()
    if "images.append(_flip_agentview(obs['agentview_image']))" in content:
        print("✅ [3/3] libero_client.py 官方原版（无需修改）")
        return True
    print("⚠️ [3/3] libero_client.py 非官方版本，建议还原")
    return False

def main():
    sp = get_site_packages()
    print("=" * 60)
    print("LIBERO 仿真 NPU 迁移 — patch 应用")
    print(f"site-packages: {sp}")
    print("=" * 60)

    # 核心修复（唯一必要）：read_pixels GL 坐标翻转
    patch_read_pixels(sp / "robosuite" / "utils" / "binding_utils.py")
    # mujoco 3.10 API 兼容
    patch_mj_fullm(sp / "robosuite" / "controllers" / "base_controller.py")
    # 确认 libero_client 官方原版
    import os
    verify_libero_client(os.environ.get('X_VLA_ROOT', os.path.expanduser('~/work/X-VLA')) + "/evaluation/libero/libero_client.py")

    print()
    print("核心修复说明：")
    print("  唯一修改点 = read_pixels 的 np.flip(img, 0)")
    print("  libero_client.py 保持官方原样，零改动")
    print("  这确保视频方向正确 + 模型输入与训练数据一致")

if __name__ == "__main__":
    main()
