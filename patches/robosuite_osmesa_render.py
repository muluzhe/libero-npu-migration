class _PatchedMjRenderContext:
    """用 mujoco 3.10 原生 Renderer 实现的渲染上下文（替代崩溃的 MjrContext）。"""

    def __init__(self, sim, offscreen=True, device_id=-1, max_width=640, max_height=480):
        assert offscreen
        self.sim = sim
        self.offscreen = offscreen
        self.device_id = device_id
        sim.forward()
        sim.add_render_context(self)
        self.model = sim.model
        self.data = sim.data
        # mujoco 3.10 原生 Renderer（OSMesa 软件渲染兼容）
        self._renderer = mujoco.Renderer(sim.model._model, max_height, max_width)
        self._cam = mujoco.MjvCamera()
        self._cam.fixedcamid = 0
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # scn/vopt 兼容外部访问（Renderer 内部有 _scn/_mujoco_renderer，这里建独立 MjvScene）
        self.scn = mujoco.MjvScene(sim.model._model, maxgeom=1000)
        self.vopt = mujoco.MjvOption()

    def update_offscreen_size(self, width, height):
        if width > self._renderer.height or height > self._renderer.width:
            self._renderer = mujoco.Renderer(self.model._model, height, width)

    def upload_texture(self, tex_id):
        pass  # 原生 Renderer 自动处理纹理

    def render(self, width, height, camera_id=None, segmentation=False):
        if camera_id is not None:
            if camera_id == -1:
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._cam.fixedcamid = camera_id
        self._renderer.update_scene(self.data._data, self._cam, self.vopt)
        # 禁用 swrast GLSL 崩溃特性（阴影/反射/雾/天空盒/线框/CSG）
        try:
            scn = self._renderer.scene
            scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
            scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            scn.flags[mujoco.mjtRndFlag.mjRND_CSG_FACE] = 0
            scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0
        except Exception:
            pass
        self._last_img = self._renderer.render()

    def read_pixels(self, width, height, depth=False, segmentation=False):
        img = getattr(self, '_last_img', None)
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
        # 调整尺寸（若渲染尺寸 != 请求尺寸）
        if img.shape[0] != height or img.shape[1] != width:
            from PIL import Image
            img = np.array(Image.fromarray(img).resize((width, height)))
        # 关键修复：mujoco.Renderer.render() 返回正常方向(原点左上)，
        # 但 GPU 的 mjr_readPixels 返回 GL 坐标(原点左下，上下颠倒)。
        # robosuite/LIBERO 客户端依赖 GPU 的颠倒行为(再 flip 恢复)。
        # 这里模拟 GPU 行为：上下翻转，让 robosuite 拿到与 GPU 一致的图像。
        img = np.flip(img, 0).copy()
        if depth:
            return (img, np.zeros((height, width), dtype=np.float32))
        return img

    def __del__(self):
        pass



