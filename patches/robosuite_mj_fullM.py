# 这是一个 patch 片段，不是独立模块。
# 被 scripts/apply_patches.py 字符串注入到 robosuite controllers 的方法体内。
# 下方缩进的代码块用 `if False:` 守卫仅为了让本文件能通过 py_compile 验证，
# 实际运行时由 apply_patches.py 把内部代码嵌入到目标方法中。
if False:
            mass_matrix = np.ndarray(shape=(self.sim.model.nv, self.sim.model.nv), dtype=np.float64, order="C")
            # mujoco 3.10: mj_fullM(m, d, dst) — 适配新签名
            try:
                mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix)
            except (TypeError, AttributeError):
                mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)
            mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
            self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]
