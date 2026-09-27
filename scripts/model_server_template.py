"""通用 VLA NPU server 适配模板。

新模型适配时复制到 scripts/<model>/ 或 models/<model>/，只替换：
1. load_policy：加载 checkpoint 到 NPU
2. build_observation：把 LIBERO 请求转换为模型输入
3. predict_action：返回 [T, 7] 的 LIBERO 动作

接口约定：
- 输入图像必须明确 dtype/range（常见为 uint8 或 float32 [0, 1]）
- 明确姿态约定、state schema、绝对/相对动作语义和夹爪语义
- 验证脚本必须保存每个 rollout 的视频，并把 JSON/日志/视频放入 results/<model>/
"""

from __future__ import annotations

from typing import Any

import numpy as np


def load_policy(checkpoint: str, device: str) -> Any:
    """加载模型并移动到 device；实现时保持模型推理 dtype 与输入 dtype 一致。"""
    raise NotImplementedError


def build_observation(request: dict[str, Any]) -> dict[str, Any]:
    """将 /act 请求转换为模型训练时的输入 schema。"""
    raise NotImplementedError


def predict_action(policy: Any, observation: dict[str, Any]) -> np.ndarray:
    """返回 [T, 7] 的 LIBERO action：[delta_pos3, delta_aa3, grip1]。"""
    raise NotImplementedError
