# 视频倒置问题根因与修复

## 问题现象
- 仿真产出的视频上下颠倒（桌面在上方，天花板在下方）
- 成功率 0%（模型收到错误方向图像，动作完全错误）

## 根因分析

### GPU 渲染 vs OSMesa 渲染的坐标系差异

| 渲染方式 | 图像原点 | 图像方向 | 桌面位置 |
|---|---|---|---|
| GPU `mjr_readPixels` | 左下（GL 标准） | 上下颠倒 | 顶部（亮） |
| OSMesa `Renderer.render()` | 左上（正常） | 正常方向 | 底部（亮） |

### LIBERO 客户端的数据流

```
robosuite env → obs['agentview_image'] → _flip_agentview(双翻转) → 发给模型 + 存视频
```

`_flip_agentview` (libero_client.py:69) 做 `np.flip(np.flip(img,0),1)`，它假设输入是 GPU 的颠倒图，翻转后恢复正确方向给模型。

### 问题的产生

OSMesa 渲染图已是正常方向（桌面在底），客户端再 `_flip_agentview` 翻转 → 桌面到顶 → 倒置 + 模型收到错误方向 → 0% 成功率。

## 修复方案（唯一修改点）

**文件**：`robosuite/utils/binding_utils.py` 的 `MjRenderContext.read_pixels()`

**修改**：在返回 OSMesa 渲染图前，做 `np.flip(img, 0)` 模拟 GPU 的 GL 坐标（上下颠倒）。

```python
def read_pixels(self, width, height, depth=False, segmentation=False):
    img = self._last_img  # OSMesa 渲染图（正常方向，桌面在底）
    # 关键修复：模拟 GPU 的 GL 坐标（上下翻转），让 robosuite 拿到与 GPU 一致的图像
    img = np.flip(img, 0).copy()  # 翻转后桌面在顶（与 GPU 一致）
    return img
```

**效果**：
- robosuite 的 `obs['agentview_image']` = GPU 方向（桌面在顶，颠倒）
- 客户端 `_flip_agentview` 翻转 → 桌面在底（正常）→ 模型收到正确方向 → 90% 成功率
- 视频存储用同一翻转图 → 方向正常

## 为什么只改这一处

| 方案 | 改动点 | 风险 |
|---|---|---|
| **本方案** | 仅 `read_pixels` 1 处 | 低，`libero_client` 零改动 |
| 错误方案A | 改 `read_pixels` + 改 `_rollout` | 依赖两处一致，易出错 |
| 错误方案B | 仅改 `_rollout` 不翻转 | 模型输入仍错误，0% 成功率 |

**关键原则**：`libero_client.py` 必须保持官方原样，所有适配在渲染层（`read_pixels`）完成。这确保：
1. 视频方向正确
2. 模型输入与训练数据一致
3. 客户端可随官方更新，无需维护本地改动

## 验证方法

```python
img = obs['agentview_image']  # read_pixels 后
print('顶部均值:', img[0].mean(), '底部均值:', img[-1].mean())
# 修复后: 顶部亮(>150) 底部暗(<110) = 模拟GPU成功
# 未修复: 顶部暗 底部亮 = OSMesa原样，会倒置
```
