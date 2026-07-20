# 开源发布清单（你需要做什么）

从本地整理好的项目目录到 GitHub 公开仓库，按以下顺序操作。每一步都列了"为什么"和"验证方法"，做完一步再走下一步。

---

## 0. 前置：确认本地项目已整理完毕

本仓库已经完成：
- 隐私信息清理（用户名、绝对路径、云厂商代理、内部主机名全替换为环境变量）
- 开源规范文件补全（LICENSE / .gitignore / CONTRIBUTING / CODE_OF_CONDUCT / CITATION / CHANGELOG）
- 目录结构与文件命名规范化
- 视频/日志/`__pycache__` 移出

最终体积约 976KB（不含模型权重和视频，这些已被 .gitignore 排除）。

---

## 1. 在 GitHub 上创建空仓库

1. 登录你的 GitHub，点右上角 `+` → `New repository`。
2. 填写：
   - Repository name: `libero-npu-migration`（或你喜欢的名字）
   - Description: `VLA closed-loop evaluation on Ascend 910B4 NPU via OSMesa software rendering`
   - Visibility: **Public**
   - **不要**勾选 "Initialize this repository with README/.gitignore/LICENSE"——本项目已自带这些文件，勾了会冲突。
3. 点 `Create repository`。GitHub 会给一个空仓库地址，形如 `https://github.com/<你的 handle>/libero-npu-migration`。

> 记住这个地址，下面第 3 步要用。

---

## 2. 本地 git 初始化（重要：从干净历史开始）

⚠️ 强烈建议**新建一个干净的 git 历史**，不要把本地已有的（可能含早期隐私版本的）历史推上去。因为即使删了文件，git 历史里还留着旧内容，公开后别人 `git log -p` 能翻出来。

```bash
cd /path/to/your/libero-npu-migration

# 删掉可能存在的旧 .git（如有），从零开始
rm -rf .git

# 新建干净历史
git init
git add .
git status            # ←【验证】人工扫一眼，确认没有 .mp4 / .log / __pycache__ / 权重文件被加进来
git commit -m "Initial public release: VLA closed-loop eval on Ascend 910B4 NPU" -m "Co-Authored-By: AtomCode (GLM-5.2) <noreply@atomgit.com>"
```

**验证**：`git status` 输出的 staged 文件里**不应该**出现：
- 任何 `.mp4`（视频）
- 任何 `.log`（日志）
- 任何 `__pycache__` / `*.pyc`
- 任何 `.bin` / `.safetensors` / `.pt`（模型权重）
- `X-VLA/` / `openvla/` / `libero/`（外部依赖源码目录）
- `render_libs/`（OSMesa 库）

如果出现了，说明 `.gitignore` 没生效，回到第 0 步检查。

---

## 3. 推送到 GitHub

```bash
git remote add origin https://github.com/<你的 handle>/libero-npu-migration.git
git branch -M main
git push -u origin main
```

首次推送可能要输 GitHub 用户名 + Personal Access Token（GitHub 已不接受密码，用 PAT 当密码，在 Settings → Developer settings → Personal access tokens 生成，勾 `repo` 权限即可）。

**验证**：浏览器打开仓库地址，能看到 README 首页正常渲染。

---

## 4. 补 GitHub 仓库元信息

在 GitHub 仓库页面点 ⚙️（About）设置：
- **Description**: 复制第 1 步填的
- **Topics**（点齿轮 → Topics 加标签，有助于被搜到）：
  `ascend-npu` `ascend-910b` `vla` `vision-language-action` `openvla` `x-vla` `libero` `robotics-evaluation` `cann` `torch-npu`
- **Releases**: 如果想给个正式版本号，点 `Releases` → `Draft a new release`，tag 填 `v0.1.0`，标题 `Initial release`，描述可贴 `CHANGELOG.md` 顶部内容。

---

## 5. 用 GitHub 自带工具做最后一道隐私扫描

1. 仓库页面 → `Security` 标签 → `Secret scanning` → 确认已 Enable（Public 仓库默认开）。
2. 等 1-2 分钟，看 `Security` → `Secret scanning alerts` 有没有命中。有就按提示处理（一般是误报或漏网 token）。
3. （可选）`Code scanning` → `Set up code scanning` → 选 `Default`，跑一次 CodeQL 静态扫描，看有没有明显的注入/密钥硬编码告警。

**验证**：Security 面板三项（Dependabot / Secret scanning / Code scanning）全绿或无未处理告警。

---

## 6.（可选）挂 CI 做语法回归

如果想给后续 PR 自动加一道语法检查，在仓库根新建 `.github/workflows/syntax.yml`：

```yaml
name: syntax-check
on: [push, pull_request]
jobs:
  py:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: find . -name "*.py" -not -path "*/.*" | xargs -n1 python -m py_compile
  sh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: find . -name "*.sh" | xargs -n1 bash -n
```

非必需，但低成本高收益。

---

## 7. 宣发（可选）

- 在相关社区（机器人/RL/NPU 迁移群）分享 README 链接。
- 若引用了 X-VLA / OpenVLA / LIBERO 论文，按 `CITATION.md` 的 BibTeX 在你的工作中回引上游。
- 给仓库点 ⭐ 让后来者更容易找到。

---

## 用户已确认的决策（2026-07-18）

| 事项 | 决定 | 执行状态 |
|---|---|---|
| GitHub handle | `muluzhe` | ✅ 已替换到 `CITATION.md` 的 BibTeX `url` |
| LICENSE | Apache-2.0 | ✅ 已替换 `LICENSE` 全文，已同步 `CONTRIBUTING.md` / `CHANGELOG.md` 许可引用 |
| `docs/OPENVLA_HANDOVER.md` 和 `docs/PROJECT_TRACKING.md` | 保留现状，随仓库公开 | ✅ 未动 |
| 2000 个 rollout 视频（37MB） | 本地留存，不入库 | ✅ 已在 `/tmp/libero_videos_backup/`，`.gitignore` 排除 `*.mp4` |

所有决策已落地，可直接按本文件第 1~6 步操作发布到 `https://github.com/muluzhe/libero-npu-migration`。
