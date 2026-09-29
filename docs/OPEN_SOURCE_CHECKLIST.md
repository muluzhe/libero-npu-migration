# 开源发布清单（你需要做什么）

从本地整理好的项目目录到 GitHub 公开仓库，按以下顺序操作。每一步都列了"为什么"和"验证方法"，做完一步再走下一步。

---

## 0. 前置：确认本地项目已整理完毕

本仓库已按当前公开范围整理过，但发布前仍必须重新扫描当前工作树和 Git 暂存内容。文档与脚本中应使用 `$HOME`、`$PROJECT_ROOT` 或相对路径；模型权重、视频、日志和缓存不纳入公开仓库。

已完成的整理项包括：
- 开源规范文件补全（LICENSE / .gitignore / CONTRIBUTING / CODE_OF_CONDUCT / CITATION / CHANGELOG）
- 目录结构与文件命名规范化
- 已知个人路径、邮箱、账号、内部地址和 token 的脱敏处理

本次 main 更新对 21 个新增或修改的暂存文件执行了令牌、邮箱、个人绝对路径及大文件扫描，未检出上述敏感信息；公开版本仅包含脱敏的 PI0.5/G0.5 统计汇总。原始 JSON、日志、权重、视频仍在本机并已从暂存范围排除。既有公开 Git 历史不会因本次提交被重写，因此这一检查结论仅适用于本次新增或修改的文件，不能作为既有历史的无泄露保证。

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

## 2. 审查当前 main 分支提交范围

本仓库已有 Git 历史和远端 main；更新时保留历史，不重建仓库、不执行 `git add .`。早期公开的历史文件无法靠本次更新抹去，审计结论要区分“本次新增内容”和“既有历史”。

```bash
git status --short
git diff --check
git diff --cached --check
# 仅明确选择将发布的文档、代码、脱敏汇总以及经过审查的删除项。
```

**验证**：暂存文件里**不应该**出现：
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
git remote -v         # 确认 origin 指向目标仓库
git branch --show-current  # 确认位于 main
git push origin main
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

## 历史发布决策记录（2026-07-18）

| 事项 | 决定 | 执行状态 |
|---|---|---|
| GitHub 仓库地址 | 以发布时实际创建的公开仓库为准 | ⏳ 发布前确认 |
| LICENSE | Apache-2.0 | ✅ 已替换 `LICENSE` 全文，已同步 `CONTRIBUTING.md` / `CHANGELOG.md` 许可引用 |
| `docs/openvla/OPENVLA_HANDOVER.md` 和 `docs/PROJECT_TRACKING.md` | 保留现状，随仓库公开 | ✅ 未动 |
| 2000 个 rollout 视频（37MB） | 本地留存，不入库 | ✅ 已在 `/tmp/libero_videos_backup/`，`.gitignore` 排除 `*.mp4` |

这些是历史发布决策记录。实际发布前仍需重新确认仓库地址、权限、许可证、扫描结果和当前工作树内容，不应直接把本段当作当前发布结论。

---

## 附录 A：发布前检查记录（2026-07-18，原 FINAL_REVIEW.md 并入）

**结论（07-18 时点）**：当时的文件清理检查通过。该历史快照不覆盖后续新增文件，也不等同于当前仓库不存在个人数据、内部地址或凭证；发布前必须按附录 A/B 重新扫描。

复查命令（发布前重跑）：
```bash
grep -rE "(/home/|/anaconda3/|[[:alnum:]_.-]+@[[:alnum:]_.-]+|192\.168\.|10\.[0-9]+\.|token|password|secret|api[_-]?key)" \
  --exclude-dir=.git --exclude-dir=kernel_meta . 2>/dev/null | grep -v "\.pyc"
find . -name "*.py" -not -path "*/.*" | xargs -n1 python -m py_compile && find . -name "*.sh" | xargs -n1 bash -n
```

## 附录 B：07-18 检查后新增内容的重扫清单（2026-09-26）

07-18 的"零残留"结论**不覆盖** 07-20 后新增的 PI0.5 阶段文件。2026-09-26 项目整理时发现并已处理：

| 项 | 状态（2026-09-26） |
|---|---|
| GPU 服务器地址/SSH 凭证（PI0.5 历史记录及交接材料） | ✅ 已脱敏为占位符或移出公开文档 |
| PI0.5 系列脚本中的本机绝对路径 | ⚠️ 发布前需继续扫描并统一为环境变量或相对路径 |
| 07-17 隐私清理误伤：bash 语法 `${X_VLA_ROOT:?...}` 混入 5 个 .py 文件（eval/diag/apply_patches） | ✅ 已修复为 os.environ.get + 绝对路径默认值 |
| `kernel_meta/`（NPU 编译缓存）与 `__pycache__/` 重新累积 | ⚠️ 发布前删除并确认 .gitignore 生效 |
| `results/pi05_spatial/progress.log`、`ASSET_SNAPSHOT_2026-07-21.md`（过期结论） | ⚠️ ASSET_SNAPSHOT 已删除；progress.log 按 .gitignore 排除 |
| gpu_infer_compare.py / i2_real_obs_infer.py 的 state schema 漂移（P4.28） | ✅ 已对齐 server_v2.py（P4.7/P4.8 schema） |

**发布前必做**：重跑附录 A 复查命令 + `git status` 人工过目 staged 文件。
