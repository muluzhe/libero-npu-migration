# Changelog

All notable changes to this project are documented here. Dates are local time. Format loosely follows Keep a Changelog.

## [Unreleased]

### Documentation audit (2026-09-27)
- Corrected current model summaries from the raw JSON/log evidence: X-VLA suite average 95.75%; OpenVLA full results spatial/goal/object/libero_10 = 77/79/75/57%; PI0.5 spatial 96/100 with task5 70%, task8 100%, task9 90%.
- Corrected PI0.5 elapsed time to 10359s = 172.65min = 2.8775h (about 103.59s/episode); separated raw statistics from paper references and marked unverified references as pending verification.
- Recorded the user report that historical X-VLA/OpenVLA video organization was lost; current saved video evidence is only the 10-episode OpenVLA object recheck at 5/10, and no deletion proof is retained.
- Added an audit record to `docs/PROJECT_TRACKING.md`; historical entries remain historical snapshots and are not current status.

## 2026-09-26（下午）— PI0.5 闭环 0% 终结（P4.29）+ OpenVLA 断点续跑

### Fixed (P4.29, PI0.5 真根因)
- **图像 dtype（主因）**: server_v2.py 直传 uint8 图像 → 模型内 `to(float32)` 得 0-255 → `resize_with_pad_torch` float32 分支 `clamp(-1,1)` 静默摧毁成近二值图。修复：`permute(2,0,1).float()/255.0`（对齐官方 observation_processor.py:90）。
- **state 姿态 OOD**: P4.8 把 `robot0_eef_quat` 按 wxyz 误解释（robosuite 实为 xyzw）→ aa≈0 vs 训练分布 aa_x≈π（~10σ OOD）。修复：按 xyzw 转 aa（lerobot env_processor 同源），字段必填 fail-fast。
- **proprio 冻结**: client `self.proprio` 仅首查初始化 + action 回传 → server 每查收到首帧位置。修复：eval 每步 `client.proprio=None`（client 文件零改动）。
- **wrist 未翻转**: 训练对全部相机图 H+W 双翻转，client 只翻 agentview。修复：server 端补翻 wrist。
- **grip 反转错配**: `1-grip` 移植自 OpenVLA 约定，π0.5 训练 grip 即 env 语义（-1=开/+0.92=合）。修复：直传。

### Removed (P4.29 清理无效止血)
- F1 chunk=2 硬覆盖（恢复官方 n_action_steps=10）、P4.5v2 末步 proprio 回传 hack、P4.8 ori6d perm+signs 转换链——均为错误输入时代的产物。
- GPU 对比方向 D/G 作废（旧基准 `pi05_npu_baseline_2026-09-26.json` 与 server 同源同错，照跑会两端一致误判"模型坏"）。

### Added
- `eval_pi05_spatial.py`：monkey-patch 注入 env 真值 `robot0_eef_quat`/`robot0_gripper_qpos`（client 官方零改动）+ 每步 fresh proprio。
- `eval_openvla_suite.py --resume` / 编排器 `RESUME=1`：断点续跑（seeding 与 task 顺序无关，统计等价于全新跑）。
- 诊断方法资产：**训练分布对照法**（server 输入逐维对照 ckpt norm_stats）+ 四输入敏感性测试 → PI0.5 半天收口 3 个月闭环 0%（教训 #36/#37）。

### Validated
- **PI0.5 spatial 原始统计：96.0%（96/100，seed42，horizon220）**。逐 task：task0-4/6/7/8=100%、task5=70%、task9=90%。总耗时 10359.117s（172.65min，2.8775h，约103.59s/ep）；结果 `results/pi05/spatial_100ep_2026-09-26/spatial_results.json`。外部参考值和协议来源待核实。
- OpenVLA spatial 全量 100 rollouts：**77.0%**（官方协议）vs 官方 84.7%±0.9%。
- X-VLA 审查：5 seed 全一致（95.75%），无需完善。

## 2026-09-26 — Project overhaul: P0 fixes, doc consolidation, OpenVLA full validation launch, PI0.5 baseline regen

### Fixed (P0)
- **Broken scripts from 07-17 privacy cleanup**: bash syntax `${X_VLA_ROOT:?...}` had been mechanically written into 5 Python files (`eval_spatial_full.py`, `eval_spatial_task0_v2/v3`, `diag_compare_action.py`, `diag_stage1_official.py`, `apply_patches.py`) — replaced with `os.environ.get()` + absolute-path defaults.
- **Credential leak**: GPU server IP/SSH username/password in `PI05_RECORD`/`PI05_TRACKING`/`ATOMCODE_HANDOVER`/`gpu_http_proxy.sh` — all sanitized to placeholders; `gpu_http_proxy.sh` now auto-detects `GPU_PUBLIC_IP`.
- **run_eval.sh wrong server entry**: openvla/pi0 now explicitly rejected with pointers to their dedicated paths (v1/skeleton servers were known-broken); port-based kill; fail-fast on server not ready.
- **P4.28 schema drift**: `gpu_infer_compare.py`/`i2_real_obs_infer.py` still used old state schema `[pos3, quat4, grip1]` vs server's P4.7/P4.8 `[pos3, axis_angle3, gripper_qpos2]` — would have polluted the NPU-vs-GPU root-cause comparison. Both aligned; `gpu_infer_compare.py` rewritten as dual-endpoint (npu/cuda) single script.
- **P4.28b processor construction**: `make_pre_post_processors(policy.config)` without ckpt path silently used hardcoded `google/paligemma-3b-pt-224` tokenizer (fails offline) AND skipped norm_stats normalization — now passes CKPT like `server_v2.py:362`.

### Added
- `scripts/eval_openvla_suite.py` — generalized suite eval (4 suites, official per-suite step limits 220/280/300/520, argparse), replaces `eval_spatial_full.py`.
- `scripts/run_openvla_full_validation.sh` — full-validation orchestrator (auto server start/stop, shard integrity check, hardened Ascend env sourcing).
- OpenVLA 4-suite full validation launched (400 rollouts, dual-NPU streams) — results in `results/openvla_full/`.
- PI0.5 NPU baseline regeneration with fixed schema/processor → `results/pi05_npu_baseline_2026-09-26.json`.
- Evaluation protocol convention table in `docs/LIBERO_NPU_MIGRATION.md` §5.

### Changed (docs consolidation, 14 → 8)
- `docs/LIBERO_NPU_MIGRATION.md` — now the summary doc; absorbed `EXTENSIBILITY.md` + `VIDEO_ORIENTATION.md`; milestones updated through 09-26.
- `docs/OPENVLA_HANDOVER.md` — absorbed OpenVLA issues from `LIBERO_NPU_RECORD.md`; added §8 full validation; §2-§4 marked as historical snapshot.
- `docs/PI05_TRACKING.md` — absorbed `ATOMCODE_HANDOVER.md` + `GPU_INFER_README.md` into new "接手指引" appendix; added §六.19 (P4.28/28b).
- `docs/OPEN_SOURCE_CHECKLIST.md` — absorbed `FINAL_REVIEW.md` as appendix A + new re-scan checklist appendix B.
- Deprecated markers added: `models/openvla/server.py` (v1), `models/pi0/server.py` (skeleton), `run_openvla_libero.sh`, `i2_real_obs_infer.py`.

### Removed
- `docs/EXTENSIBILITY.md`, `docs/VIDEO_ORIENTATION.md`, `docs/LIBERO_NPU_RECORD.md`, `docs/FINAL_REVIEW.md`, `docs/ATOMCODE_HANDOVER.md`, `scripts/GPU_INFER_README.md`, `results/pi05_spatial/ASSET_SNAPSHOT_2026-07-21.md` — all merged into the 8-doc structure.

## 2026-07-20 ~ 2026-07-25 — PI0.5 validation & root-cause diagnosis

### Added
- `models/pi0/server_v2.py` — PI0.5 NPU inference server adapted to lerobot framework (inference + closed-loop run through).
- `scripts/eval_pi05_task0.py`, `eval_pi05_spatial.py`, `run_pi05_spatial.sh`, `i2_real_obs_infer.py`, `gpu_infer_compare.py` (initial), `gpu_http_proxy.sh`.
- `docs/PI05_TRACKING.md` + `docs/PI05_RECORD.md` — PI0.5 tracking & issue records (P4.1–P4.28).
- Isolated conda env `lerobot-pi05` (torch 2.7.1 + torch_npu 2.7.1.post2 + lerobot fc296548).

### Diagnosed
- PI0.5 spatial closed-loop 0% (100 eps): root cause converged to "model inference itself outputs wrong direction" (fixed negative-z delta_pos per chunk). 8 hypotheses disproven (ckpt mismatch / source version / abs-rel / sign convention / bf16 / chunk accumulation / official-script compare / API difference); 5 fixes verified effective but not root cause (P4.8 ori6d perm+signs, F1 chunk=2, F4 fp32, P4.5v2 proprio, F6 use_peft).
- Remaining directions D/G (GPU vs NPU inference comparison) blocked by outbound network; manual handover packages prepared.

## 2026-07-17 — Public release preparation

### Added
- `LICENSE` (Apache-2.0), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.md`, `CHANGELOG.md`.
- `.gitignore` excluding model weights, videos, logs, `__pycache__`, editor dirs.
- `docs/LIBERO_NPU_MIGRATION.md` — unified project description.
- `docs/LIBERO_NPU_RECORD.md` — issue/solution record.

### Changed
- Replaced all hard-coded absolute paths with environment variables (`$PROJECT_ROOT`, `$MODEL_PATH`, `$X_VLA_ROOT`, `$OPENVLA_ROOT`, `$CKPT_BASE`).
- Replaced `hf-mirror.com` with official `huggingface.co` in `setup_env.sh`.
- Removed cloud-provider proxy hard-coding.
- Renamed `test_openvla闭环_v2.py` → `eval_spatial_task0_v2.py`; `test_openvla闭环_v3_optimized.py` → `eval_spatial_task0_v3_optimized.py`.

### Removed
- Legacy `test_openvla闭环.py` (pre-root-cause version).

## 2026-07-17 — OpenVLA spatial validation

### Fixed
- **Root cause**: OpenVLA outputs delta actions but X-VLA client defaults to `act_type="abs"`, forcing `controller.use_delta=False`. Delta pos was misinterpreted as absolute target coordinates → robot pulled out of workspace → 0% success. Fix: use `act_type="rel"` so `env.step` keeps default `use_delta=True` (matches official `run_libero_eval.py:228`).

### Performance
- `server_v2.py`: pre-compute crop params, fix seed once, patch `json_numpy` once → single-step inference 1.5s → 0.45s (3.3x).

### Validated
- OpenVLA spatial suite: 76.0% (38/50 rollouts, seed 42) vs official 84.7% ± 0.9%.

## 2026-07-13 ~ 2026-07-15 — X-VLA validation

### Validated
- X-VLA on Ascend 910B4 NPU, 5 seeds × 4 suites: spatial 0.90 / goal 0.99 / object 1.00 / long 0.94 — consistent across seeds, matches paper baseline.

## 2026-07-02 — Project start

### Added
- OSMesa software-rendering environment.
- robosuite patches for headless render.
- 6 VLA model server skeletons (xvla/openvla/pi0/smolvla/act/diffusion_policy).
- Idempotent `apply_patches.py`.
