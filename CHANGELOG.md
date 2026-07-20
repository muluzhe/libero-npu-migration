# Changelog

All notable changes to this project are documented here. Dates are local time. Format loosely follows Keep a Changelog.

## [Unreleased]

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
