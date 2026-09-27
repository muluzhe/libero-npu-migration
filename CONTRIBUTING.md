# Contributing to LIBERO NPU Migration

Thanks for your interest in contributing! This project migrates Vision-Language-Action (VLA) model evaluation to Ascend 910B4 NPU. We welcome bug reports, new model adapters, and reproducibility improvements.

## Ways to contribute

- **Bug reports**: open an issue with reproduction steps (env, command, expected vs actual).
- **New VLA adapters**: add `models/<model>/server.py` following the existing `xvla`/`openvla` pattern. See [docs/ADAPTING_NEW_MODEL.md](docs/ADAPTING_NEW_MODEL.md).
- **Reproducibility**: if you ran a benchmark with different seeds/suites, share the `results.json` (no videos).

## Development workflow

1. Fork the repo and create a feature branch: `git checkout -b feat/your-feature`.
2. Keep changes focused; one concern per PR.
3. **Do not commit** model weights, videos, logs, or any file containing personal info (usernames, absolute paths, internal hostnames).
4. Verify Python syntax: `python -m py_compile <your_file.py>` before submitting.
5. If you add a script, follow naming: `eval_*` for benchmark scripts, `diag_*` for diagnostics, `setup_*` / `run_*` for orchestration.
6. Update the relevant doc under `docs/` and bump `CHANGELOG.md`.

## Code style

- Python 3.12, 4-space indent, UTF-8.
- Chinese comments are preserved; new code prefers English identifiers.
- No hard-coded absolute paths — use environment variables (`$PROJECT_ROOT`, `$MODEL_PATH`, `$X_VLA_ROOT`, `$CKPT_BASE`).
- Match existing comment density; don't narrate obvious code.

## Reporting security / privacy issues

If you find leaked personal info in the repo history, please open a private security advisory rather than a public issue.

## License

By submitting a PR you agree your contribution is licensed under the project's Apache-2.0 license (see `LICENSE`).
