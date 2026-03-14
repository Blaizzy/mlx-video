# Repository Guidelines

## Project Structure & Module Organization
- `mlx_video/` holds the core package. Entry points live in `mlx_video/generate.py` and `mlx_video/generate_av.py`, with utilities in `mlx_video/utils.py` and `mlx_video/postprocess.py`.
- Model code is under `mlx_video/models/ltx/`, including VAE components in `video_vae/` and `audio_vae/`, plus prompt templates in `prompts/`.
- Tests are in `tests/` (e.g., `tests/test_rope.py`, `tests/test_vae_streaming.py`).
- Sample output media lives in `examples/`.

## Build, Test, and Development Commands
- Install (editable): `pip install -e '.[dev]'` or `uv pip install -e '.[dev]'` (installs pytest).
- Run the CLI: `uv run mlx_video.generate --prompt "..." -n 100 --width 768`.
- Module execution: `python -m mlx_video.generate --prompt "..." --height 768 --width 768`.
- Tests: `pytest` from the repo root.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard PEP 8 conventions.
- Prefer descriptive, lowercase module and function names (`generate_av.py`, `decode_with_tiling`).
- Tests follow `test_*.py` naming and `Test*` classes with `test_*` methods.
- No formatter/linter is configured in-repo; keep formatting consistent with existing files.

## Testing Guidelines
- Test framework: pytest (declared in `pyproject.toml` under `dev` extras).
- Keep tests deterministic by seeding MLX where applicable (see `mx.random.seed(42)` patterns).
- Cover shape invariants and numerical stability; use `np.testing.assert_allclose` for tensor checks.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative summaries (e.g., “add tests”, “Enhance video generation…”). Keep the subject concise and action-oriented.
- PRs should include a brief description of changes, testing performed (commands + results), and any relevant sample outputs (e.g., generated `.mp4` or `.gif`).
- If your change alters model outputs or performance, note the expected impact and any new flags.

## Runtime & Environment Notes
- Target environment is macOS on Apple Silicon with Python >= 3.11 and MLX >= 0.22.0.
- Model weights are expected from Hugging Face (default `Lightricks/LTX-2`); ensure access and disk space before runs.
