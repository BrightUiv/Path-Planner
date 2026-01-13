# Repository Guidelines

## Project Structure & Module Organization
- `genesis/`: main Python package (engine, solvers/entities, visualization, utilities).
- `tests/`: `pytest` test suite and fixtures (`tests/conftest.py`).
- `examples/`: runnable demos grouped by domain (e.g. `examples/drone/`, `examples/rendering/`).
- `docker/`: Dockerfiles and GPU runtime configs.
- `doc/`: documentation git submodule (not always checked out).
- Assets/media: `genesis/assets/`, `imgs/`, `videos/`.

## Build, Test, and Development Commands
- Dev install (editable): `python -m pip install -e ".[dev]"`
- After switching branches/HEAD: re-run `python -m pip install -e ".[dev]"` to refresh deps/entrypoints.
- Run tests (required gate): `pytest -v --forked -m required ./tests`
- Run tests (full suite): `pytest ./tests` (configured to run in parallel by default).
- Format on commit: `python -m pip install pre-commit && pre-commit install`.
- CLI utilities: `gs view path/to/robot.urdf` and `gs animate "renders/*.png" --fps 30`.
- Docker build: `docker build -t genesis -f docker/Dockerfile docker`

## Coding Style & Naming Conventions
- Python: `>=3.10,<3.14`; 4-space indentation.
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Formatting: Black (line length 120). The Black hook is defined in `.pre-commit-config.yaml`.
- Avoid editing `genesis/ext/` unless necessary; it contains external/vendor code and is excluded from formatting.

## Testing Guidelines
- Prefer adding/adjusting unit tests in `tests/` alongside the feature/bug fix.
- Use markers to scope runs: `-m "not slow"`, `-m benchmarks`, `-m examples`.

## Commit & Pull Request Guidelines
- Match upstream conventions: use tags like `[BUG FIX]`, `[FEATURE]`, `[MISC]` in PR titles; add `[CHANGING]` for breaking changes.
- PRs should include: what changed, how to reproduce/verify (commands or a minimal script), and relevant logs/screenshots for rendering/GUI changes.

## Submodules & Configuration Tips
- Initialize submodules when needed (docs + external render/mesher deps): `git submodule update --init --recursive`.
