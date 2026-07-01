# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `robo_orchard_sim/`: `orchard_env/` builds environments,
`benchmark/` registers tasks, `models/` defines assets and robots,
`envs/managers/` handles runtime managers, and `viz/` contains display helpers.
Tests live in `tests/python/`, examples in `examples/`, developer scripts in
`scm/`, and longer internal notes in `agent_docs/`.

## Build, Test, and Development Commands

Use the top-level `Makefile` by default: `make install-editable` installs the
package, `make dev-env` installs dev tools and `pre-commit`,
`make auto-format` runs Ruff fix + format, `make check-lint` runs Ruff,
`make type-check` runs `pyright`, `make test` runs the serial pytest suite,
and `make test-cluster` runs the `xdist` suite. For quick sanity checks, run
`python3 examples/manipulation-app/scripts/simple_orchard_env_example.py`.

## Coding Style & Naming Conventions

Target Python 3.10 with 4-space indentation and a 79-character line limit.
Ruff enforces import sorting, quote style, and Google-style docstrings; use
absolute imports because relative imports are banned. Follow existing naming
patterns: `snake_case.py` modules, `test_*.py` test files, and descriptive
config wrapper names such as `*_cfg.py`. In Python development, avoid
`getattr` where practical; prefer explicit attribute access or structured
mappings, and only use dynamic attribute lookup when there is a clear need.
Keep the repository copyright header on Python files.

## Testing Guidelines

Pytest is the test runner; common plugins include `pytest-xdist`,
`pytest-asyncio`, and `pytest-cov`. Put new tests near the affected code path
under `tests/python/`, and name files `test_<feature>.py`. Run focused tests
with `pytest -c tests/pytest.ini tests/python/...`. The full suite expects
Isaac-related assets and environment variables such as `ORCHARD_CI` and
`NV_ASSET_ROOT_DIR`, which are set in `tests/Makefile`.

For tests that start an Isaac app, make sure
`tests/python/robo_orchard_sim_ut/conftest.py` is in scope, because it creates
the session-level `SimpleIsaacAppLauncher(enable_cameras=True,
virtual_display=True)` fixture used by the Isaac test suite.

When running example scripts that start Isaac with
`launcher = SimpleIsaacAppLauncher(enable_cameras=True, virtual_display=True)`,
prefer the built-in virtual display path. If you must run with
`virtual_display=False`, start the script with an explicit X display, for
example `DISPLAY=:<id> python3 examples/...`, where `<id>` is the active Xvfb
display number discovered from the running Xvfb process.

## Test Quality Standards

These rules apply to every test added or modified in this repository. They
exist so that tests survive internal refactors and give clear failure messages.

### Behavioral correctness

- Assert only public API and observable output, never private attributes or
  internal state.
- A test must still be valid after an internal refactor that preserves
  externally visible behavior.
- Do not assert call counts or call order of internal helpers.

### Structural soundness

- One test covers one behavior. Use parameterization for multiple inputs that
  exercise the same behavior, not separate test functions.
- Each test is fully self-contained. Setup must not rely on another test
  having run first.
- Mock at the integration boundary (filesystem, network, simulation engine,
  GPU). Do not mock the module under test.

### Naming

Every test function must follow this pattern:

```
test_<feature_or_function>_<scenario>_<expected_result>
```

Examples:

```python
def test_reward_calculator_empty_env_returns_zero():
def test_robot_arm_joint_limit_exceeded_raises_value_error():
def test_task_registry_duplicate_name_logs_warning():
```

Names like `test_1`, `test_it`, or `test_<feature>` alone are not acceptable.

### Red flags

Any of the following requires an immediate fix before merging:

- The test body reads an attribute that starts with `_`
- The test has three or more `assert` statements on unrelated values
- The test passes without any production code written yet
- The test fails when run in isolation but passes when run with the full suite

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit style, for example `feat: ...` and
`chore(docs): ...`. Prefer `type(scope): short summary` when a scope is useful.
Pull requests should describe user-visible impact, list the commands you ran,
and link related issues or design notes. Include screenshots only for
visualization-facing changes.
