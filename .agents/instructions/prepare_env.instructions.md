---
description: Load these instructions when preparing a fresh local development environment for the repository root workspace.
---

# Environment Preparation Instructions

## When to Use

- Use this guide only for first-time setup, intentional environment
  rebuilds, or explaining how to prepare the repository from an empty
  Python environment.
- For routine task execution, runtime debugging, or dependency triage inside
  an already prepared environment, use
  `.agents/instructions/environment.instructions.md` instead.

## Baseline Requirements

- Follow the Python version required by the project configuration instead of
  assuming a fixed interpreter version.
- Confirm system-level prerequisites only when the task actually depends on
  them, such as CUDA, ROS, Isaac Sim/Lab, or external services.
- Do not assume optional developer tools are already installed.
- Use `README.md`, `Makefile`, `.env.example`, `pyproject.toml`, and
  `dev/requirements.txt` as the primary setup sources of truth.

## Recommended Setup Flow

- Start from the repository root.
- If creating a project-local virtual environment for repository work,
  prefer:

  ```bash
  python -m venv .venv --system-site-packages
  ```

- Prefer explicit executables such as `.venv/bin/python` and
  `.venv/bin/pip` for direct Python commands instead of relying on shell
  activation persistence across commands.
- Treat a project-local `.venv` plus the smallest relevant local editable
  install as the default local development path unless the task or
  confirmed runtime constraints require another environment.
- Do not create `.env` by default.
- Copy `.env.example` to `.env` only when `make` targets need local
  overrides such as a non-default interpreter, pip frontend, or command
  runner.
- Treat `.env` overrides as optional `make` configuration, not as a
  replacement for the project-local `.venv` flow.
- If using a project-local `.venv`, keep any `.env` overrides that select
  Python, pip, or command runners aligned with that environment unless the
  task explicitly requires another runtime.
- Install repository packages with the smallest relevant scope first:

  ```bash
  make install-editable-pkg PKG=<package_name>
  ```

- Use the broader editable install only when the task clearly needs the
  full local package set:

  ```bash
  make install-editable
  ```

- Install development-only tooling only when needed by the task:

  ```bash
  make dev-env
  ```

## Cache and Data Expectations

- Do not override `HF_HOME` by default.
- If the existing environment relies on pre-downloaded Hugging Face models,
  datasets, or artifacts, preserve the current `HF_HOME` so setup and tests
  can reuse them.
- If a Hugging Face component needs a writable cache path, prefer setting a
  narrower cache variable such as `HF_DATASETS_CACHE` or `HF_HUB_CACHE`
  instead of changing `HF_HOME`.

## Validation

- If using a project-local `.venv`, verify that it resolves to the expected
  interpreter:

  ```bash
  .venv/bin/python -c "import sys; print(sys.executable)"
  ```

- If the task intentionally uses another environment, verify the actual
  interpreter that will run the task before installing or validating.
- After editable install, verify that at least one task-relevant local
  package imports from the local checkout, for example:

  ```bash
  .venv/bin/python -c "import robo_orchard_jobs; print(robo_orchard_jobs.__file__)"
  ```

- If the task targets another local package, replace the example import with
  that package instead of assuming `robo_orchard_jobs`.
- Run the smallest relevant validation for the task after setup instead of
  defaulting to the full test suite.

## Common Pitfalls

- Do not rely on shell activation persistence across commands; prefer
  explicit executables or targeted `.env` overrides.
- Do not treat `.env` overrides as a second environment-management system;
  use them only to steer `make` when the default command wiring is not
  sufficient.
- Do not use `prepare_env` as the default guide for ordinary task execution
  after the environment is already working.
- Do not bulk-install all local packages when the task only needs one or a
  small subset.
- If validation fails because a cache path, shared-memory helper, optional
  service, or system dependency is unavailable, report the environment
  limitation before treating it as a code defect.
