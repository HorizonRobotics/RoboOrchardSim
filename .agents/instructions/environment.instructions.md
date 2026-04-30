---
description: Load these instructions when tasks depend on the active Python environment, optional extras, external services, hardware, or other runtime-specific conditions.
---

# Environment and Runtime Instructions

## Environment

- Use the active environment unless the task requires a change.
- If the repository root contains `.venv/`, prefer that environment for
  Python-related commands unless repository instructions, task
  requirements, or confirmed runtime dependencies require another
  environment.
- Prefer explicit executables such as `.venv/bin/python` and
  `.venv/bin/pip` over relying on shell activation persistence across
  commands when working in a project-local virtual environment.
- Do not override `HF_HOME` by default. If tests or validation depend on
  pre-downloaded Hugging Face data, models, or datasets under the existing
  `HF_HOME`, preserve it so cached artifacts remain available.
- If Hugging Face cache writes are blocked, prefer redirecting the
  narrowest writable cache path needed by the failing tool, such as
  `HF_DATASETS_CACHE` or `HF_HUB_CACHE`, instead of changing `HF_HOME`.
- Do not assume optional extras, developer tools, or external services are installed.
- Check environment-dependent requirements before running related validation.
- If running repository code fails because a `robo_orchard_*` Python package is missing, first check whether the package exists under `python/` in this repository.
- For a single missing local package, prefer the smallest relevant editable install via `make install-editable-pkg PKG=<package_name>` before changing code or treating it as an external dependency.
- Do not bulk-install all local packages by default; use full `make install-editable` only when the task clearly needs the broader local Python package set.

## Runtime and Reporting

- Do not assume network access, hardware, display servers, or background services are available.
- Treat optional services such as `ray` as unavailable until confirmed.
- If a task materially depends on GPU execution and the sandbox cannot access CUDA or NVIDIA devices, request escalated execution for the smallest command that requires GPU access.
- Treat signals such as `torch.cuda.is_available()` returning `False`, `nvidia-smi` not seeing devices, or CUDA/NVML initialization failures as environment-access issues first, not immediate proof of a code bug.
- Do not escalate for code reading, CPU-only validation, or steps that do not require GPU.
- If validation is blocked by the environment, state what ran, what was unavailable, and the remaining risk.
- If GPU-dependent validation is rerun with escalated permissions, report what failed in the sandbox, what was rerun outside it, and any remaining risk.
