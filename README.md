# RoboOrchardSim

`robo_orchard_sim` is the simulation repository for RoboOrchard. It provides
the simulation-side building blocks used to assemble orchard manipulation
environments, launch Isaac-based applications, and evaluate policies against
task setups used in RoboOrchard workflows.

## Overview

This repository focuses on orchard manipulation simulation on top of the Isaac
Sim / Isaac Lab ecosystem. It packages reusable environment, task, and launch
utilities into a Python package that can be used for local development,
integration, and evaluation.

Key features:

- Environment and task assembly utilities for RoboOrchard simulation workflows
- Isaac application launch helpers for headless and scripted execution
- Example scripts for building orchard environments and running policy
  evaluation
- Development tooling for linting, type checking, and testing

## Quick Start

### 1. Installation

Installation has two steps:

1. **Prepare assets** — download the simulation assets and configure their
   paths.
2. **Setup Environment** — use a local Python virtual environment or the
   recommended Docker image.

#### Step 1: Prepare Assets

Download the `instructmove_v1` branch of
`HorizonRobotics/robo_orchard_sim_assets` from Hugging Face:

```bash
export ORCHARD_ASSET=/absolute/path/to/robo_orchard_sim_assets
mkdir -p "${ORCHARD_ASSET}"
python3 -m pip install -U huggingface_hub
hf download HorizonRobotics/robo_orchard_sim_assets \
  --repo-type dataset \
  --revision instructmove_v1 \
  --local-dir "${ORCHARD_ASSET}"
```

Configure the runtime asset paths:

```bash
export ASSETS_DIR="${ORCHARD_ASSET}/OBJECTS"
export NV_ASSET_ROOT_DIR="${ORCHARD_ASSET}/NVIDIA/Assets/Isaac/4.1"
```

> `NV_ASSET_ROOT_DIR` above matches the NVIDIA asset layout in the downloaded
> dataset. Adjust it if the NVIDIA assets are stored in a different directory.

> `ORCHARD_ASSET` and `NV_ASSET_ROOT_DIR` should be set before you run any
> program.

#### Step 2: Setup Environment

Choose either the local virtual environment or Docker installation path.

##### Option 1: Local Virtual Environment

Local installation requires Python 3.10, an NVIDIA driver compatible with
Isaac Sim 4.5.0, and access to the package indexes used by `isaacsim`,
`isaaclab`, and `robo_orchard_core`.

From the repository root:

```bash
git clone <repo_url>
cd robo_orchard_sim
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install-editable \
  PIP_ARGS="--extra-index-url https://pypi.nvidia.com"
```

The editable installation reads `pyproject.toml` and installs Isaac Sim 4.5.0
and Isaac Lab 2.0.2 automatically.

For a non-editable installation, use `make install` with the same package
index:

```bash
make install PIP_ARGS="--extra-index-url https://pypi.nvidia.com"
```

##### Option 2: Docker (Recommended)

The prebuilt Docker image includes the tested Isaac Sim, Isaac Lab, PyTorch,
CUDA, and cuRobo stack. See the
[Docker installation and usage guide](docker/README.md) for image setup,
asset mounts, GPU and X11 configuration, and container launch instructions.

### 2. Development Workflow

Install development dependencies and hooks:

```bash
make dev-env
```

Common local development commands:

```bash
make auto-format
make test
```

Additional test entry point:

```bash
make test-cluster
```

### 3. Run Examples

#### Run `simple_orchard_env_example.py`

The example below builds the default `place_a2b` task via
`PlaceA2BTaskDefinition.build()`, serializes the generated environment config,
resets the runtime environment, and steps the simulation for a few frames.
For the current implementation, scene and embodiment are resolved from
`place_a2b.yaml`, while task assets are defined in
`PlaceA2BTaskDefinition.build()`.

```bash
python3 examples/manipulation-app/scripts/simple_orchard_env_example.py
```

By default, the script writes the generated config to:

```bash
configs/place_a2b_orchard_env_example.json
```

You can override the output path:

```bash
python3 examples/manipulation-app/scripts/simple_orchard_env_example.py \
  --output configs/place_a2b_orchard_env_example.json
```

#### Run `data_synthesis_example.py`

This example resamples task assets per seed, builds a fresh `OrchardEnv` for
each episode, executes the task atomic action plan, and optionally records the
result as MCAP data.

If `ORCHARD_ASSET_LIBRARY` is not set, pass the asset library explicitly:

```bash
python3 examples/manipulation-app/scripts/data_synthesis_example.py \
  --task place_a2b_easy \
  --asset-root ${ASSETS_DIR} \
  --episodes 3 \
  --seed 0
```

By default, recordings are written under:

```bash
logs/data_synthesis/<task>_<timestamp>/
```

and the per-episode serialized env configs are written under:

```bash
configs/data_synthesis/
```

Useful optional flags:

```bash
python3 examples/manipulation-app/scripts/data_synthesis_example.py \
  --task place_a2b_easy \
  --asset-root ${ASSETS_DIR} \
  --config path/to/task.yaml \
  --max-steps 300 \
  --record-dir logs/my_synthesis \
  --output-config-dir configs/my_synthesis
```

To run the synthesis loop without MCAP recording:

```bash
python3 examples/manipulation-app/scripts/data_synthesis_example.py \
  --task place_a2b_easy \
  --asset-root ${ASSETS_DIR} \
  --disable-recording
```

#### Run `eval_policy.py`

`eval_policy.py` runs multi-task policy evaluation. The entire run —
policy, per-task settings, splits, batch plans — is described by one
eval-config YAML; the CLI only carries runtime knobs (output dir, GPUs,
recording).

```bash
python3 examples/manipulation-app/scripts/eval_policy.py \
  --eval-config examples/manipulation-app/configs/eval_example.yaml \
  --output-dir XXXXX \
  --gpus 0,1,2,3 \
  [--enable-recording]
```

- `--eval-config`: eval-config YAML (`policy` / `defaults` / `tasks`).
- `--output-dir`: top-level output directory; each task writes to
  `<output-dir>/<task>/`, summary to `<output-dir>/summary.json`.
- `--gpus`: comma-separated GPU ids; tasks run one per GPU and queue
  when they exceed cards. Defaults to `CUDA_VISIBLE_DEVICES` or `0`.
- `--enable-recording`: turn on MCAP recording for every task.

See `examples/manipulation-app/configs/eval_example.yaml` for the YAML
schema.

## License

This project is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the full license text.
