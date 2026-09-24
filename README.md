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

Local installation requires Python 3.11, an NVIDIA driver compatible with
Isaac Sim 5.1.0, and access to the package indexes used by `isaacsim`,
`isaaclab`, and `robo_orchard_core`.

From the repository root:

```bash
git clone <repo_url>
cd robo_orchard_sim
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install PIP_ARGS="--extra-index-url https://pypi.nvidia.com"
```

For an editable installation, use the following command instead:

```bash
make install-editable PIP_ARGS="--extra-index-url https://pypi.nvidia.com"
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


#### Build and Step an Environment

Select a registered task with `--task`. Its default YAML defines the scene,
robot, assets, and task settings; `--config path/to/task.yaml` overrides it.

```bash
python3 examples/manipulation-app/scripts/simple_orchard_env_example.py \
  --task place_a2b \
  --asset-root "${ASSETS_DIR}"
```

This smoke test resets and steps the environment, and saves its config to
`configs/orchard_env_example.json` (`--output` overrides the path).
It does not load asset splits; use synthesis or evaluation for split-based
sampling.

#### Synthesize Data

Sample assets per episode and execute the task's atomic action plan:

```bash
python3 examples/manipulation-app/scripts/data_synthesis_example.py \
  --task place_a2b \
  --asset-root "${ASSETS_DIR}" \
  --episodes 3 \
  --seed 0 \
  --task-save-root logs/data_synthesis
```

Configs and MCAP recordings go to `config/` and `data/` under
`logs/data_synthesis/place_a2b_<timestamp>/`.

- `--config path/to/task.yaml`: override the task YAML.
- `--splits path/to/splits.yaml`: supply the named splits used by the task
  YAML's `split` fields when sampling assets.
- `--disable-recording`: run without MCAP output.


#### Evaluate a Policy

Edit [eval_example.yaml](examples/manipulation-app/configs/eval_example.yaml)
to set the policy, asset/split paths, and tasks. It defaults to a dummy policy;
configure your model for actual evaluation. Each task specifies a registered
`task_type` and optionally a task `yaml`. Set `split_type` to
`seen`, `unseen_instance`, or `unseen_category` to override the task YAML's
split selection; omit it to use the task YAML. `split_type` cannot be combined
with `batch_plan`.

```bash
python3 examples/manipulation-app/scripts/eval_policy.py \
  --eval-config examples/manipulation-app/configs/eval_example.yaml \
  --output-dir eval_result/run_001 \
  --gpus 0
```

Use `--gpus 0,1,2,3` for multiple GPUs; sharding and scheduling are automatic.
Add `--enable-recording --export-video` for MCAP recordings and MP4 previews.

Results go to `eval_result/run_001/`: `summary.json` contains leaderboard
scores, and each task instance has its own subdirectory.
To serve a model from a separate Python environment, use
[robo_orchard_server](robo_orchard_server/README.md).

## License

This project is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the full license text.
