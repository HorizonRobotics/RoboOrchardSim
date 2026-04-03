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

Installation consists of four steps:

1. **Prepare assets** — download simulation data from Hugging Face
2. **Set up environment** — pull/build the Docker image, or install prerequisites locally
3. **Launch container** — start with GPU and X11 forwarding, then register the Vulkan ICD
4. **Install package** — install `robo_orchard_sim` in editable mode and pin the protobuf version

#### Prepare Assets

Download simulation assets from Hugging Face repository
`HorizonRobotics/sim_task_suite_assets` to a host directory:

```bash
export ASSETS_DIR=/absolute/path/to/sim_task_suite_assets
mkdir -p ${ASSETS_DIR}
python3 -m pip install -U "huggingface_hub[cli]"
# Login first
huggingface-cli login
huggingface-cli download HorizonRobotics/sim_task_suite_assets \
  --repo-type dataset \
  --local-dir ${ASSETS_DIR}
```

#### Choose One Setup Path

Use either **Prerequisites (local installation)** or **Docker**
depending on your environment.

#### Prerequisites (Local Installation)

- Python 3.10
- Access to the package sources required by `isaacsim`, `isaaclab`, and
  `robo_orchard_core`

#### Docker

This directory contains the Dockerfile and usage notes for the
`robo_orchard_sim` image.

##### Software Stack

- Ubuntu 22.04
- CUDA 11.8
- Python 3.10
- GCC 11.4
- PyTorch 2.5.1 + cu118
- Isaac Sim 4.5.0
- Isaac Lab 2.0.2
- cuRobo

##### Option 1 (Recommended): Pull From Docker Hub

Pull the prebuilt image from Docker Hub:

```bash
docker pull horizonrobotics/robo_orchard_sim:cuda11.8-ubuntu22.04-py3.10-isaacsim4.5.0-isaaclab2.0.2-curobo-gui
```

##### Option 2: Build From Dockerfile

Build the image locally from the Dockerfile:

```bash
docker build \
  --target full \
  -t hub.hobot.cc/auto/robo_orchard_sim:cuda11.8-ubuntu22.04-py3.10-isaacsim4.5.0-isaaclab2.0.2-curobo-gui \
  -f Dockerfile .
```

##### Option 3: Load From TAR

If you receive a prebuilt TAR package, load it into the local Docker daemon:

```bash
docker load -i robo_orchard_sim_cuda11.8_ubuntu22.04_py3.10_isaacsim4.5.0_isaaclab2.0.2_curobo_gui.tar
```

##### Run With GUI

On a machine with an NVIDIA driver and X11 display available:

Set the required environment variables and run the provided script:

```bash
export CONTAINER_NAME=<your_container_name>
export HOST_WORKSPACE=<path/to/your/workspace>
export ASSETS_DIR=<path/to/sim_task_suite_assets>  # set in "Prepare Assets" above
bash run_container.sh
```

After entering the container, validate the environment.
**From this point onward, all commands in this section should be executed
inside the container.**

```bash
python3 -c "import isaacsim; print('isaacsim ok')"
```

###### Register Vulkan ICD

```bash
mkdir -p /usr/share/vulkan/icd.d
cat > /usr/share/vulkan/icd.d/nvidia_icd.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.194"
    }
}
EOF
```

###### Optional: VSCode/Cursor Compatibility

If you want to attach VSCode/Cursor to the running container, add these
compatibility fixes once inside the container as `root`:

```bash
ln -sf /usr/lib/os-release /etc/os-release

cat >/usr/local/bin/base64 <<'EOF'
#!/bin/sh
if [ "$1" = "-D" ]; then
  shift
  exec /usr/bin/base64 -d "$@"
fi
exec /usr/bin/base64 "$@"
EOF

chmod +x /usr/local/bin/base64
```


##### Notes

- This image is intended for `robo_orchard_sim` distribution, not as a generic
  public base image.
- The image includes Isaac Sim, Isaac Lab, and cuRobo, so users must follow the
  applicable NVIDIA software terms.

After cloning the repository, install the package from the repository root:

```bash
git clone <your-repo-url> robo_orchard_sim
cd robo_orchard_sim
python3 -m pip install -e .
```

You can also install it with the repository `Makefile`:

```bash
make install-editable
```

Reinstall protobuf with the compatible version:

```bash
pip install protobuf==5.29.5
```

### 2. Development Workflow

Install development dependencies and hooks:

```bash
make dev-env
```

Common local development commands:

```bash
make check-lint
make type-check
make test
```

Additional test entry point:

```bash
make test-cluster
```

### 3. Run Examples

#### Run `simple_orchard_env_example.py`

The example below assembles a `place_a2b` orchard environment, serializes the
generated environment config, resets the runtime environment, and steps the
simulation for a few frames.

```bash
python3 examples/manipulation-app/scripts/simple_orchard_env_example.py
```

By default, the script writes the generated config to:

```bash
configs/place_a2b_orchard_env_example.json
```

You can also override runtime parameters such as output path, number of
environments, and simulation frequencies:

```bash
python3 examples/manipulation-app/scripts/simple_orchard_env_example.py \
  --output configs/place_a2b_orchard_env_example.json \
  --num_envs 1 \
  --env_spacing 2.5 \
  --physics_fps 600 \
  --render_fps 30 \
  --action_fps 30
```

#### Run `eval_policy.py`

You can override the evaluation seed, number of episodes, maximum steps, and
output path:

```bash
python3 examples/manipulation-app/scripts/eval_policy.py \
  --task-name place_a2b \
  --seed 0 \
  --episode-num 3 \
  --max-steps 10 \
  --output eval_result/isaac_eval/eval_result.json
```

## License

This project is licensed under the Apache License 2.0. See
[`LICENSE`](LICENSE) for the full license text.
