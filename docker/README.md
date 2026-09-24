# Docker Installation and Usage

The Docker workflow is the recommended way to run `robo_orchard_sim`.

## Prerequisites

- Linux with an NVIDIA driver compatible with Isaac Sim 5.1.0
- Docker Engine and NVIDIA Container Toolkit
- The simulation assets described in the
  [root installation guide](../README.md#1-installation)

## Get the Image

```bash
docker pull horizonrobotics/robo_orchard_sim:ubuntu22.04-py3.11-cuda12.8-isaacsim5.1-isaac_lab-v2.3.2-v0.1
```

The image ships Ubuntu 22.04, Python 3.11, CUDA 12.8, PyTorch 2.7.0, Isaac Sim
5.1.0, Isaac Lab 2.3.2, and cuRobo 0.7.6, but not `robo_orchard_sim` itself.

## Run the Container

```bash
export WORKSPACE=/absolute/path/to/your/workspace
bash docker/run_container.sh
docker exec -it robo_orchard_sim bash
```

`WORKSPACE` becomes the container HOME, and `${WORKSPACE}/.cache` is reused as
the Kit cache. Add your own `-v` for the asset directory, then set
`ORCHARD_ASSET` and `NV_ASSET_ROOT_DIR` inside the container as described in the
[root installation guide](../README.md#1-installation).

If you write your own launch script, copy the `docker run` flags from
`run_container.sh` — the Vulkan mounts, `--cap-add=ALL`, and
`NVIDIA_DRIVER_CAPABILITIES` are required for Isaac Sim to render.

## Install and Verify

Inside the container, from your clone of this repository:

```bash
cd robo_orchard_sim
make install-editable
```

## Run With GUI

The container shares the host X11 socket, so prefix the command with the host
`DISPLAY` and the window shows up on the host desktop:

```bash
DISPLAY=:{id} python3 your_isaac_example.py
```

Replace `{id}` with the host display number (`echo $DISPLAY` on the host).
Launcher-based scripts need
`SimpleIsaacAppLauncher(headless=False, virtual_display=False)`.

> Only GPU 0 can present to a display; runs on other GPUs must be headless.
