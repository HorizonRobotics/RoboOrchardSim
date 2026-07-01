# Docker Installation and Usage

The Docker workflow is the recommended installation path for
`robo_orchard_sim`. It provides the tested simulation stack while keeping it
isolated from the host Python environment.

## Prerequisites

- Linux with an NVIDIA driver
- Docker Engine
- NVIDIA Container Toolkit
- An X11 display for GUI execution
- The simulation assets described in the
  [root installation guide](../README.md#1-installation)

## Software Stack

- Ubuntu 22.04
- CUDA 11.8
- Python 3.10
- GCC 11.4
- PyTorch 2.5.1 with CUDA 11.8
- Isaac Sim 4.5.0
- Isaac Lab 2.0.2
- cuRobo

## Get the Image

### Pull From Docker Hub

Pull the prebuilt image:

```bash
docker pull horizonrobotics/robo_orchard_sim:cuda11.8-ubuntu22.04-py3.10-isaacsim4.5.0-isaaclab2.0.2-curobo-gui-v1.0
```

## Run With GUI

Run these commands from the repository root on a machine with an NVIDIA driver
and X11 display:

```bash
export CONTAINER_NAME=robo-orchard-sim
export HOST_WORKSPACE=/absolute/path/to/your/robo_orchard_sim
export ASSETS_DIR=/absolute/path/to/sim_task_suite_assets
bash docker/run_container.sh
```

The launcher maps the host asset directory into the container as follows:

```text
ASSETS_DIR (host)
  -> CONTAINER_ASSETS (container, default: /assets)
     -> ORCHARD_ASSET=/assets
     -> NV_ASSET_ROOT_DIR=/assets/NVIDIA/Assets/Isaac/4.1
```

To use a different container mount point, set `CONTAINER_ASSETS` before
running the launcher:

```bash
export CONTAINER_ASSETS=/data/assets
bash docker/run_container.sh
```

The launcher mounts `HOST_WORKSPACE` at `/workspace` and opens an interactive
Bash shell there.

## Configure and Verify the Container

The commands in this section run inside the container.

Install `robo_orchard_sim` from the mounted repository:

```bash
git clone <repo-url>
cd robo_orchard_sim
make install-editable
```

Verify that Isaac Sim is importable:

```bash
python3 -c "import isaacsim; print('isaacsim ok')"
```

### Optional: VSCode/Cursor Compatibility

To attach VSCode or Cursor to the running container, apply these compatibility
fixes once inside the container as `root`:

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

## Notes

- This image is intended for `robo_orchard_sim` distribution, not as a generic
  public base image.
- The image includes Isaac Sim, Isaac Lab, and cuRobo. Users must follow the
  applicable NVIDIA software terms.
