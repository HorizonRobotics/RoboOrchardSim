#!/bin/bash
set -e

# CONTAINER_NAME:   name for the container
# HOST_WORKSPACE:   local workspace directory mounted into the container (set above)
# ASSETS_DIR:       path to sim_task_suite_assets downloaded in "Prepare Assets" (set above)
# CONTAINER_ASSETS: mount point for assets inside the container (default: /assets)
: "${CONTAINER_NAME:?Please set CONTAINER_NAME}"
: "${HOST_WORKSPACE:?Please set HOST_WORKSPACE}"
: "${ASSETS_DIR:?Please set ASSETS_DIR}"
CONTAINER_ASSETS="${CONTAINER_ASSETS:-/assets}"

IMAGE="horizonrobotics/robo_orchard_sim:cuda11.8-ubuntu22.04-py3.10-isaacsim4.5.0-isaaclab2.0.2-curobo-gui"

mkdir -p "${HOST_WORKSPACE}"
xhost +local:docker

docker run -it \
  --gpus 'all,"capabilities=compute,utility,graphics,video,display"' \
  --name "${CONTAINER_NAME}" \
  --network host \
  --shm-size=256g \
  -e USER=root \
  -e HOME=/root \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=/root/.Xauthority \
  -e OMNI_KIT_ACCEPT_EULA=YES \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e ORCHARD_ASSET="${CONTAINER_ASSETS}" \
  -e NV_ASSET_ROOT_DIR="${CONTAINER_ASSETS}/NVIDIA/Assets/Isaac/4.1" \
  -w /workspace \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/root/.Xauthority:ro \
  -v $HOME/.cache:/root/.cache \
  -v "${HOST_WORKSPACE}":/workspace:rw \
  -v "${ASSETS_DIR}":"${CONTAINER_ASSETS}":rw \
  "${IMAGE}" \
  /bin/bash
