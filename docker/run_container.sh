#!/bin/bash
set -e

# WORKSPACE: local workspace directory, mounted as the container HOME
#
# Assets are not mounted for you. Add your own mount below, for example:
#   -v /path/to/robo_orchard_sim_assets:/assets:rw \
# then set ORCHARD_ASSET and NV_ASSET_ROOT_DIR inside the container.
: "${WORKSPACE:?Please set WORKSPACE}"

CONTAINER_NAME="${CONTAINER_NAME:-robo_orchard_sim}"
CACHE_DIR="${WORKSPACE}/.cache"
IMAGE="horizonrobotics/robo_orchard_sim:ubuntu22.04-py3.11-cuda12.8-isaacsim5.1-isaac_lab-v2.3.2-v0.1"

mkdir -p "${CACHE_DIR}"

docker run -itd --gpus all \
    --network=host \
    --cap-add=ALL \
    --name "${CONTAINER_NAME}" \
    -u "$(id -u):$(id -g)" \
    -e HOME \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video,display,ngx \
    -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    -w "${HOME}/" \
    -v /etc/vulkan/icd.d:/etc/vulkan/icd.d:ro \
    -v /etc/vulkan/implicit_layer.d:/etc/vulkan/implicit_layer.d:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${WORKSPACE}:/home/users/${USER}:rw" \
    -v "${CACHE_DIR}:/home/linuxbrew/.cache:rw" \
    "${IMAGE}"

echo "docker exec -it ${CONTAINER_NAME} bash"
