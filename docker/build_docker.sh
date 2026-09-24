#!/usr/bin/env bash
set -euo pipefail

docker build \
    -f docker/Dockerfile \
    -t hub.hobot.cc/auto/robot_lab_base:ubuntu22.04-py3.11-cuda12.8-isaacsim5.1-isaac_lab-v2.3.2-internal-v0.1 \
    --build-arg UBUNTU_APT_MIRROR=http://mirrors.aliyun.com/ubuntu \
    --build-arg USER_NAME="$(id -un)" \
    --build-arg USER_UID="$(id -u)" \
    --build-arg USER_GID="$(id -g)" \
    .
