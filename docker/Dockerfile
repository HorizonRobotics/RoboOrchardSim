# syntax=docker/dockerfile:1.6
# ============================================================
# robo_orchard_sim base image v0.1
# ------------------------------------------------------------
# Purpose: minimal image with IsaacSim 5.1 + IsaacLab 2.3.2 to
# validate the base stack. This image does NOT include
# robo_orchard_sim itself; it is used to run the official
# examples first, then evolves into the full image.
#
# Build:
#   docker build \
#     -f docker/Dockerfile51 \
#     -t robo_orchard_sim:isaacsim5.1.0-isaaclab2.3.2-py3.11-cuda12.8-v0.1 \
#     .
#
# Optional build args:
#   UBUNTU_APT_MIRROR       Ubuntu mirror used for all apt pockets
#                           (default: archive.ubuntu.com)
#   USER_NAME / USER_UID / USER_GID
#                           Local user identity recorded in the image
#   PIP_INDEX_URL          primary pip index (default: pypi.org)
#   PIP_EXTRA_INDEX_URLS   space-separated extra indexes
#                          (default: nvidia + pytorch cu128)
# ============================================================

# FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
FROM docker.hobot.cc/dlp/base:ubuntu2204-gcc11.4-cu128-nccl2277-torch271-erdma

# --- build-time knobs ---------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11
ARG PYTHON_PATCH_VERSION=3.11.13
ARG ISAACSIM_VERSION=5.1.0
ARG ISAACLAB_VERSION=2.3.2
ARG TORCH_VERSION=2.7.0
ARG UBUNTU_APT_MIRROR=http://archive.ubuntu.com/ubuntu

# pip index configuration (override via --build-arg to switch to internal mirror)
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URLS="https://pypi.nvidia.com/simple https://download.pytorch.org/whl/cu128"

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URLS}" \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# --- system packages ----------------------------------------------------------
# libgl / libglx / vulkan / xvfb: required by Kit for offscreen and X11 rendering
# libglib / libsm / libxext / libxrender: common Qt / GUI deps
# build-essential + git: needed for pip packages that build C/CUDA extensions
# locales: silence Kit locale warnings
RUN sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        -e "s|http://security.ubuntu.com/ubuntu|${UBUNTU_APT_MIRROR}|g" \
        /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        wget \
        locales \
        build-essential \
        pkg-config \
        cmake \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        libffi-dev \
        liblzma-dev \
        libgdbm-dev \
        libnss3-dev \
        tk-dev \
        uuid-dev \
        xz-utils \
        libglib2.0-0 \
        libgl1 \
        libglx-mesa0 \
        libegl1 \
        libxext6 \
        libxrender1 \
        libxrandr2 \
        libsm6 \
        libice6 \
        libxi6 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libxcb-cursor0 \
        libxcb-icccm4 \
        libxcb-image0 \
        libxcb-keysyms1 \
        libxcb-randr0 \
        libxcb-render-util0 \
        libxcb-shape0 \
        libxcb-sync1 \
        libxcb-xfixes0 \
        libxcb-xinerama0 \
        libxcb-xkb1 \
        libnss3 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbfile1 \
        libasound2 \
        libgomp1 \
        libvulkan1 \
        vulkan-tools \
        mesa-utils \
        xvfb \
        x11-apps \
        ffmpeg \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# --- Python 3.11 --------------------------------------------------------------
RUN curl -fsSL \
        "https://www.python.org/ftp/python/${PYTHON_PATCH_VERSION}/Python-${PYTHON_PATCH_VERSION}.tgz" \
        -o /tmp/python.tgz \
    && tar -xzf /tmp/python.tgz -C /tmp \
    && cd "/tmp/Python-${PYTHON_PATCH_VERSION}" \
    && ./configure \
        --enable-shared \
        --with-ensurepip=install \
        --with-system-ffi \
    && make -j"$(nproc)" \
    && make altinstall \
    && cd / \
    && ldconfig \
    && rm -rf /tmp/python.tgz "/tmp/Python-${PYTHON_PATCH_VERSION}" \
    && ln -sf "/usr/local/bin/python${PYTHON_VERSION}" /usr/local/bin/python3 \
    && ln -sf "/usr/local/bin/python${PYTHON_VERSION}" /usr/local/bin/python \
    && python3 -m pip install --upgrade pip setuptools wheel

# --- PyTorch (cu128) ----------------------------------------------------------
# torch/torchvision from https://download.pytorch.org/whl/cu128 (via PIP_EXTRA_INDEX_URL)
RUN python3 -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision"

# --- IsaacSim 5.1 -------------------------------------------------------------
# Comes from https://pypi.nvidia.com/simple (via PIP_EXTRA_INDEX_URL)
# extscache is the Kit extension cache; without it Kit tries to download
# extensions from the online registry at runtime, which fails in offline CI.
RUN python3 -m pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}"

# --- IsaacLab 2.3.2 -----------------------------------------------------------
# flatdict 4.0.1 imports pkg_resources from its legacy setup.py. Newer
# setuptools releases remove that module, and wheel 0.47 requires a newer
# packaging version than the one pinned by IsaacSim 5.1.
RUN python3 -m pip install \
        "setuptools==80.9.0" \
        "wheel==0.42.0" \
    && python3 -m pip install --no-build-isolation "flatdict==4.0.1"

# IsaacLab leaves ONNX and Transformers unbounded. Their latest releases
# require newer click/typing_extensions versions than IsaacSim 5.1 permits.
ARG ONNX_VERSION=1.18.0
ARG TRANSFORMERS_VERSION=4.57.6
RUN python3 -m pip install \
        "isaaclab==${ISAACLAB_VERSION}" \
        "onnx==${ONNX_VERSION}" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "click==8.1.7" \
        "typing_extensions==4.12.2" \
    && python3 -m pip check

# --- Apply IsaacLab kit patches -----------------------------------------------
# Stock IsaacLab kit manifests declare online-only extensions and pin extension
# versions that don't match the pip-installed IsaacSim 5.1 (see
# patches/isaaclab/v${ISAACLAB_VERSION}/README.md for the full rationale).
# `scm/patch_isaac.py` reads `pip show isaaclab` at runtime and picks the
# matching patches/isaaclab/v{VERSION}/ folder automatically, so we ship all
# version subdirectories and let the script select the right one.
COPY patches/isaaclab/ /tmp/robo_orchard/patches/isaaclab/
COPY scm/patch_isaac.py /tmp/robo_orchard/scm/patch_isaac.py
RUN cd /tmp/robo_orchard \
    && OMNI_KIT_ACCEPT_EULA=YES python3 scm/patch_isaac.py \
    && rm -rf /tmp/robo_orchard

# The pip distribution's Iray/MDL plugins dynamically load libGLU.so.1.
# Runtime tools provide X11/VNC display access, SSH access, and Git LFS support.
RUN unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        git-lfs \
        libglu1-mesa \
        openssh-server \
        tmux \
        vim \
        x11vnc \
    && mkdir -p /run/sshd \
    && rm -rf /var/lib/apt/lists/*

# --- cuRobo 0.7.6 ------------------------------------------------------------
# Install against the existing Isaac/PyTorch/CUDA stack. The constraints file
# prevents pip from upgrading or downgrading any package already in the image.
# RTX 5090 uses CUDA compute capability 12.0; setting it explicitly is required
# because GPU discovery is unavailable during a normal docker build.
ARG CUROBO_VERSION=0.7.6
ARG CUROBO_COMMIT=2fbffc35225398cf9d5f382804faa9de2608753b
RUN git lfs install --system \
    && for attempt in 1 2 3 4 5; do \
        git -c http.version=HTTP/1.1 clone \
            --branch "v${CUROBO_VERSION}" \
            --depth 1 \
            https://github.com/NVlabs/curobo.git \
            /opt/curobo \
        && break; \
        rm -rf /opt/curobo; \
        sleep 3; \
    done \
    && test -d /opt/curobo/.git \
    && cd /opt/curobo \
    && test "$(git rev-parse HEAD)" = "${CUROBO_COMMIT}" \
    && git lfs pull \
    && python3 -m pip install --no-deps \
        tomli \
        ninja \
        "setuptools_scm==8.3.1" \
    && python3 -m pip freeze > /tmp/curobo-base-constraints.txt \
    && SETUPTOOLS_SCM_PRETEND_VERSION="${CUROBO_VERSION}" \
        TORCH_CUDA_ARCH_LIST=12.0 \
        python3 -m pip install \
        --editable ".[isaacsim]" \
        --no-build-isolation \
        --constraint /tmp/curobo-base-constraints.txt \
    && rm -f /tmp/curobo-base-constraints.txt \
    && python3 -m pip check \
    && python3 - <<'PY'
from importlib.metadata import version

import torch

assert version("nvidia-curobo") == "0.7.6"
assert version("isaacsim") == "5.1.0.0"
assert version("isaaclab") == "2.3.2"
assert torch.__version__ == "2.7.0+cu128"
assert torch.version.cuda == "12.8"

print(f"cuRobo: {version('nvidia-curobo')}")
print(f"Isaac Sim: {version('isaacsim')}")
print(f"IsaacLab: {version('isaaclab')}")
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
PY

# --- RoboOrchard runtime packages for internal only ---------------------------
RUN unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY \
    && export PIP_INDEX_URL=https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple \
    && export PIP_EXTRA_INDEX_URL=https://pypi.hobot.cc/hobot-local/simple \
    && python3 -m pip install \
        "numpy==1.26.0" \
        "opencv-python<4.12" \
        "robo_orchard_core==0.4.0" \
        "mcap-protobuf-support==0.5.4" \
        "foxglove-schemas-protobuf==0.3.0" \
        "pyarrow==25.0.0" \
    && python3 -m pip install --no-deps "robo_orchard_schemas==0.2.0"

# --- NVIDIA EGL vendor config -------------------------------------------------
# nvidia-container-toolkit injects the Vulkan ICD but not this glvnd EGL vendor
# file, so libglvnd only sees mesa and NVIDIA falls back to an X11-only path --
# headless runs then fail with VK_ERROR_INCOMPATIBLE_DRIVER unless Xvfb is up.
# Contents are driver-version independent, so shipping our own copy is safe.
RUN mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' \
        '{' \
        '    "file_format_version" : "1.0.0",' \
        '    "ICD" : { "library_path" : "libEGL_nvidia.so.0" }' \
        '}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# --- runtime user -------------------------------------------------------------
# Create the host identity for local bind mounts, but keep root as the final
# runtime user so AIDI-managed workspace and log mounts remain writable.
RUN unset http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY \
    && apt-get update \
    && apt-get install -y --no-install-recommends libnss-wrapper \
    && rm -rf /var/lib/apt/lists/*

ARG USER_NAME=developer
ARG USER_UID=1000
ARG USER_GID=1000
RUN mkdir -p \
        "/home/users/${USER_NAME}" \
        /usr/local/lib/python3.11/site-packages/isaacsim/kit/cache \
        /usr/local/lib/python3.11/site-packages/isaacsim/kit/data \
        /usr/local/lib/python3.11/site-packages/isaacsim/kit/logs \
    && printf '%s:x:%s:\n' \
        "${USER_NAME}" "${USER_GID}" >> /etc/group \
    && printf '%s:x:%s:%s:%s:%s:%s\n' \
        "${USER_NAME}" \
        "${USER_UID}" \
        "${USER_GID}" \
        "${USER_NAME}" \
        "/home/users/${USER_NAME}" \
        /bin/bash >> /etc/passwd \
    && pwconv \
    && chown "${USER_UID}:${USER_GID}" \
        "/home/users/${USER_NAME}" \
    && chmod -R a+rwX \
        /usr/local/lib/python3.11/site-packages/isaacsim/kit/cache \
        /usr/local/lib/python3.11/site-packages/isaacsim/kit/data \
        /usr/local/lib/python3.11/site-packages/isaacsim/kit/logs

# --- sanity check smoke test --------------------------------------------------
COPY <<'EOF' /opt/robo_orchard_sim/smoke_import.py
"""Minimal smoke test: import isaacsim / isaaclab / torch and print versions."""
import sys

import torch  # noqa: E402


def main() -> int:
    print(f"python: {sys.version}")
    print(f"torch: {torch.__version__} (cuda={torch.cuda.is_available()})")

    import isaacsim  # noqa: F401, E402
    print(f"isaacsim: OK (module={isaacsim.__file__})")

    import isaaclab  # noqa: F401, E402
    print(f"isaaclab: OK (module={isaaclab.__file__})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF

# --- workdir & default shell --------------------------------------------------
COPY docker/container-entrypoint.sh /usr/local/bin/container-entrypoint
RUN chmod 0755 /usr/local/bin/container-entrypoint

ENV HOME=/home/users/${USER_NAME} \
    OMNI_KIT_ACCEPT_EULA=YES \
    __NV_PRIME_RENDER_OFFLOAD=1 \
    __GLX_VENDOR_LIBRARY_NAME=nvidia \
    __VK_LAYER_NV_optimus=NVIDIA_only
USER root
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/container-entrypoint"]
CMD ["/bin/bash"]
