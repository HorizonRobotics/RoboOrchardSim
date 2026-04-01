# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Patch installed isaaclab experience (.kit) files for offline CI use.

The stock ``isaaclab.python.kit`` experience file declares dependencies on
``omni.kit.menu.edit`` and other UI extensions that are only resolvable via
the NVIDIA online extension registry.  In a CI environment that has no
internet access to those registries, Kit's dependency solver fails and the
whole SimulationApp fails to start.

This script overwrites the affected experience files with versions from the
``patches/isaaclab/<version>/apps/`` directory in this repository, which
comment out the problematic online-only dependencies.
"""

import functools
import os
import re
import shutil
import subprocess
import sys


@functools.lru_cache(maxsize=None)
def get_package_version(package_name: str) -> str | None:
    try:
        reqs = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", package_name]
        )
    except subprocess.CalledProcessError:
        return None
    reg_find_res = re.search(r"Version: (.+)", reqs.decode("utf-8"))
    if reg_find_res is None:
        raise ValueError(
            f"Cannot get the version of package {package_name}. "
            f"pip show result: {reqs!r}"
        )
    return reg_find_res.group(1)


def patch_isaaclab() -> None:
    import isaaclab

    isaaclab_folder = isaaclab.__path__[0]  # type: ignore

    isaaclab_version = get_package_version("isaaclab")
    if isaaclab_version is None:
        raise ValueError("Cannot get the version of isaaclab.")

    proj_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_folder = os.path.join(
        proj_folder, "patches", "isaaclab", f"v{isaaclab_version}"
    )
    if not os.path.exists(src_folder):
        print(
            f"[patch_isaac] No patch folder found for "
            f"isaaclab v{isaaclab_version} "
            f"at {src_folder}, skipping."
        )
        return

    print(
        f"[patch_isaac] Applying patches from {src_folder} "
        f"to {isaaclab_folder}"
    )
    for root, _, files in os.walk(src_folder):
        for file in files:
            src_file = os.path.join(root, file)
            rel_path = os.path.relpath(src_file, src_folder)
            dst_file = os.path.join(isaaclab_folder, rel_path)
            if not os.path.exists(dst_file):
                print(
                    f"[patch_isaac] WARNING: target file {rel_path} "
                    f"not found in isaaclab installation, skipping."
                )
                continue
            print(f"[patch_isaac] Patching {rel_path}")
            shutil.copyfile(src_file, dst_file)

    print("[patch_isaac] Done.")


if __name__ == "__main__":
    patch_isaaclab()
