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


import os
import subprocess
import warnings

from setuptools import find_packages, setup

PROJECT_NAME = "robo_orchard_sim"
PYTHON_BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def repo_path(*path_parts) -> str:
    return os.path.join(PYTHON_BASE_DIR, *path_parts)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf8") as fp:
        return fp.read()


def get_version() -> str:
    version_path = repo_path("VERSION")
    return read_text(version_path).strip()


def get_packages() -> list[str]:
    return find_packages(where=PYTHON_BASE_DIR, include=[f"{PROJECT_NAME}*"])


def write_git_hash() -> None:
    hash_file_name = repo_path(PROJECT_NAME, "__git_hash__.py")
    try:
        repo_git_hash = subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%h"],
            cwd=PYTHON_BASE_DIR,
        ).decode()
    except Exception:
        warnings.warn(
            "Failed to get git hash, using 'unknown' as placeholder",
            UserWarning,
            stacklevel=2,
        )
        repo_git_hash = "unknown"

    with open(hash_file_name, "w", encoding="utf8") as git_hash_file:
        git_hash_file.write('__git_hash_str = "' + repo_git_hash + '"\n')


if __name__ == "__main__":
    write_git_hash()
    setup(
        version=get_version(),
        include_package_data=True,
        packages=get_packages(),
    )
