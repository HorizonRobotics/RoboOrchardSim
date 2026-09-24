# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
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

"""Verify editable upgrades and live source changes in isolation."""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import venv
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EditableInstallation:
    python: Path
    checkout: Path
    source: Path
    outside: Path
    environ: dict[str, str]

    def run(self, *args: str, cwd: Path | None = None) -> str:
        result = subprocess.run(
            [str(self.python), *args],
            cwd=self.checkout if cwd is None else cwd,
            env=self.environ,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode:
            pytest.fail(result.stdout + result.stderr)
        return result.stdout.strip()


@pytest.fixture(scope="module")
def editable_installation(tmp_path_factory):
    root = tmp_path_factory.mktemp("editable-server")
    checkout = root / "checkout"
    source = checkout / "robo_orchard_server"
    shutil.copytree(
        REPO_ROOT / "robo_orchard_server",
        source,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__"
        ),
    )
    environment = root / "venv"
    # Reuse NumPy/websockets, but install this package only in the disposable
    # venv. Build isolation supplies the declared setuptools version.
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(
        environment
    )
    outside = root / "outside"
    outside.mkdir()
    installation = EditableInstallation(
        python=environment / "bin/python",
        checkout=checkout,
        source=source,
        outside=outside,
        environ={
            **os.environ,
            # Reproduce a checkout with a same-named package directory on
            # sys.path, independently of the shell's existing PYTHONPATH.
            "PYTHONPATH": str(checkout),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        },
    )
    installation.run(
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--ignore-installed",
        str(source),
    )
    installation.run(
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--force-reinstall",
        "--config-settings",
        "editable_mode=strict",
        "-e",
        str(source),
    )
    return installation


@pytest.mark.parametrize("directory", ["checkout", "source", "outside"])
def test_editable_upgrade_same_named_directory_imports_current_source(
    editable_installation, directory
):
    installation = editable_installation
    locations = {
        "checkout": installation.checkout,
        "source": installation.source,
        "outside": installation.outside,
    }
    result = installation.run(
        "-c",
        """
import inspect
import json
from pathlib import Path
from robo_orchard_server import server
from robo_orchard_server.server import PolicyFactory, PolicyWebsocketServer
from robo_orchard_server.policy.holobrain import main

print(json.dumps({
    "server": str(Path(inspect.getfile(server)).resolve()),
    "main": str(Path(inspect.getfile(main)).resolve()),
}))
""",
        cwd=locations[directory],
    )
    assert json.loads(result) == {
        "server": str(
            installation.source / "robo_orchard_server/server/__init__.py"
        ),
        "main": str(
            installation.source
            / "robo_orchard_server/policy/holobrain/main.py"
        ),
    }


def test_editable_install_source_change_takes_effect_without_reinstall(
    editable_installation,
):
    installation = editable_installation
    source = installation.source / "robo_orchard_server/server/__init__.py"
    original = source.read_text()
    observed = []
    try:
        for value in ("before", "after-update"):
            source.write_text(original + f"\nEDITABLE_MARKER = {value!r}\n")
            observed.append(
                installation.run(
                    "-c",
                    "from robo_orchard_server.server import EDITABLE_MARKER; "
                    "print(EDITABLE_MARKER)",
                    cwd=installation.outside,
                )
            )
    finally:
        source.write_text(original)
    assert observed == ["before", "after-update"]


def test_editable_holobrain_entrypoint_help_uses_current_module(
    editable_installation,
):
    output = editable_installation.run(
        "-m",
        "robo_orchard_server.policy.holobrain.main",
        "--help",
        cwd=editable_installation.outside,
    )
    assert "--model-dir" in output


def test_holobrain_launcher_conflicting_top_level_packages_shows_help(
    editable_installation, tmp_path
):
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "__init__.py").write_text(
        'raise RuntimeError("Unrelated top-level policy imported")\n'
    )
    (tmp_path / "server.py").write_text(
        'raise RuntimeError("Unrelated top-level server imported")\n'
    )
    installation = editable_installation
    script = (
        installation.source
        / "robo_orchard_server/policy/holobrain/start_holobrain.sh"
    )
    result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=tmp_path,
        env={
            **installation.environ,
            "PATH": (
                str(installation.python.parent)
                + os.pathsep
                + os.environ["PATH"]
            ),
        },
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert "--model-dir" in result.stdout
