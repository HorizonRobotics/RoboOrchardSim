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

"""Build/install the package and run its CLI in an isolated environment."""

from __future__ import annotations
import json
import shutil
import signal
import socket
import subprocess
import sys
import time
import venv
from contextlib import ExitStack

import pytest
import torch
from test_server_interop import REPO_ROOT, PolicyClient, build_observation

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand


@pytest.fixture(scope="module")
def installed_package(tmp_path_factory):
    root = tmp_path_factory.mktemp("installed-server")
    source = root / "source"
    shutil.copytree(
        REPO_ROOT / "robo_orchard_server",
        source,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__"
        ),
    )
    artifacts = root / "dist"
    # Build defaults to creating an sdist, then a wheel from that sdist.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            str(source),
            "--outdir",
            str(artifacts),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    (wheel,) = artifacts.glob("*.whl")
    environment = root / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin/python"
    # Ignore the parent PYTHONPATH so pip cannot mistake the source tree
    # for a distribution already installed in this clean environment.
    subprocess.run(
        [str(python), "-I", "-m", "pip", "install", str(wheel)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    return python


def test_package_built_distribution_installs_with_only_runtime_dependencies(
    installed_package, tmp_path
):
    result = subprocess.run(
        [
            str(installed_package),
            "-I",
            "-c",
            """
import importlib.util
import json
from importlib.metadata import metadata, version
from robo_orchard_server.server import (
    BasePolicy, JointAction, Observation, PolicyFactory,
    PolicyWebsocketServer,
)
from robo_orchard_server.server.policy import BasePolicy, PolicyFactory

package = metadata("robo_orchard_server")
print(json.dumps({
    "version": version("robo_orchard_server"),
    "requires_python": package["Requires-Python"],
    "dependencies": sorted(package.get_all("Requires-Dist")),
    "license": package["License-Expression"],
    "license_files": package.get_all("License-File"),
    "unwanted_imports": [name for name in (
        "robo_orchard_sim", "robo_orchard_core", "torch", "pydantic"
    ) if importlib.util.find_spec(name) is not None],
}))
""",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert json.loads(result.stdout) == {
        "version": "0.1.0",
        "requires_python": ">=3.11",
        "dependencies": ["numpy", "websockets>=12"],
        "license": "Apache-2.0",
        "license_files": ["LICENSE"],
        "unwanted_imports": [],
    }


@pytest.fixture(params=["fixed", "stateful", "dummy"])
def installed_cli(installed_package, tmp_path, monkeypatch, request):
    # The client lives in the sim environment; only its GPU boundary is mocked.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if request.param == "dummy":
        entrypoint = ["-m", "robo_orchard_server.policy.dummy.main"]
    else:
        filename = (
            "stateful_policy_server_example.py"
            if request.param == "stateful"
            else "policy_server_example.py"
        )
        script = tmp_path / filename
        shutil.copyfile(REPO_ROOT / "examples/server" / filename, script)
        entrypoint = [str(script)]
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                str(installed_package),
                "-I",
                *entrypoint,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--scale" if request.param == "stateful" else "--value",
                "0.125",
            ],
            cwd=tmp_path,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 15
            while "Listening on" not in log_path.read_text():
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail(
                        f"Server failed to start: {log_path.read_text()}"
                    )
                time.sleep(0.02)
            yield process, port, request.param
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_cli_installed_package_interoperates_without_repo_on_path(
    installed_cli,
):
    _, port, policy_kind = installed_cli
    observation, _ = build_observation("dualarm_piperx", batch=2)
    with PolicyClient(host="127.0.0.1", port=port) as client:
        client.reset_remote_policy()
        sequence = client.request_action_sequence(observation)
    expected = [0.125, 0.25, 0.375] if policy_kind == "stateful" else [0.125]
    values = []
    for action in sequence:
        if not isinstance(action, UnifiedJointCommand):
            pytest.fail("Expected a named joint command")
        values.append(action.values)
    torch.testing.assert_close(
        torch.stack(values),
        torch.stack([torch.full((2, 16), value) for value in expected]),
    )


@pytest.mark.parametrize("client_count", [1, 3])
def test_cli_interrupt_with_idle_clients_exits(installed_cli, client_count):
    process, port, _ = installed_cli
    with ExitStack() as stack:
        for _ in range(client_count):
            client = stack.enter_context(
                PolicyClient(host="127.0.0.1", port=port)
            )
            client.reset_remote_policy()
        process.send_signal(signal.SIGINT)
        try:
            result = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pytest.fail("Ctrl-C must stop the server with idle clients")
    assert result == 0
