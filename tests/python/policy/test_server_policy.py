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

from __future__ import annotations
import importlib
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load_factory_module():
    return importlib.import_module("robo_orchard_sim.policy.factory")


def load_server_module():
    return importlib.import_module("robo_orchard_sim.policy.server")


class Sensor:
    def __init__(
        self,
        sensor_data: torch.Tensor,
        intrinsic_matrices: torch.Tensor | None = None,
        pose=None,
    ) -> None:
        self.sensor_data = sensor_data
        self.intrinsic_matrices = intrinsic_matrices
        self.pose = pose


class Pose:
    def __init__(self, xyz: torch.Tensor, quat: torch.Tensor) -> None:
        self.xyz = xyz
        self.quat = quat


def build_public_test_observation() -> dict:
    return {
        "/camera": {
            "left_hand_camera_term": {
                "rgb": Sensor(
                    torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
                    intrinsic_matrices=torch.eye(3).unsqueeze(0),
                    pose=Pose(
                        xyz=torch.tensor([[1.0, 2.0, 3.0]]),
                        quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    ),
                ),
                "depth": Sensor(torch.ones((1, 2, 2, 1))),
            }
        },
        "/robot": {
            "left_joint_position": torch.ones((1, 7)),
            "right_joint_position": torch.ones((1, 7)),
        },
    }


def install_fake_websocket_client(monkeypatch, fake_ws: Mock) -> Mock:
    connect = Mock(return_value=fake_ws)
    client_module = types.ModuleType("websockets.sync.client")
    client_module.connect = connect
    sync_module = types.ModuleType("websockets.sync")
    sync_module.client = client_module
    websockets_module = types.ModuleType("websockets")
    websockets_module.sync = sync_module
    monkeypatch.setitem(sys.modules, "websockets", websockets_module)
    monkeypatch.setitem(sys.modules, "websockets.sync", sync_module)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", client_module)
    return connect


def test_policy_server_module_given_help_flag_exits_successfully() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "robo_orchard_sim.policy.server", "--help"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Policy WebSocket Server" in result.stdout
    assert "--model-type" in result.stdout
    assert "--model-yaml" in result.stdout


def test_create_policy_from_model_cfg_given_server_policy_returns_cfg():
    factory_module = load_factory_module()
    policy_cfg = factory_module.create_policy_from_model_cfg(
        {
            "policy": "server",
            "host": "127.0.0.1",
            "port": 9000,
            "logging_tag": "eval-remote",
        }
    )

    assert policy_cfg.host == "127.0.0.1"
    assert policy_cfg.port == 9000
    assert policy_cfg.logging_tag == "eval-remote"


def test_policy_client_request_action_given_camera_obs_returns_actions(
    monkeypatch,
) -> None:
    fake_ws = Mock()
    fake_ws.recv.return_value = json.dumps(
        {
            "actions": {
                "left_robot_joint_position": {
                    "__tensor__": {
                        "dtype": "float32",
                        "shape": [1, 6],
                        "data": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    }
                }
            },
            "logging_tag": "remote-eval",
        }
    )
    connect = install_fake_websocket_client(monkeypatch, fake_ws)
    server_module = load_server_module()
    client = server_module.PolicyClient(host="127.0.0.1", port=8765)

    actions = client.request_action(build_public_test_observation())

    connect.assert_called_once()
    fake_ws.send.assert_called_once()
    request = json.loads(fake_ws.send.call_args.args[0])
    assert request["type"] == "act"
    assert request["obs_data"]["cameras"]["left_hand_camera_term"]["rgb"][
        "shape"
    ] == [1, 2, 2, 3]
    assert request["obs_data"]["robot"]["left_joint_position"]["__tensor__"][
        "shape"
    ] == [1, 7]
    assert torch.equal(
        actions["left_robot_joint_position"].cpu(), torch.zeros((1, 6))
    )
    assert client.logging_tag == "remote-eval"


def test_server_policy_reset_given_remote_client_sends_reset_request(
    monkeypatch,
) -> None:
    fake_ws = Mock()
    fake_ws.recv.return_value = json.dumps(
        {
            "ok": True,
            "logging_tag": "remote-eval",
        }
    )
    connect = install_fake_websocket_client(monkeypatch, fake_ws)
    server_module = load_server_module()
    policy = server_module.ServerPolicy(
        server_module.ServerPolicyCfg(host="127.0.0.1", port=8765)
    )

    policy.reset()

    connect.assert_called_once()
    fake_ws.send.assert_called_once()
    request = json.loads(fake_ws.send.call_args.args[0])
    assert request["type"] == "reset"
    assert policy.logging_tag == "remote-eval"
