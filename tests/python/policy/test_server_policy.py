# ruff: noqa: E402
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
import asyncio
import importlib
import json
import struct
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.contracts.policy_binding import (
    CameraBinding,
    CanonicalPolicyInput,
    ManipulatorBinding,
    PolicyBindingSchema,
)
from robo_orchard_sim.policy.action_layout import compile_action_layout


def _build_dualarm_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="dualarm_piper",
        camera_slots={
            "left_wrist": CameraBinding(obs_term="left_hand_camera_term"),
            "right_wrist": CameraBinding(obs_term="right_hand_camera_term"),
            "base": CameraBinding(obs_term="static_camera_term"),
        },
        manipulator_slots={
            "left_arm": ManipulatorBinding(
                joint_position_obs_key="left_joint_position",
                arm_joint_name_specs=("left_joint[1-6]",),
                gripper_joint_name_specs=("left_joint[7-8]",),
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            ),
            "right_arm": ManipulatorBinding(
                joint_position_obs_key="right_joint_position",
                arm_joint_name_specs=("right_joint[1-6]",),
                gripper_joint_name_specs=("right_joint[7-8]",),
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            ),
        },
    )


def _build_franka_schema() -> PolicyBindingSchema:
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type="franka_panda",
        camera_slots={
            "wrist": CameraBinding(obs_term="wrist_camera_term"),
            "base": CameraBinding(obs_term="ext1_camera_term"),
            "right_wrist": CameraBinding(obs_term="ext2_camera_term"),
        },
        manipulator_slots={
            "single_arm": ManipulatorBinding(
                joint_position_obs_key="joint_position",
                arm_joint_name_specs=("panda_joint[1-7]",),
                gripper_joint_name_specs=(
                    "panda_finger_joint1",
                    "panda_finger_joint2",
                ),
                gripper_policy_representation="first_joint",
                gripper_decode_coupling="symmetric",
                gripper_policy_scale=2.0,
            )
        },
    )


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


def build_openpi_test_observation() -> dict:
    camera_obs = {
        "rgb": Sensor(
            torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
            intrinsic_matrices=torch.eye(3).unsqueeze(0),
        ),
        "depth": Sensor(torch.ones((1, 2, 2, 1))),
    }
    return {
        "/camera": {
            "left_hand_camera_term": camera_obs,
            "right_hand_camera_term": camera_obs,
            "static_camera_term": camera_obs,
        },
        "/robot": {
            "left_joint_position": torch.ones((1, 7)),
            "right_joint_position": torch.ones((1, 7)),
        },
        "instruction": "pick apple",
    }


def build_openpi_canonical_observation() -> CanonicalPolicyInput:
    camera_obs = {
        "rgb": Sensor(
            torch.zeros((1, 2, 2, 3), dtype=torch.uint8),
            intrinsic_matrices=torch.eye(3).unsqueeze(0),
        ),
        "depth": Sensor(torch.ones((1, 2, 2, 1))),
    }
    return CanonicalPolicyInput(
        cameras={
            "left_wrist": camera_obs,
            "right_wrist": camera_obs,
            "base": camera_obs,
        },
        manipulators={
            "left_arm": {"joint_position": torch.ones((1, 7))},
            "right_arm": {"joint_position": torch.ones((1, 7))},
        },
        instruction="pick apple",
        action_layout=compile_action_layout(_build_dualarm_schema()),
    )


def build_single_arm_canonical_observation() -> CanonicalPolicyInput:
    camera_obs = {
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
    return CanonicalPolicyInput(
        cameras={
            "wrist": camera_obs,
            "right_wrist": camera_obs,
            "base": camera_obs,
        },
        manipulators={
            "single_arm": {
                "joint_position": torch.tensor(
                    [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]]
                ),
            }
        },
        instruction="pick apple",
        action_layout=compile_action_layout(_build_franka_schema()),
    )


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


def encode_binary_message_for_test(payload: dict) -> bytes:
    tensors: list[bytes] = []

    def extract(value):
        if isinstance(value, (bytes, bytearray, memoryview)):
            tensor_idx = len(tensors)
            tensors.append(bytes(value))
            return {"__bytes_idx__": tensor_idx}
        if isinstance(value, dict):
            if "__tensor__" in value:
                tensor_payload = dict(value["__tensor__"])
                tensor_data = tensor_payload.pop("data")
                if isinstance(tensor_data, list):
                    tensor_bytes = (
                        torch.tensor(
                            tensor_data,
                            dtype=getattr(torch, tensor_payload["dtype"]),
                        )
                        .contiguous()
                        .numpy()
                        .tobytes()
                    )
                else:
                    tensor_bytes = bytes(tensor_data)
                tensor_idx = len(tensors)
                tensors.append(tensor_bytes)
                return {
                    "__tensor__": tensor_payload,
                    "__tensor_idx__": tensor_idx,
                }
            return {key: extract(item) for key, item in value.items()}
        if isinstance(value, list):
            return [extract(item) for item in value]
        return value

    header_payload = extract(payload)
    header_json = json.dumps(header_payload, separators=(",", ":")).encode(
        "utf-8"
    )
    tensor_blob = b"".join(
        struct.pack("!Q", len(tensor_bytes)) + tensor_bytes
        for tensor_bytes in tensors
    )
    return struct.pack("!Q", len(header_json)) + header_json + tensor_blob


def encode_value_for_test(value):
    if isinstance(value, UnifiedJointCommand):
        return {
            "__joint_command__": {
                "values": {
                    "__tensor__": encode_tensor_payload_for_test(value.values)
                },
                "joint_names": list(value.joint_names),
            }
        }
    if isinstance(value, torch.Tensor):
        return {"__tensor__": encode_tensor_payload_for_test(value)}
    if isinstance(value, dict):
        return {
            key: encode_value_for_test(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [encode_value_for_test(item) for item in value]
    if isinstance(value, tuple):
        return [encode_value_for_test(item) for item in value]
    return value


def encode_tensor_payload_for_test(tensor: torch.Tensor) -> dict:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(cpu.dtype).replace("torch.", ""),
        "shape": list(cpu.shape),
        "data": cpu.numpy().tobytes(),
    }


def extract_canonical_obs_data_for_test(
    observations: CanonicalPolicyInput,
) -> dict:
    obs: dict = {
        "format": "canonical",
        "instruction": observations.instruction,
    }
    if observations.action_layout is not None:
        obs["action_layout"] = observations.action_layout.to_payload()

    cameras: dict = {}
    for slot, camera_obs in observations.cameras.items():
        cam: dict = {}
        if "rgb" in camera_obs:
            rgb = camera_obs["rgb"]
            cam["rgb"] = encode_tensor_payload_for_test(rgb.sensor_data)
            if rgb.intrinsic_matrices is not None:
                cam["intrinsic_matrices"] = encode_tensor_payload_for_test(
                    rgb.intrinsic_matrices
                )
            if rgb.pose is not None:
                cam["pose"] = {
                    "xyz": encode_tensor_payload_for_test(rgb.pose.xyz),
                    "quat": encode_tensor_payload_for_test(rgb.pose.quat),
                }
        if "depth" in camera_obs:
            cam["depth"] = encode_tensor_payload_for_test(
                camera_obs["depth"].sensor_data
            )
        cameras[slot] = cam
    obs["cameras"] = cameras
    obs["manipulators"] = {
        slot: encode_value_for_test(manipulator_obs)
        for slot, manipulator_obs in observations.manipulators.items()
    }
    return obs


def decode_binary_message_for_test(message: bytes) -> dict:
    header_len = struct.unpack("!Q", message[:8])[0]
    header_end = 8 + header_len
    header_payload = json.loads(message[8:header_end].decode("utf-8"))
    tensors: list[bytes] = []
    cursor = header_end
    while cursor < len(message):
        tensor_len = struct.unpack("!Q", message[cursor : cursor + 8])[0]
        cursor += 8
        tensors.append(message[cursor : cursor + tensor_len])
        cursor += tensor_len

    def inject(value):
        if isinstance(value, dict):
            if "__bytes_idx__" in value:
                return tensors[value["__bytes_idx__"]]
            if "__tensor__" in value:
                tensor_payload = dict(value["__tensor__"])
                tensor_payload["data"] = tensors[value["__tensor_idx__"]]
                return {"__tensor__": tensor_payload}
            return {key: inject(item) for key, item in value.items()}
        if isinstance(value, list):
            return [inject(item) for item in value]
        return value

    return inject(header_payload)


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
    assert policy_cfg.remote_policy_type == "full"


def test_create_policy_from_model_cfg_given_remote_policy_type_returns_cfg():
    factory_module = load_factory_module()
    policy_cfg = factory_module.create_policy_from_model_cfg(
        {
            "policy": "server",
            "host": "127.0.0.1",
            "port": 9000,
            "remote_policy_type": "openpi",
        }
    )

    assert policy_cfg.remote_policy_type == "openpi"


def test_create_policy_from_model_cfg_none_remote_type_uses_full():
    factory_module = load_factory_module()
    policy_cfg = factory_module.create_policy_from_model_cfg(
        {
            "policy": "server",
            "host": "127.0.0.1",
            "port": 9000,
            "remote_policy_type": None,
        }
    )

    assert policy_cfg.remote_policy_type == "full"


def test_policy_client_request_action_given_camera_obs_returns_actions(
    monkeypatch,
) -> None:
    fake_ws = Mock()
    server_module = load_server_module()
    fake_ws.recv.return_value = encode_binary_message_for_test(
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
    client = server_module.PolicyClient(host="127.0.0.1", port=8765)

    actions = client.request_action(build_public_test_observation())

    connect.assert_called_once()
    fake_ws.send.assert_called_once()
    sent_payload = fake_ws.send.call_args.args[0]
    assert isinstance(sent_payload, bytes)
    request = decode_binary_message_for_test(sent_payload)
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


def test_policy_client_request_sequence_given_camera_obs_returns_sequence(
    monkeypatch,
) -> None:
    fake_ws = Mock()
    server_module = load_server_module()
    fake_ws.recv.return_value = encode_binary_message_for_test(
        {
            "actions": [
                {
                    "left_robot_joint_position": {
                        "__tensor__": {
                            "dtype": "float32",
                            "shape": [1, 6],
                            "data": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                        }
                    }
                },
                {
                    "left_robot_joint_position": {
                        "__tensor__": {
                            "dtype": "float32",
                            "shape": [1, 6],
                            "data": [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                        }
                    }
                },
            ],
            "logging_tag": "remote-eval",
        }
    )
    install_fake_websocket_client(monkeypatch, fake_ws)
    client = server_module.PolicyClient(host="127.0.0.1", port=8765)

    sequence = client.request_action_sequence(build_public_test_observation())

    fake_ws.send.assert_called_once()
    request = decode_binary_message_for_test(fake_ws.send.call_args.args[0])
    assert request["type"] == "act_sequence"
    assert len(sequence) == 2
    assert torch.equal(
        sequence[0]["left_robot_joint_position"].cpu(),
        torch.ones((1, 6)),
    )
    assert torch.equal(
        sequence[1]["left_robot_joint_position"].cpu(),
        torch.full((1, 6), 2.0),
    )


def test_policy_client_request_action_sequence_given_openpi_type_omits_depth(
    monkeypatch,
) -> None:
    fake_ws = Mock()
    server_module = load_server_module()
    fake_ws.recv.return_value = encode_binary_message_for_test(
        {
            "actions": encode_value_for_test(
                [
                    UnifiedJointCommand.from_specs(
                        torch.tensor(
                            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                            dtype=torch.float32,
                        ),
                        ["left_joint[1-6]"],
                    )
                ]
            ),
            "logging_tag": "remote-eval",
        }
    )
    install_fake_websocket_client(monkeypatch, fake_ws)
    client = server_module.PolicyClient(
        host="127.0.0.1",
        port=8765,
        remote_policy_type="openpi",
    )

    sequence = client.request_action_sequence(
        build_openpi_canonical_observation()
    )

    request = decode_binary_message_for_test(fake_ws.send.call_args.args[0])
    assert request["obs_data"]["format"] == "canonical"
    compact_camera = request["obs_data"]["cameras"]["left_wrist"]
    assert set(compact_camera) == {"depth", "intrinsic_matrices", "rgb"}
    assert "pose" not in compact_camera
    assert sequence[0].select("left_joint1")[0, 0].item() == 1.0


def test_policy_client_request_action_sequence_single_arm_openpi_keeps_layout(
    monkeypatch,
) -> None:
    fake_ws = Mock()
    server_module = load_server_module()
    fake_ws.recv.return_value = encode_binary_message_for_test(
        {
            "actions": encode_value_for_test(
                [
                    UnifiedJointCommand.from_specs(
                        torch.tensor(
                            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
                            dtype=torch.float32,
                        ),
                        ["panda_joint[1-7]"],
                    )
                ]
            ),
            "logging_tag": "remote-eval",
        }
    )
    install_fake_websocket_client(monkeypatch, fake_ws)
    client = server_module.PolicyClient(
        host="127.0.0.1",
        port=8765,
        remote_policy_type="openpi",
    )

    client.request_action_sequence(build_single_arm_canonical_observation())

    request = decode_binary_message_for_test(fake_ws.send.call_args.args[0])
    assert request["obs_data"]["format"] == "canonical"
    assert request["obs_data"]["action_layout"]["manipulator_order"] == [
        "single_arm"
    ]
    assert "camera_slots" not in request["obs_data"]["action_layout"]
    assert "camera_order" not in request["obs_data"]["action_layout"]


def test_policy_client_request_action_sequence_openpi_raw_obs_raises(
    monkeypatch,
) -> None:
    fake_ws = Mock()
    server_module = load_server_module()
    install_fake_websocket_client(monkeypatch, fake_ws)
    client = server_module.PolicyClient(
        host="127.0.0.1",
        port=8765,
        remote_policy_type="openpi",
    )

    try:
        client.request_action_sequence(build_openpi_test_observation())
    except ValueError as exc:
        assert "CanonicalPolicyInput" in str(exc)
    else:
        raise AssertionError("Expected ValueError for raw openpi observations")


def test_server_policy_reset_given_remote_client_sends_reset_request(
    monkeypatch,
) -> None:
    fake_ws = Mock()
    server_module = load_server_module()
    fake_ws.recv.return_value = encode_binary_message_for_test(
        {
            "ok": True,
            "logging_tag": "remote-eval",
        }
    )
    connect = install_fake_websocket_client(monkeypatch, fake_ws)
    policy = server_module.ServerPolicy(
        server_module.ServerPolicyCfg(host="127.0.0.1", port=8765)
    )

    policy.reset()

    connect.assert_called_once()
    fake_ws.send.assert_called_once()
    request = decode_binary_message_for_test(fake_ws.send.call_args.args[0])
    assert request["type"] == "reset"
    assert policy.logging_tag == "remote-eval"


def test_server_policy_act_given_cached_remote_sequence_reuses_local_cache(
    monkeypatch,
) -> None:
    server_module = load_server_module()
    fake_ws = Mock()
    fake_ws.recv.return_value = encode_binary_message_for_test(
        {
            "actions": [
                {
                    "left_robot_joint_position": {
                        "__tensor__": {
                            "dtype": "float32",
                            "shape": [1, 6],
                            "data": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
                        }
                    }
                },
                {
                    "left_robot_joint_position": {
                        "__tensor__": {
                            "dtype": "float32",
                            "shape": [1, 6],
                            "data": [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                        }
                    }
                },
            ]
        }
    )
    connect = install_fake_websocket_client(monkeypatch, fake_ws)
    policy = server_module.ServerPolicy(
        server_module.ServerPolicyCfg(host="127.0.0.1", port=8765)
    )

    first = policy.act(build_public_test_observation())
    second = policy.act(build_public_test_observation())

    connect.assert_called_once()
    fake_ws.send.assert_called_once()
    request = decode_binary_message_for_test(fake_ws.send.call_args.args[0])
    assert request["type"] == "act_sequence"
    assert torch.equal(
        first["left_robot_joint_position"].cpu(), torch.ones((1, 6))
    )
    assert torch.equal(
        second["left_robot_joint_position"].cpu(), torch.full((1, 6), 2.0)
    )


def test_policy_websocket_server_handle_canonical_obs_applies_instruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_module = load_server_module()

    class _CapturingPolicy:
        def __init__(self) -> None:
            self.observations = []

        def act(self, observations):
            self.observations.append(observations)
            return UnifiedJointCommand.from_specs(
                torch.tensor([[1.0]], dtype=torch.float32),
                ["joint1"],
            )

        def reset(self) -> None:
            return None

    class _FakeWebsocket:
        def __init__(self, message: bytes) -> None:
            self.remote_address = ("127.0.0.1", 8765)
            self._messages = [message]
            self.sent = []

        def __aiter__(self):
            self._iter = iter(self._messages)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def send(self, message: bytes) -> None:
            self.sent.append(message)

    policy = _CapturingPolicy()
    server = server_module.PolicyWebsocketServer(policy=policy)
    websocket = _FakeWebsocket(
        encode_binary_message_for_test(
            {
                "type": "act",
                "obs_data": extract_canonical_obs_data_for_test(
                    build_single_arm_canonical_observation()
                ),
                "instruction": "updated instruction",
            }
        )
    )

    asyncio.run(server.handle_client(websocket))

    assert len(policy.observations) == 1
    assert isinstance(policy.observations[0], CanonicalPolicyInput)
    assert policy.observations[0].instruction == "updated instruction"
