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

"""DreamZero policy configuration and transport tests."""

from robo_orchard_sim.policy.dreamzero.client import DreamZeroClient
from robo_orchard_sim.policy.dreamzero.policy import DreamZeroPolicyCfg
from robo_orchard_sim.policy.factory import create_policy_from_model_cfg


class _FakeConnection:
    def recv(self) -> bytes:
        return b"metadata"

    def close(self) -> None:
        pass


def test_dreamzero_client_supported_websocket_kwargs_connects(
    monkeypatch,
) -> None:
    import websockets.sync.client

    monkeypatch.setattr(
        websockets.sync.client,
        "connect",
        lambda uri, **kwargs: _FakeConnection(),
    )
    monkeypatch.setattr(
        DreamZeroClient,
        "_build_packer",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(
        DreamZeroClient,
        "_unpack",
        staticmethod(lambda payload: {"action_space": "joint_position"}),
    )

    client = DreamZeroClient(host="127.0.0.1", port=8000)

    assert client.metadata == {"action_space": "joint_position"}


def test_policy_factory_dreamzero_config_returns_dreamzero_cfg() -> None:
    cfg = create_policy_from_model_cfg(
        {
            "policy": "dreamzero",
            "host": "10.12.121.165",
            "port": 8000,
            "logging_tag": "dreamzero-droid",
            "valid_action_step": 8,
            "camera_mapping": {
                "static_camera": "ext1_camera",
                "left_hand_camera": "ext2_camera",
                "right_hand_camera": "wrist_camera",
            },
        }
    )

    assert (
        isinstance(cfg, DreamZeroPolicyCfg),
        cfg.host,
        cfg.port,
        cfg.valid_action_step,
        cfg.camera_mapping["right_hand_camera"],
    ) == (True, "10.12.121.165", 8000, 8, "wrist_camera")
