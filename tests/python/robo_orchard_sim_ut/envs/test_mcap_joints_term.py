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

from __future__ import annotations
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from google.protobuf.timestamp_pb2 import Timestamp


class _FakeTensor:
    def __init__(self, value: np.ndarray):
        self._value = value

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _StubRecordTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._cfg = cfg
        self.env = env

    def __class_getitem__(cls, _item):
        return cls

    def _parse_data_from_dict(self, data, key):
        def split_str(string: str) -> list[str]:
            raw = string.split("/")
            parts = []
            for seg in raw:
                if seg == "":
                    if not parts or not parts[-1].startswith("/"):
                        parts.append("/")
                else:
                    if parts and parts[-1] == "/":
                        parts[-1] = "/" + seg
                    else:
                        parts.append(seg)
            return parts

        current = data
        for item in split_str(key):
            if item not in current:
                raise KeyError(item)
            current = current[item]
        return {key: current}


class _StubRecordTermBaseCfg:
    def __class_getitem__(cls, _item):
        return cls


class _StubClassTypeCo:
    def __class_getitem__(cls, _item):
        return cls


@dataclass
class _StubMessage:
    data: object
    log_time: Timestamp
    pub_time: Timestamp


def _load_joints_term_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "tested_mcap_joints_term"
    repo_root = Path(__file__).resolve().parents[4]
    file_path = (
        repo_root
        / "robo_orchard_sim/ext/envs/managers/record/mcap/joints_term.py"
    )

    package_names = [
        "robo_orchard_sim",
        "robo_orchard_sim.ext.envs",
        "robo_orchard_sim.ext.envs.managers",
        "robo_orchard_sim.ext.envs.managers.record",
        "robo_orchard_sim.ext.envs.managers.record.mcap",
        "robo_orchard_sim.utils",
    ]
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    env_base = types.ModuleType("robo_orchard_sim.ext.envs.env_base")
    env_base.IsaacEnvType_co = object
    monkeypatch.setitem(sys.modules, env_base.__name__, env_base)

    record_module_name = "robo_orchard_sim.ext.envs.managers.record"
    record_module = types.ModuleType(record_module_name)
    record_module.__path__ = []
    record_module.RecordTermBase = _StubRecordTermBase
    record_module.RecordTermBaseCfg = _StubRecordTermBaseCfg
    monkeypatch.setitem(sys.modules, record_module.__name__, record_module)

    message_module = types.ModuleType(
        "robo_orchard_sim.ext.envs.managers.record.mcap.message"
    )
    message_module.Message = _StubMessage
    monkeypatch.setitem(sys.modules, message_module.__name__, message_module)

    config_module = types.ModuleType("robo_orchard_sim.utils.config")
    config_module.ClassType_co = _StubClassTypeCo
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_cfg(**overrides):
    cfg = {
        "topic": "/joint_states",
        "fps": 10.0,
        "position_key": "obs/joint_pos",
        "velocity_key": "obs/joint_vel",
        "effort_key": "obs/joint_eff",
        "joint_name_prefix": "joint",
        "joint_ids": None,
        "joint_id_offset": 0,
    }
    cfg.update(overrides)
    return types.SimpleNamespace(**cfg)


def _make_term(monkeypatch: pytest.MonkeyPatch, **cfg_overrides):
    module = _load_joints_term_module(monkeypatch)
    term = module.McapJointsTerm(
        _make_cfg(**cfg_overrides),
        types.SimpleNamespace(),
    )
    return module, term


def _record_messages(
    monkeypatch: pytest.MonkeyPatch,
    *,
    obs: dict[str, _FakeTensor],
    timestamp: Timestamp | None = None,
    **cfg_overrides,
):
    _, term = _make_term(monkeypatch, **cfg_overrides)
    return term(
        {"obs": obs},
        timestamp or Timestamp(seconds=10, nanos=20),
    )


def test_mcap_joints_term_with_only_positions_returns_joint_positions(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = _record_messages(
        monkeypatch,
        obs={"joint_pos": _FakeTensor(np.array([[1.0, 2.0]]))},
    )

    [message] = messages["/joint_states"]
    assert [state.name for state in message.data.states] == [
        "joint1",
        "joint2",
    ]
    assert [state.position for state in message.data.states] == [1.0, 2.0]


def test_mcap_joints_term_with_only_positions_zero_fills_missing_velocity(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = _record_messages(
        monkeypatch,
        obs={"joint_pos": _FakeTensor(np.array([[1.0, 2.0]]))},
    )

    [message] = messages["/joint_states"]
    assert [state.velocity for state in message.data.states] == [0.0, 0.0]


def test_mcap_joints_term_with_only_positions_zero_fills_missing_effort(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = _record_messages(
        monkeypatch,
        obs={"joint_pos": _FakeTensor(np.array([[1.0, 2.0]]))},
    )

    [message] = messages["/joint_states"]
    assert [state.effort for state in message.data.states] == [0.0, 0.0]


def test_mcap_joints_term_with_multiple_instances_returns_one_message_each(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = _record_messages(
        monkeypatch,
        obs={
            "joint_pos": _FakeTensor(np.array([[1.0, 2.0], [3.0, 4.0]])),
            "joint_vel": _FakeTensor(np.array([[0.1, 0.2], [0.3, 0.4]])),
        },
        timestamp=Timestamp(seconds=123, nanos=456),
    )

    env_messages = messages["/joint_states"]
    assert len(env_messages) == 2


def test_mcap_joints_term_with_multiple_instances_copies_timestamp_to_each(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = _record_messages(
        monkeypatch,
        obs={
            "joint_pos": _FakeTensor(np.array([[1.0, 2.0], [3.0, 4.0]])),
            "joint_vel": _FakeTensor(np.array([[0.1, 0.2], [0.3, 0.4]])),
        },
        timestamp=Timestamp(seconds=123, nanos=456),
    )

    env_messages = messages["/joint_states"]
    assert [msg.data.timestamp.seconds for msg in env_messages] == [123, 123]
    assert [msg.data.timestamp.nanos for msg in env_messages] == [456, 456]
    assert [msg.log_time.seconds for msg in env_messages] == [123, 123]
    assert [msg.pub_time.seconds for msg in env_messages] == [123, 123]


def test_mcap_joints_term_with_multiple_instances_returns_positions_by_row(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = _record_messages(
        monkeypatch,
        obs={
            "joint_pos": _FakeTensor(np.array([[1.0, 2.0], [3.0, 4.0]])),
            "joint_vel": _FakeTensor(np.array([[0.1, 0.2], [0.3, 0.4]])),
        },
        timestamp=Timestamp(seconds=123, nanos=456),
    )

    env_messages = messages["/joint_states"]
    assert [state.position for state in env_messages[0].data.states] == [
        1.0,
        2.0,
    ]
    assert [state.position for state in env_messages[1].data.states] == [
        3.0,
        4.0,
    ]


def test_mcap_joints_term_with_missing_effort_zero_fills_each_instance(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = _record_messages(
        monkeypatch,
        obs={
            "joint_pos": _FakeTensor(np.array([[1.0, 2.0], [3.0, 4.0]])),
            "joint_vel": _FakeTensor(np.array([[0.1, 0.2], [0.3, 0.4]])),
        },
        timestamp=Timestamp(seconds=123, nanos=456),
    )

    env_messages = messages["/joint_states"]
    assert [state.effort for state in env_messages[0].data.states] == [
        0.0,
        0.0,
    ]


def test_mcap_joints_term_with_non_2d_joint_input_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _, term = _make_term(monkeypatch)

    with pytest.raises(ValueError, match="num_instances, num_joints"):
        term(
            {
                "obs": {
                    "joint_pos": _FakeTensor(
                        np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
                    ),
                }
            },
            Timestamp(seconds=1),
        )


def test_mcap_joints_term_with_joint_ids_returns_selected_joint_names(
    monkeypatch: pytest.MonkeyPatch,
):
    _, term = _make_term(
        monkeypatch,
        joint_ids=[2, 5],
        joint_id_offset=10,
    )

    messages = term(
        {
            "obs": {
                "joint_pos": _FakeTensor(
                    np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
                ),
            }
        },
        Timestamp(seconds=1),
    )

    [message] = messages["/joint_states"]
    assert [state.name for state in message.data.states] == [
        "joint13",
        "joint16",
    ]


def test_mcap_joints_term_with_joint_ids_returns_selected_joint_positions(
    monkeypatch: pytest.MonkeyPatch,
):
    _, term = _make_term(
        monkeypatch,
        joint_ids=[2, 5],
        joint_id_offset=10,
    )

    messages = term(
        {
            "obs": {
                "joint_pos": _FakeTensor(
                    np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
                ),
            }
        },
        Timestamp(seconds=1),
    )

    [message] = messages["/joint_states"]
    assert [state.position for state in message.data.states] == [2.0, 5.0]


def test_mcap_joints_term_with_empty_joint_positions_skips_recording(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = _record_messages(
        monkeypatch,
        obs={"joint_pos": _FakeTensor(np.array([]))},
    )

    assert messages == {}
