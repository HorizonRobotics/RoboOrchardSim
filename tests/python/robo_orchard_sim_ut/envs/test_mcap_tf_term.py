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

"""Tests for the MCAP TF record term."""

from __future__ import annotations
import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
from google.protobuf.timestamp_pb2 import Timestamp
from robo_orchard_core.datatypes.geometry import BatchFrameTransform
from robo_orchard_core.datatypes.tf_graph import BatchFrameTransformGraph


class _StubRecordTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._cfg = cfg
        self._env = env

    def __class_getitem__(cls, _item):
        return cls

    def _parse_data_from_dict(
        self, data: dict[str, Any | dict[str, Any]], key: str
    ) -> dict[str, Any]:
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

        current: Any = data
        for segment in split_str(key):
            current = current[segment]
        return {key: current}


class _StubRecordTermBaseCfg:
    def __init__(self, **kwargs):
        for name, value in type(self).__dict__.items():
            if (
                name.startswith("_")
                or callable(value)
                or isinstance(value, property)
            ):
                continue
            setattr(self, name, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class _StubClassTypeCo:
    def __class_getitem__(cls, _item):
        return cls


@dataclass
class _StubMessage:
    data: object
    log_time: Timestamp
    pub_time: Timestamp


@pytest.fixture
def tf_term_module(monkeypatch: pytest.MonkeyPatch):
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

    record_module = sys.modules["robo_orchard_sim.ext.envs.managers.record"]
    record_module.__path__ = []
    record_module.RecordTermBase = _StubRecordTermBase
    record_module.RecordTermBaseCfg = _StubRecordTermBaseCfg

    message_module = types.ModuleType(
        "robo_orchard_sim.ext.envs.managers.record.mcap.message"
    )
    message_module.Message = _StubMessage
    monkeypatch.setitem(sys.modules, message_module.__name__, message_module)

    env_base_module = types.ModuleType("robo_orchard_sim.ext.envs.env_base")
    env_base_module.IsaacEnvType_co = object
    monkeypatch.setitem(sys.modules, env_base_module.__name__, env_base_module)

    record_manager_module = types.ModuleType(
        "robo_orchard_sim.ext.envs.managers.record.record_manager"
    )
    record_manager_module.MsgsType = dict
    monkeypatch.setitem(
        sys.modules,
        record_manager_module.__name__,
        record_manager_module,
    )

    utils_config_module = types.ModuleType("robo_orchard_sim.utils.config")
    utils_config_module.ClassType_co = _StubClassTypeCo
    monkeypatch.setitem(
        sys.modules, utils_config_module.__name__, utils_config_module
    )

    module_path = (
        Path(__file__).resolve().parents[4]
        / "robo_orchard_sim"
        / "ext"
        / "envs"
        / "managers"
        / "record"
        / "mcap"
        / "tf_term.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_mcap_tf_term_module",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_tf_graph(*frame_pairs: tuple[str, str]) -> BatchFrameTransformGraph:
    tf_list = []
    for index, (parent_frame_id, child_frame_id) in enumerate(frame_pairs, 1):
        tf_list.append(
            BatchFrameTransform(
                xyz=torch.tensor(
                    [[float(index), float(index + 1), float(index + 2)]],
                    dtype=torch.float32,
                ),
                quat=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                ),
                parent_frame_id=parent_frame_id,
                child_frame_id=child_frame_id,
            )
        )
    return BatchFrameTransformGraph(tf_list=tf_list, bidirectional=False)


def _make_term(tf_term_module, **cfg_kwargs):
    cfg_data = {
        "fps": 30,
        "key": "/tf/test_tf",
        "topic": "/observation/test/tf",
    }
    cfg_data.update(cfg_kwargs)
    cfg = tf_term_module.McapTFTermCfg(**cfg_data)
    return tf_term_module.McapTFTerm(
        cfg,
        env=types.SimpleNamespace(physics_dt=1 / 60),
    )


def test_mcap_tf_term_without_configured_frames_returns_runtime_frame_ids(
    tf_term_module,
):
    term = _make_term(
        tf_term_module,
        topic="/observation/robot_state/link/left_link{id}/tf",
    )
    timestamp = Timestamp(seconds=123, nanos=456)
    graph = _make_tf_graph(
        (
            "robots/dualarm_piper/left_base_link",
            "robots/dualarm_piper/left_link1",
        ),
    )

    msgs = term({"/tf": {"test_tf": graph}}, timestamp)

    [msg] = msgs["/observation/robot_state/link/left_link1/tf"]
    assert (
        msg.data.parent_frame_id,
        msg.data.child_frame_id,
    ) == (
        "robots/dualarm_piper/left_base_link",
        "robots/dualarm_piper/left_link1",
    )


def test_mcap_tf_term_with_topic_placeholder_returns_indexed_topics(
    tf_term_module,
):
    term = _make_term(
        tf_term_module,
        topic="/observation/robot_state/link/left_link{id}/tf",
    )
    graph = _make_tf_graph(
        (
            "robots/dualarm_piper/left_base_link",
            "robots/dualarm_piper/left_link1",
        ),
        (
            "robots/dualarm_piper/left_link1",
            "robots/dualarm_piper/left_link2",
        ),
    )

    msgs = term({"/tf": {"test_tf": graph}}, Timestamp(seconds=123, nanos=456))

    assert set(msgs) == {
        "/observation/robot_state/link/left_link1/tf",
        "/observation/robot_state/link/left_link2/tf",
    }


def test_mcap_tf_term_returns_transform_values_in_message_payload(
    tf_term_module,
):
    term = _make_term(tf_term_module)
    timestamp = Timestamp(seconds=123, nanos=456)
    graph = _make_tf_graph(("world", "tool"))

    msgs = term({"/tf": {"test_tf": graph}}, timestamp)

    [msg] = msgs["/observation/test/tf"]
    assert (
        msg.data.translation.x,
        msg.data.translation.y,
        msg.data.translation.z,
        msg.data.rotation.w,
    ) == pytest.approx((1.0, 2.0, 3.0, 1.0))


def test_mcap_tf_term_with_multiple_transforms_appends_topic_id_suffix(
    tf_term_module,
):
    term = _make_term(tf_term_module)
    timestamp = Timestamp(seconds=123, nanos=456)
    graph = _make_tf_graph(("world", "tool1"), ("world", "tool2"))

    msgs = term({"/tf": {"test_tf": graph}}, timestamp)

    assert set(msgs) == {"/observation/test/tf/1", "/observation/test/tf/2"}
    assert [
        msgs["/observation/test/tf/1"][0].data.child_frame_id,
        msgs["/observation/test/tf/2"][0].data.child_frame_id,
    ] == ["tool1", "tool2"]


def test_mcap_tf_term_with_multiple_transforms_and_strict_topic_id_raises_value_error(  # noqa: E501
    tf_term_module,
):
    term = _make_term(tf_term_module, strict_topic_id=True)
    timestamp = Timestamp(seconds=123, nanos=456)
    graph = _make_tf_graph(("world", "tool1"), ("world", "tool2"))

    with pytest.raises(ValueError, match="topic"):
        term({"/tf": {"test_tf": graph}}, timestamp)


def test_mcap_tf_term_with_only_parent_frame_raises_value_error(
    tf_term_module,
):
    cfg = tf_term_module.McapTFTermCfg(
        topic="/tf",
        fps=30,
        key="/tf/test_tf",
        parent_frame="world",
    )

    with pytest.raises(ValueError, match="parent_frame and child_frame"):
        tf_term_module.McapTFTerm(
            cfg,
            env=types.SimpleNamespace(physics_dt=1 / 60),
        )


def test_mcap_tf_term_with_configured_frames_returns_override_frame_ids(
    tf_term_module,
):
    term = _make_term(
        tf_term_module,
        topic="/observation/cameras/camera_{id}/tf",
        parent_frame="robots/dualarm_piper/left_base_link",
        child_frame=["static_camera", "left_hand_camera"],
    )
    graph = _make_tf_graph(
        (
            "robots/dualarm_piper/left_base_link",
            "cameras/static_camera",
        ),
        (
            "robots/dualarm_piper/left_link6",
            "cameras/left_hand_camera",
        ),
    )

    msgs = term({"/tf": {"test_tf": graph}}, Timestamp(seconds=123, nanos=456))

    [first_msg] = msgs["/observation/cameras/camera_1/tf"]
    [second_msg] = msgs["/observation/cameras/camera_2/tf"]
    assert [
        (
            first_msg.data.parent_frame_id,
            first_msg.data.child_frame_id,
        ),
        (
            second_msg.data.parent_frame_id,
            second_msg.data.child_frame_id,
        ),
    ] == [
        ("robots/dualarm_piper/left_base_link", "static_camera"),
        ("robots/dualarm_piper/left_base_link", "left_hand_camera"),
    ]


def test_mcap_tf_term_with_configured_frames_returns_indexed_topics(
    tf_term_module,
):
    term = _make_term(
        tf_term_module,
        topic="/observation/cameras/camera_{id}/tf",
        parent_frame="robots/dualarm_piper/left_base_link",
        child_frame=["static_camera", "left_hand_camera"],
    )
    graph = _make_tf_graph(
        ("robots/dualarm_piper/left_base_link", "cameras/static_camera"),
        ("robots/dualarm_piper/left_link6", "cameras/left_hand_camera"),
    )

    msgs = term({"/tf": {"test_tf": graph}}, Timestamp(seconds=123, nanos=456))

    assert set(msgs) == {
        "/observation/cameras/camera_1/tf",
        "/observation/cameras/camera_2/tf",
    }
