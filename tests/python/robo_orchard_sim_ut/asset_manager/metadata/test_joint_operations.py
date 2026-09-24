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

"""Behavioral tests for typed joint-operation metadata loading."""

from __future__ import annotations
import json
from pathlib import Path

import pytest

from robo_orchard_sim.asset_manager.metadata import (
    ArticulationOperationMeta,
    JointOperationMeta,
    get_joint,
    get_operation,
    load_articulation_metadata,
)


def _valid_metadata() -> dict:
    return {
        "uuid": "laptop-uuid",
        "articulation": {
            "root": {"fix_root_link": True},
            "joints": [
                {
                    "joint_name": "lid_joint",
                    "semantic_name": "lid",
                    "outcome_link": "lid_link",
                    "operations": {
                        "open": {
                            "interaction_link": "button_link",
                            "initial_joint_position": -0.5,
                            "target_joint_fraction": 0.0,
                        },
                        "close": {
                            "interaction_link": "lid_link",
                            "initial_joint_position": -1.5,
                            "target_joint_fraction": 1.0,
                        },
                    },
                }
            ],
        },
    }


def _write_metadata(path: Path, raw: dict) -> None:
    path.write_text(json.dumps(raw))


def test_load_articulation_metadata_valid_file_returns_typed_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.json"
    _write_metadata(path, _valid_metadata())

    metadata = load_articulation_metadata(
        str(path),
        expected_uuid="laptop-uuid",
    )

    assert metadata.uuid == "laptop-uuid"
    assert metadata.root.fix_root_link is True
    assert metadata.joints[0].outcome_link == "lid_link"
    assert (
        metadata.joints[0].operations["open"].interaction_link == "button_link"
    )
    assert metadata.joints[0].operations["close"].target_joint_fraction == 1.0


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (
            lambda raw: raw["articulation"]["joints"][0].update(
                outcome_link=" "
            ),
            "outcome_link",
        ),
        (
            lambda raw: raw["articulation"]["joints"][0]["operations"][
                "open"
            ].update(interaction_link=" "),
            "interaction_link",
        ),
        (
            lambda raw: raw["articulation"]["joints"][0]["operations"][
                "open"
            ].update(target_joint_fraction=1.5),
            "target_joint_fraction",
        ),
        (
            lambda raw: raw["articulation"]["joints"][0]["operations"].update(
                unknown={
                    "initial_joint_position": 0.0,
                    "target_joint_fraction": 0.5,
                }
            ),
            "literal_error",
        ),
        (
            lambda raw: raw["articulation"]["joints"].append(
                dict(raw["articulation"]["joints"][0])
            ),
            "duplicate joint_name",
        ),
    ],
)
def test_load_articulation_metadata_invalid_semantics_raises_value_error(
    tmp_path: Path,
    mutate,
    error: str,
) -> None:
    raw = _valid_metadata()
    mutate(raw)
    path = tmp_path / "metadata.json"
    _write_metadata(path, raw)

    with pytest.raises(ValueError, match=error):
        load_articulation_metadata(str(path))


def test_load_articulation_metadata_mismatched_uuid_raises_value_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.json"
    _write_metadata(path, _valid_metadata())

    with pytest.raises(ValueError, match="expected 'different-uuid'"):
        load_articulation_metadata(
            str(path),
            expected_uuid="different-uuid",
        )


def test_load_articulation_metadata_delta_target_returns_typed_value(
    tmp_path: Path,
) -> None:
    raw = _valid_metadata()
    operation = raw["articulation"]["joints"][0]["operations"]["open"]
    operation.pop("target_joint_fraction")
    operation["target_joint_delta"] = 1.25
    path = tmp_path / "metadata.json"
    _write_metadata(path, raw)

    metadata = load_articulation_metadata(str(path))

    assert metadata.joints[0].operations["open"].target_joint_delta == 1.25


@pytest.mark.parametrize(
    "targets",
    [
        {},
        {
            "target_joint_fraction": 0.5,
            "target_joint_delta": 1.0,
        },
    ],
)
def test_load_articulation_metadata_invalid_target_choice_raises_value_error(
    tmp_path: Path,
    targets: dict[str, float],
) -> None:
    raw = _valid_metadata()
    operation = raw["articulation"]["joints"][0]["operations"]["open"]
    operation.pop("target_joint_fraction")
    operation.update(targets)
    path = tmp_path / "metadata.json"
    _write_metadata(path, raw)

    with pytest.raises(ValueError, match="exactly one"):
        load_articulation_metadata(str(path))


@pytest.mark.parametrize("target", [0.0, -0.1, float("inf")])
def test_load_articulation_metadata_invalid_delta_raises_value_error(
    tmp_path: Path,
    target: float,
) -> None:
    raw = _valid_metadata()
    operation = raw["articulation"]["joints"][0]["operations"]["open"]
    operation.pop("target_joint_fraction")
    operation["target_joint_delta"] = target
    path = tmp_path / "metadata.json"
    _write_metadata(path, raw)

    with pytest.raises(ValueError, match="target_joint_delta"):
        load_articulation_metadata(str(path))


def test_load_articulation_metadata_missing_outcome_link_raises_value_error(
    tmp_path: Path,
) -> None:
    raw = _valid_metadata()
    del raw["articulation"]["joints"][0]["outcome_link"]
    path = tmp_path / "metadata.json"
    _write_metadata(path, raw)

    with pytest.raises(ValueError, match="outcome_link"):
        load_articulation_metadata(str(path))


def test_load_metadata_missing_interaction_link_raises_value_error(
    tmp_path: Path,
) -> None:
    raw = _valid_metadata()
    del raw["articulation"]["joints"][0]["operations"]["open"][
        "interaction_link"
    ]
    path = tmp_path / "metadata.json"
    _write_metadata(path, raw)

    with pytest.raises(ValueError, match="interaction_link"):
        load_articulation_metadata(str(path))


def test_load_articulation_metadata_legacy_target_link_raises_value_error(
    tmp_path: Path,
) -> None:
    raw = _valid_metadata()
    joint = raw["articulation"]["joints"][0]
    joint["target_link"] = joint.pop("outcome_link")
    path = tmp_path / "metadata.json"
    _write_metadata(path, raw)

    with pytest.raises(ValueError, match="outcome_link"):
        load_articulation_metadata(str(path))


def _joint(
    semantic_name: str,
    *operations: str,
) -> JointOperationMeta:
    return JointOperationMeta(
        joint_name=f"{semantic_name}_joint",
        semantic_name=semantic_name,
        outcome_link=f"{semantic_name}_link",
        operations={
            operation: ArticulationOperationMeta(
                interaction_link=f"{semantic_name}_link",
                initial_joint_position=0.0,
                target_joint_fraction=1.0,
            )
            for operation in (operations or ("open",))
        },
    )


def test_get_joint_returns_the_joint_declaring_that_name() -> None:
    joints = (_joint("door"), _joint("control knob", "rotate"))

    found = get_joint(joints, "control knob")

    assert found.joint_name == "control knob_joint"
    assert found.outcome_link == "control knob_link"


def test_get_joint_does_not_match_the_mechanical_name() -> None:
    """Lookup is by semantic name; the joint_name must not match."""
    joints = (_joint("door"),)

    with pytest.raises(KeyError, match="No joint named"):
        get_joint(joints, "door_joint")


def test_get_joint_unknown_name_lists_the_available_ones() -> None:
    joints = (_joint("door"), _joint("lid"))

    with pytest.raises(KeyError) as exc_info:
        get_joint(joints, "hatch")

    message = str(exc_info.value)
    assert "hatch" in message
    assert "door" in message
    assert "lid" in message


def test_get_joint_on_empty_joints_raises_key_error() -> None:
    with pytest.raises(KeyError, match="No joint named"):
        get_joint((), "door")


def test_get_operation_returns_the_declared_parameters() -> None:
    joints = (_joint("lid", "open", "close"),)

    meta = get_operation(joints, "lid", "close")

    assert meta.target_joint_fraction == 1.0
    assert meta.initial_joint_position == 0.0


def test_get_operation_unsupported_lists_the_supported_ones() -> None:
    """Operations are sparse: a knob rotates but does not open."""
    joints = (_joint("control knob", "rotate"),)

    with pytest.raises(KeyError) as exc_info:
        get_operation(joints, "control knob", "open")

    message = str(exc_info.value)
    assert "control knob" in message
    assert "open" in message
    assert "rotate" in message


def test_get_operation_unknown_joint_reports_the_joint_not_the_operation() -> (
    None
):
    joints = (_joint("door"),)

    with pytest.raises(KeyError, match="No joint named"):
        get_operation(joints, "hatch", "open")
