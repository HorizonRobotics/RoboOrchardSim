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

"""Typed articulation joint-operation metadata and filesystem loading."""

from __future__ import annotations
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator
from robo_orchard_core.utils.config import Config

from robo_orchard_sim.contracts.articulated_operation import (
    ArticulatedOperation,
)

OperationStartMode = Literal["metadata", "opposing_midpoint"]
OPPOSING_OPERATION_PAIRS: tuple[
    tuple[ArticulatedOperation, ArticulatedOperation], ...
] = (("open", "close"), ("pull", "push"))


class ArticulationRootMeta(Config):
    """Asset-level root mounting metadata."""

    fix_root_link: bool | None = None


class ArticulationOperationMeta(Config):
    """Asset-specific state for one semantic joint operation."""

    interaction_link: str | None = None
    interaction_links: tuple[str, ...] | None = None
    """Allowed interaction bodies; legacy interaction_link is also accepted."""
    initial_joint_position: float
    target_joint_fraction: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    target_joint_delta: float | None = Field(default=None, gt=0.0)

    @field_validator("interaction_links")
    @classmethod
    def validate_interaction_links(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """Require a non-empty list of unique, non-blank body names."""
        if value is not None and (
            not value
            or any(not name.strip() for name in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError(
                "interaction_links must contain unique non-empty names"
            )
        return value

    @property
    def effective_interaction_links(self) -> tuple[str, ...]:
        """Return the allowed bodies for interaction criteria."""
        if self.interaction_links is None:
            assert self.interaction_link is not None
            return (self.interaction_link,)
        return self.interaction_links

    @field_validator("interaction_link")
    @classmethod
    def validate_non_empty_interaction_link(
        cls, value: str | None
    ) -> str | None:
        """Reject empty or whitespace-only interaction link names."""
        if value is not None and not value.strip():
            raise ValueError("must be non-empty")
        return value

    @field_validator("initial_joint_position")
    @classmethod
    def validate_finite_initial_position(cls, value: float) -> float:
        """Reject NaN and infinite joint positions."""
        if not math.isfinite(value):
            raise ValueError("initial_joint_position must be finite")
        return value

    @field_validator("target_joint_delta")
    @classmethod
    def validate_finite_target_delta(cls, value: float | None) -> float | None:
        """Reject an infinite target joint displacement."""
        if value is not None and not math.isfinite(value):
            raise ValueError("target_joint_delta must be finite")
        return value

    @model_validator(mode="after")
    def validate_exactly_one_target(self) -> "ArticulationOperationMeta":
        """Require exactly one bounded or displacement-based joint target."""
        if (self.interaction_link is None) == (self.interaction_links is None):
            raise ValueError(
                "exactly one of interaction_link or interaction_links "
                "must be set"
            )
        targets = (self.target_joint_fraction, self.target_joint_delta)
        if sum(target is not None for target in targets) != 1:
            raise ValueError(
                "exactly one of target_joint_fraction or target_joint_delta "
                "must be set"
            )
        return self


class JointOperationMeta(Config):
    """Semantic operations supported by one mechanical joint."""

    joint_name: str
    semantic_name: str
    outcome_link: str
    operations: dict[ArticulatedOperation, ArticulationOperationMeta]

    @field_validator("joint_name", "semantic_name", "outcome_link")
    @classmethod
    def validate_non_empty_name(cls, value: str) -> str:
        """Reject empty or whitespace-only metadata names."""
        if not value.strip():
            raise ValueError("must be non-empty")
        return value

    @field_validator("operations")
    @classmethod
    def validate_non_empty_operations(
        cls,
        value: dict[ArticulatedOperation, ArticulationOperationMeta],
    ) -> dict[ArticulatedOperation, ArticulationOperationMeta]:
        """Require at least one semantic operation per joint."""
        if not value:
            raise ValueError("operations must be non-empty")
        return value


def _semantic_names(joints: Sequence[JointOperationMeta]) -> list[str]:
    return [joint.semantic_name for joint in joints]


def get_joint(
    joints: Sequence[JointOperationMeta],
    semantic_name: str,
) -> JointOperationMeta:
    """Look up one joint by the semantic part name it declares."""
    for joint in joints:
        if joint.semantic_name == semantic_name:
            return joint
    raise KeyError(
        f"No joint named '{semantic_name}'. "
        f"Available: {_semantic_names(joints)}."
    )


def get_operation(
    joints: Sequence[JointOperationMeta],
    semantic_name: str,
    operation: ArticulatedOperation,
) -> ArticulationOperationMeta:
    """Look up what one joint declares for one operation."""
    joint = get_joint(joints, semantic_name)
    try:
        return joint.operations[operation]
    except KeyError:
        raise KeyError(
            f"Joint '{semantic_name}' does not support '{operation}'. "
            f"Supported: {sorted(joint.operations)}."
        ) from None


def get_operation_start_position(
    joint: JointOperationMeta,
    operation: ArticulatedOperation,
    mode: OperationStartMode = "metadata",
) -> float:
    """Return an operation start using only this joint's annotations.

    Midpoint mode requires both opposing operations on the same joint.
    The returned position is unclipped; runtime soft limits apply on reset.
    """
    selected = get_operation((joint,), joint.semantic_name, operation)
    if mode == "metadata":
        return selected.initial_joint_position
    if mode != "opposing_midpoint":
        raise ValueError(f"Unsupported operation start mode: '{mode}'.")
    for pair in OPPOSING_OPERATION_PAIRS:
        if operation in pair and all(
            name in joint.operations for name in pair
        ):
            return (
                joint.operations[pair[0]].initial_joint_position
                + joint.operations[pair[1]].initial_joint_position
            ) / 2.0
    raise ValueError(
        f"Operation '{operation}' requires a complete opposing pair on the "
        f"same joint '{joint.semantic_name}' ({joint.joint_name}) for "
        f"opposing_midpoint reset. Supported: {sorted(joint.operations)}."
    )


class ArticulationMetadata(Config):
    """Strongly typed representation of metadata.json['articulation']."""

    uuid: str | None = None
    root: ArticulationRootMeta = Field(default_factory=ArticulationRootMeta)
    joints: tuple[JointOperationMeta, ...]

    @model_validator(mode="after")
    def validate_unique_joint_names(self) -> "ArticulationMetadata":
        """Reject empty metadata and duplicate mechanical joint names."""
        if not self.joints:
            raise ValueError("articulation.joints must be non-empty")
        names = [joint.joint_name for joint in self.joints]
        if len(names) != len(set(names)):
            raise ValueError(
                "articulation metadata contains duplicate joint_name values"
            )
        return self


def _metadata_context(path: Path, expected_uuid: str | None) -> str:
    if expected_uuid is None:
        return f"articulation metadata at '{path}'"
    return f"articulation metadata for asset '{expected_uuid}' at '{path}'"


def _read_json(path: Path, expected_uuid: str | None) -> dict[str, Any]:
    context = _metadata_context(path, expected_uuid)
    try:
        raw = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Unable to read {context}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {context}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must contain a JSON object")
    return raw


def load_articulation_metadata(
    metadata_path: str,
    *,
    expected_uuid: str | None = None,
) -> ArticulationMetadata:
    """Load and validate articulation semantics from an asset metadata file."""
    path = Path(metadata_path)
    raw = _read_json(path, expected_uuid)
    context = _metadata_context(path, expected_uuid)

    metadata_uuid = raw.get("uuid")
    if (
        expected_uuid is not None
        and metadata_uuid is not None
        and metadata_uuid != expected_uuid
    ):
        raise ValueError(
            f"{context} declares uuid '{metadata_uuid}', expected "
            f"'{expected_uuid}'"
        )

    articulation = raw.get("articulation")
    if not isinstance(articulation, dict):
        raise ValueError(f"{context} is missing an 'articulation' object")
    try:
        metadata = ArticulationMetadata.model_validate(articulation)
        if isinstance(metadata_uuid, str):
            metadata = metadata.model_copy(update={"uuid": metadata_uuid})
        return metadata
    except ValueError as exc:
        raise ValueError(f"Invalid {context}: {exc}") from exc


__all__ = [
    "ArticulationMetadata",
    "ArticulationOperationMeta",
    "ArticulationRootMeta",
    "JointOperationMeta",
    "OPPOSING_OPERATION_PAIRS",
    "OperationStartMode",
    "get_joint",
    "get_operation",
    "get_operation_start_position",
    "load_articulation_metadata",
]
