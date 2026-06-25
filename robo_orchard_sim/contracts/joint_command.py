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

"""Unified joint-command payload shared by policies and embodiments."""

from __future__ import annotations
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

EnvAction = dict[str, torch.Tensor]

_JOINT_RANGE_PATTERN = re.compile(
    r"^(?P<prefix>.+)\[(?P<start>\d+)-(?P<end>\d+)\]$"
)


def resolve_joint_name_specs(
    joint_specs: Sequence[str],
    *,
    available_joint_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Expand compact joint specs into concrete canonical joint names.

    Supported specs are concrete names such as ``"left_joint1"`` and
    contiguous ranges such as ``"left_joint[1-6]"``.
    """
    joint_names: list[str] = []
    for spec in joint_specs:
        joint_names.extend(_resolve_one_joint_name_spec(spec))

    if available_joint_names is not None:
        missing = [
            joint_name
            for joint_name in joint_names
            if joint_name not in available_joint_names
        ]
        if missing:
            raise ValueError(
                "Joint specs resolved names that are not available: "
                f"{', '.join(missing)}. Available joints: "
                f"{', '.join(available_joint_names)}."
            )

    return tuple(joint_names)


def _resolve_one_joint_name_spec(joint_spec: str) -> tuple[str, ...]:
    match = _JOINT_RANGE_PATTERN.match(joint_spec)
    if match is None:
        return (joint_spec,)

    start = int(match.group("start"))
    end = int(match.group("end"))
    if end < start:
        raise ValueError(
            f"Joint range spec '{joint_spec}' has end smaller than start."
        )

    prefix = match.group("prefix")
    return tuple(f"{prefix}{idx}" for idx in range(start, end + 1))


@dataclass(frozen=True)
class UnifiedJointCommand:
    """Robot-level joint command with explicit column metadata.

    Attributes:
        values: Joint command values with shape ``[batch, joint_dim]``.
        joint_names: Canonical joint name for each ``values`` column.
    """

    values: torch.Tensor
    joint_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(
                "UnifiedJointCommand.values must be a 2D tensor with shape "
                f"[batch, joint_dim], got shape {tuple(self.values.shape)}."
            )
        if not self.joint_names:
            raise ValueError(
                "UnifiedJointCommand.joint_names must not be empty."
            )
        if len(self.joint_names) != self.values.shape[1]:
            raise ValueError(
                "UnifiedJointCommand joint metadata length must match "
                f"values.shape[1], got {len(self.joint_names)} names for "
                f"joint_dim {self.values.shape[1]}."
            )
        duplicated = _find_duplicates(self.joint_names)
        if duplicated:
            raise ValueError(
                "UnifiedJointCommand.joint_names must be unique. Duplicates: "
                f"{', '.join(duplicated)}."
            )

    @classmethod
    def from_specs(
        cls,
        values: torch.Tensor,
        joint_specs: Sequence[str],
    ) -> "UnifiedJointCommand":
        """Create a command by expanding compact joint specs."""
        return cls(
            values=values,
            joint_names=resolve_joint_name_specs(joint_specs),
        )

    def select(self, *joint_specs: str) -> torch.Tensor:
        """Return columns matching the requested joint specs."""
        requested_joint_names = resolve_joint_name_specs(
            joint_specs,
            available_joint_names=self.joint_names,
        )
        indices = [
            self.joint_names.index(joint_name)
            for joint_name in requested_joint_names
        ]
        return self.values[:, indices]

    def select_if_present(self, *joint_specs: str) -> torch.Tensor | None:
        """Return selected columns only if all requested joints are present."""
        requested_joint_names = resolve_joint_name_specs(joint_specs)
        if any(
            joint_name not in self.joint_names
            for joint_name in requested_joint_names
        ):
            return None
        return self.select(*joint_specs)

    @classmethod
    def merge(
        cls,
        *commands: "UnifiedJointCommand",
    ) -> "UnifiedJointCommand":
        """Merge multiple commands, later values override earlier ones.

        When the same joint name appears in more than one command the
        column from the **last** command wins.  Batch sizes must match.
        """
        if not commands:
            raise ValueError(
                "UnifiedJointCommand.merge requires at least one command."
            )
        if len(commands) == 1:
            return commands[0]

        batch = commands[0].values.shape[0]
        device = commands[0].values.device

        ordered_names: list[str] = []
        name_to_col: dict[str, torch.Tensor] = {}
        for command in commands:
            if command.values.shape[0] != batch:
                raise ValueError(
                    "All UnifiedJointCommands passed to merge must "
                    "share the same batch dimension."
                )
            for col_idx, name in enumerate(command.joint_names):
                if name not in name_to_col:
                    ordered_names.append(name)
                name_to_col[name] = command.values[:, col_idx]

        merged = torch.stack(
            [name_to_col[n] for n in ordered_names],
            dim=1,
        ).to(device=device)
        return cls(values=merged, joint_names=tuple(ordered_names))


class EnvActionState:
    """Maintain a complete env action across partial action updates."""

    def __init__(self, action: EnvAction) -> None:
        self._action = self.clone(action)

    @classmethod
    def from_env(cls, env: Any) -> "EnvActionState":
        """Create state initialized to hold current joint positions."""
        return cls(cls.build_hold_position(env))

    @staticmethod
    def build_hold_position(env: Any) -> EnvAction:
        """Build a hold-position env action from current joint state.

        The env action schema is owned by ``env.action_manager``.  For each
        configured action term, read the current joint positions from the
        term's scene asset and use those positions as that term's action.
        """
        action: EnvAction = {}
        for term_name in env.action_manager.active_terms:
            term_cfg = env.action_manager.cfg.terms[term_name]
            asset_cfg = term_cfg.asset_cfg
            asset = env.scene[asset_cfg.name]
            action[term_name] = asset.data.joint_pos[
                :, asset_cfg.joint_ids
            ].clone()
        return action

    def action(self) -> EnvAction:
        """Return the current complete env action."""
        return self.clone(self._action)

    def update(self, partial_action: EnvAction) -> EnvAction:
        """Overlay a partial env action onto the current complete action."""
        for term_name, term_action in partial_action.items():
            if term_name not in self._action:
                raise KeyError(
                    f"Translated action term {term_name!r} is not "
                    "configured in the env action manager."
                )
            self._action[term_name] = term_action.clone()
        return self.action()

    @staticmethod
    def clone(action: EnvAction) -> EnvAction:
        """Return a tensor-cloned env action dictionary."""
        return {
            term_name: term_action.clone()
            for term_name, term_action in action.items()
        }


def _find_duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicated: list[str] = []
    for value in values:
        if value in seen and value not in duplicated:
            duplicated.append(value)
        seen.add(value)
    return tuple(duplicated)
