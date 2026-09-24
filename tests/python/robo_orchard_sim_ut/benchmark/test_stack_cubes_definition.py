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

"""Configuration coverage for the stack-cubes benchmark."""

from typing import Any, cast

import pytest

from robo_orchard_sim.benchmark.manipulation.stack_cubes import (
    StackCubesTaskDefinition,
)
from robo_orchard_sim.benchmark.manipulation.stack_cubes.colors import (
    CUBE_COLOR_NAMES,
    CUBE_COLOR_PALETTE,
    CUBE_COLOR_SRGB,
)
from robo_orchard_sim.benchmark.registry import build_task
from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.task_templates.stack_cubes_task import (
    STACK_SIZE,
    StackCubesTask,
)

_SELECTED_COLORS = ("orange", "blue", "magenta", "green")


class _FixedCubeResolver:
    def resolve(self, asset_configs):
        config = asset_configs["members"]
        return {
            "members": RigidObjectSpec(
                name=config["prim_name"],
                usd_path=config["usd_path"],
                interaction_path=config["interaction_path"],
            )
        }

    def sample_without_replacement(self, candidates, *, count):
        assert tuple(candidates) == CUBE_COLOR_NAMES
        assert count == STACK_SIZE
        return list(_SELECTED_COLORS)


def _task() -> StackCubesTask:
    orchard_env = build_task(
        "stack_cubes",
        resolver=cast(Any, _FixedCubeResolver()),
    )
    return cast(StackCubesTask, orchard_env.task)


def test_stack_cubes_palette_has_ten_unique_colors():
    assert (
        len(CUBE_COLOR_PALETTE) == len(set(CUBE_COLOR_PALETTE.values())) == 10
    )
    assert CUBE_COLOR_SRGB["green"] == (0.05, 0.60, 0.20)
    assert CUBE_COLOR_PALETTE["green"] == pytest.approx(
        (0.00393594, 0.31854678, 0.03310477)
    )


def test_stack_cubes_seeded_colors_build_ordered_members():
    task = _task()

    assert [member.name for member in task.members] == [
        f"{color}_cube" for color in _SELECTED_COLORS
    ]
    assert [member.visual_color for member in task.members] == [
        CUBE_COLOR_PALETTE[color] for color in _SELECTED_COLORS
    ]


def test_stack_cubes_definition_pose_reset_uses_non_overlap_sampling():
    params = StackCubesTaskDefinition.resolve_task_params()

    assert params["pose_reset"]["mode"] == "random_non_overlap"
