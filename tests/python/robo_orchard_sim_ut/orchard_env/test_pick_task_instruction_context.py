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

"""Tests for pick task instruction context rendering."""

from __future__ import annotations
import json
from types import SimpleNamespace

import pytest

from robo_orchard_sim.orchard_env.assets import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_templates.pick_task import (
    PickTask,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionRenderError,
    InstructionWrapper,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
)


def _context_bound_to(task) -> ValidatorContext:
    """A context whose pick role names the task's only candidate."""
    registry = RoleRegistry(num_envs=1)
    registry.bind_one(0, "pick", TargetRef(task.pick_object.scene_name))
    return ValidatorContext(robot=None, role_registry=registry)


def _write_caption(path, *, uuid: str, raw: str) -> None:
    path.write_text(
        json.dumps(
            {
                "uuid": uuid,
                "raw": raw,
                "seen": [raw],
            }
        ),
        encoding="utf-8",
    )


def test_pick_task_attribute_instruction_color_renders_attribute_category(
    tmp_path,
) -> None:
    caption_path = tmp_path / "caption_candidates.json"
    _write_caption(caption_path, uuid="u-peach-001", raw="peach")
    spec = RigidObjectSpec(
        name="pick_object",
        usd_path="/tmp/pick.usd",
        caption_path=str(caption_path),
        uuid="u-peach-001",
        category="peach",
        attributes={"color": ("yellow",)},
    )
    task = PickTask(
        TaskAssets(role_candidates={"pick": [spec]}),
        instruction=InstructionWrapper(
            "pick_attribute",
            template_mode="fixed",
            actor_description_mode="raw",
            attribute_name="color",
        ),
    )
    env = SimpleNamespace(
        scene={
            task.pick_object.scene_name: SimpleNamespace(
                cfg=spec.to_isaac_cfg()
            )
        }
    )

    actors = task.build_instruction_context(
        env, actor_description_seed=0, context=_context_bound_to(task)
    )
    instruction = task.instruction.render(actors=actors)

    assert instruction == "Pick yellow peach"


def test_pick_task_attribute_instruction_three_colors_renders_color_phrase(
    tmp_path,
) -> None:
    caption_path = tmp_path / "caption_candidates.json"
    _write_caption(caption_path, uuid="u-peach-colorful", raw="peach")
    spec = RigidObjectSpec(
        name="pick_object",
        usd_path="/tmp/pick.usd",
        caption_path=str(caption_path),
        uuid="u-peach-colorful",
        category="peach",
        attributes={"color": ("yellow", "green", "red")},
    )
    task = PickTask(
        TaskAssets(role_candidates={"pick": [spec]}),
        instruction=InstructionWrapper(
            "pick_attribute",
            template_mode="fixed",
            actor_description_mode="raw",
            attribute_name="color",
        ),
    )
    env = SimpleNamespace(
        scene={
            task.pick_object.scene_name: SimpleNamespace(
                cfg=spec.to_isaac_cfg()
            )
        }
    )

    actors = task.build_instruction_context(
        env, actor_description_seed=0, context=_context_bound_to(task)
    )
    instruction = task.instruction.render(actors=actors)

    assert instruction == "Pick green, red, and yellow peach"


def test_pick_task_attribute_instruction_missing_attribute_raises(
    tmp_path,
) -> None:
    caption_path = tmp_path / "caption_candidates.json"
    _write_caption(caption_path, uuid="u-peach-002", raw="peach")
    spec = RigidObjectSpec(
        name="pick_object",
        usd_path="/tmp/pick.usd",
        caption_path=str(caption_path),
        uuid="u-peach-002",
        category="peach",
    )
    task = PickTask(
        TaskAssets(role_candidates={"pick": [spec]}),
        instruction=InstructionWrapper(
            "pick_attribute",
            template_mode="fixed",
            actor_description_mode="raw",
            attribute_name="material",
        ),
    )
    env = SimpleNamespace(
        scene={
            task.pick_object.scene_name: SimpleNamespace(
                cfg=spec.to_isaac_cfg()
            )
        }
    )

    with pytest.raises(InstructionRenderError, match="no 'material' value"):
        task.build_instruction_context(
            env, actor_description_seed=0, context=_context_bound_to(task)
        )
