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

"""Layout-driven task variant that enriches instruction context from layout."""

from __future__ import annotations
from typing import Any

from robo_orchard_sim.orchard_env.task_templates.pick_task import PickTask
from robo_orchard_sim.task_components.instructions.base import InstructionActor


class LayoutSceneRef:
    """A reference to a scene object.

    The reference is resolved to an ``InstructionActor`` at
    instruction-build time.
    """

    __slots__ = ("scene_name",)

    def __init__(self, scene_name: str) -> None:
        self.scene_name = scene_name


LayoutContext = dict[str, Any]
"""Layout-derived context mapping.  Values may be plain strings/numbers
(used verbatim in templates) or :class:`LayoutSceneRef` instances
(resolved to ``InstructionActor`` at instruction-build time)."""


class LayoutTask(PickTask):
    """A :class:`PickTask` variant with layout-derived instruction context.

    ``layout_context`` values of type :class:`LayoutSceneRef` are
    resolved to :class:`InstructionActor` instances from the runtime
    scene in :meth:`build_instruction_context`.  Plain values pass
    through unchanged so templates can reference them directly.
    """

    def __init__(
        self,
        assets,
        params=None,
        instruction=None,
        layout_context: LayoutContext | None = None,
    ) -> None:
        super().__init__(assets=assets, params=params, instruction=instruction)
        self.layout_context = layout_context

    def build_instruction_context(
        self,
        env,
        *,
        actor_description_seed: int,
    ) -> dict[str, Any]:
        actors = super().build_instruction_context(
            env,
            actor_description_seed=actor_description_seed,
        )

        if self.layout_context is None or self.instruction is None:
            return actors

        for key, value in self.layout_context.items():
            if isinstance(value, LayoutSceneRef):
                scene_obj = env.scene[value.scene_name]
                actors[key] = InstructionActor.from_rigid_object(
                    scene_obj,
                    actor_description_mode=(
                        self.instruction.actor_description_mode
                    ),
                    actor_description_seed=actor_description_seed,
                )
            else:
                actors[key] = value

        if "obj" not in actors and "actor1" in actors:
            actors["obj"] = actors["actor1"]

        return actors
