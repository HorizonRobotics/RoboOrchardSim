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

"""Registry-backed instruction template helpers."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any

from typing_extensions import Literal

from robo_orchard_sim.tasks.instructions.base import InstructionWrapper

InstructionTemplate = Mapping[str, Any]

INSTRUCTION_TEMPLATE_REGISTRY: dict[str, InstructionTemplate] = {}


def register_instruction_template(
    name: str,
    template: InstructionTemplate,
) -> InstructionTemplate:
    """Register an in-memory instruction template by name."""
    registered = INSTRUCTION_TEMPLATE_REGISTRY.get(name)
    if registered == template:
        return template
    if registered is not None:
        raise ValueError(
            f"Duplicate instruction template registered: {name!r}."
        )
    INSTRUCTION_TEMPLATE_REGISTRY[name] = template
    return template


def get_instruction_template(template_name: str) -> InstructionTemplate:
    """Return the registered template payload for template_name."""
    try:
        return INSTRUCTION_TEMPLATE_REGISTRY[template_name]
    except KeyError as exc:
        known_templates = ", ".join(sorted(INSTRUCTION_TEMPLATE_REGISTRY))
        raise ValueError(
            "Unknown instruction template "
            f"{template_name!r}. Known templates: {known_templates}."
        ) from exc


def build_instruction_wrapper(
    template_name: str,
    template_mode: Literal["fixed", "variants"] = "fixed",
    actor_description_mode: Literal["raw", "seen", "unseen"] = "raw",
) -> InstructionWrapper:
    """Build an instruction wrapper from the registered template name."""
    get_instruction_template(template_name)
    return InstructionWrapper(
        template_name,
        template_mode=template_mode,
        actor_description_mode=actor_description_mode,
    )


register_instruction_template(
    "place_a2b_default",
    {
        "fixed": (
            "Pick up {actor1.description} and place in {actor2.description}"
        ),
        "variants": [
            "Grab {actor1.description} using robot arm and place it in {actor2.description}.",
            "Hold {actor1.description} with robot arm and place it into {actor2.description}.",
            "Grab {actor1.description} using robot arm and release it into {actor2.description}.",
            "Use robot arm to pick {actor1.description} and place in {actor2.description}.",
            "Grasp {actor1.description} with robot arm and place it inside {actor2.description}.",
            "Use robot arm to grab {actor1.description} and set in {actor2.description}.",
            "Use robot arm to grasp {actor1.description} and set in {actor2.description}.",
            "Pick {actor1.description} using robot arm and set it into {actor2.description}",
            "Lift {actor1.description} using robot arm and drop it into {actor2.description}.",
            "Pick {actor1.description} with robot arm and set it into {actor2.description}.",
            "Pick {actor1.description} up with robot arm and set it in {actor2.description}.",
            "Take {actor1.description} using robot arm and set it inside {actor2.description}.",
            "Lift {actor1.description} with robot arm and drop it into {actor2.description}.",
            "Take {actor1.description} using robot arm and place it into {actor2.description}.",
            "Use robot arm to hold {actor1.description} and place it in {actor2.description}.",
            "Use robot arm to hold {actor1.description} and place in {actor2.description}.",
            "Grasp {actor1.description} using robot arm and set it into {actor2.description}.",
            "Pick {actor1.description} with robot arm and release it into {actor2.description}.",
            "Lift {actor1.description} using robot arm and put it into {actor2.description}.",
            "Grasp {actor1.description} with robot arm and place it into {actor2.description}.",
            "Grab {actor1.description} with robot arm and drop it inside {actor2.description}.",
            "Grasp {actor1.description} with robot arm and drop it into {actor2.description}.",
            "Pick {actor1.description} using robot arm and put it into {actor2.description}.",
            "Use robot arm to pick {actor1.description} and drop in {actor2.description}.",
            "Use robot arm to grab {actor1.description} and drop it in {actor2.description}.",
            "Take {actor1.description} with robot arm and place it inside {actor2.description}.",
            "Pick up {actor1.description} with robot arm and drop into {actor2.description}.",
            "Lift {actor1.description} with robot arm and set it in {actor2.description}.",
            "Take {actor1.description} with robot arm and drop it into {actor2.description}.",
            "Grab {actor1.description} using robot arm and set it into {actor2.description}",
            "Use robot arm to grasp {actor1.description} and place into {actor2.description}.",
            "Lift {actor1.description} with robot arm and release it into {actor2.description}",
            "Take {actor1.description} using robot arm and release it into {actor2.description}.",
            "Grasp {actor1.description} with robot arm and set it inside {actor2.description}.",
            "Use robot arm to take {actor1.description} and put it in {actor2.description}.",
            "Use robot arm to lift {actor1.description} and drop into {actor2.description}.",
            "Use robot arm to pick {actor1.description} and place into {actor2.description}.",
            "Take {actor1.description} using robot arm and drop it into {actor2.description}.",
            "Pick {actor1.description} using robot arm and drop it in {actor2.description}.",
            "Pick up {actor1.description} using robot arm and put it in {actor2.description}.",
            "Use robot arm to lift {actor1.description} and set in {actor2.description}.",
            "Use robot arm to lift {actor1.description} and put it in {actor2.description}.",
            "Grasp {actor1.description} using robot arm and put it in {actor2.description}.",
            "Grab {actor1.description} with robot arm and release it into {actor2.description}.",
            "Take {actor1.description} with robot arm and set it into {actor2.description}.",
            "Grab {actor1.description} with robot arm and place it inside {actor2.description}.",
            "Hold {actor1.description} with robot arm and release it into {actor2.description}.",
            "Lift {actor1.description} with robot arm and place it into {actor2.description}.",
            "Hold {actor1.description} using robot arm and place it into {actor2.description}.",
            "Hold {actor1.description} using robot arm and place it in {actor2.description}.",
            "Use robot arm to lift {actor1.description} and place into {actor2.description}.",
            "Pick up {actor1.description} using robot arm and put in {actor2.description}.",
            "Lift {actor1.description} with robot arm and release it into {actor2.description}.",
            "Pick {actor1.description} up with robot arm and place it in {actor2.description}.",
            "Use robot arm to grasp {actor1.description} and place it in {actor2.description}.",
            "Grasp {actor1.description} using robot arm and place it inside {actor2.description}.",
            "Grab {actor1.description} with robot arm and set it inside {actor2.description}.",
            "Use robot arm to grab {actor1.description} and drop into {actor2.description}.",
            "Take {actor1.description} with robot arm and drop it in {actor2.description}.",
            "Use robot arm to pick {actor1.description} and drop into {actor2.description}.",
        ],
    },
)

register_instruction_template(
    "pick_default",
    {
        "fixed": "Pick up {actor1.description}",
        "variants": [
            "Pick up {actor1.description}.",
            "Grab {actor1.description}.",
            "Lift {actor1.description}.",
            "Pick {actor1.description} up.",
            "Grasp {actor1.description}.",
            "Take {actor1.description}.",
            "Lift the {actor1.description}.",
            "Pick up the {actor1.description}.",
        ],
    },
)
