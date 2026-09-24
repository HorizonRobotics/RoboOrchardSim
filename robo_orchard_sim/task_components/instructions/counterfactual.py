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

"""Counterfactual instruction metadata and absent-referent resolution."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from robo_orchard_sim.task_components.instructions.base import (
    InstructionActor,
    InstructionAttributeName,
    InstructionRenderError,
    InstructionWrapper,
)
from robo_orchard_sim.task_components.instructions.registry import (
    get_instruction_template,
)

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolver,
    )
    from robo_orchard_sim.orchard_env.assets import RigidObjectSpec

SemanticField = Literal[
    "super_category",
    "category",
    "color",
    "shape",
    "material",
    "size_bucket",
]
_SEMANTIC_FIELDS = frozenset(get_args(SemanticField))

CounterfactualCondition = Literal[
    "empty",
    "generic_object",
    "absent_object",
]
_COUNTERFACTUAL_CONDITIONS = frozenset(get_args(CounterfactualCondition))

__all__ = [
    "AbsentObjectRule",
    "BoundCounterfactualInstruction",
    "CounterfactualCondition",
    "bind_instruction_assets",
    "counterfactual_instruction_actors",
    "counterfactual_condition_for_task",
    "counterfactual_condition_from_template",
    "resolve_instruction_assets",
]


def counterfactual_condition_from_template(
    template: str,
) -> CounterfactualCondition | None:
    """Return the counterfactual condition declared by one template."""
    condition = get_instruction_template(template).get(
        "counterfactual_condition"
    )
    if condition is None:
        return None
    if (
        not isinstance(condition, str)
        or condition not in _COUNTERFACTUAL_CONDITIONS
    ):
        raise ValueError(
            f"Invalid counterfactual condition {condition!r} "
            f"for instruction template {template!r}."
        )
    return cast(CounterfactualCondition, condition)


def counterfactual_condition_for_task(
    task: Any,
) -> CounterfactualCondition | None:
    """Return the condition encoded by a task's instruction template."""
    if not isinstance(task.instruction, InstructionWrapper):
        return None
    return counterfactual_condition_from_template(task.instruction.template)


@dataclass(frozen=True, slots=True)
class AbsentObjectRule:
    """Describe an absent rigid-object referent relative to scene assets."""

    match: tuple[SemanticField, ...] = ()
    differ: tuple[SemanticField, ...] = ()
    referent_fields: tuple[SemanticField, ...] = ("category",)
    required_tags: frozenset[str] = frozenset({"is_graspable"})

    @classmethod
    def from_instruction(
        cls,
        instruction: "InstructionWrapper",
    ) -> "AbsentObjectRule":
        """Build the shared absent-referent rule for one instruction."""
        attribute_name = instruction.attribute_name
        if attribute_name is None:
            return cls(
                differ=("category",),
                referent_fields=("category",),
            )
        semantic_attribute = cast(SemanticField, attribute_name)
        return cls(
            match=("category",),
            differ=(semantic_attribute,),
            referent_fields=("category", semantic_attribute),
        )

    def __post_init__(self) -> None:
        fields = set(self.match + self.differ + self.referent_fields)
        unknown = fields - _SEMANTIC_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown absent-object semantic fields: {sorted(unknown)}."
            )
        if not self.referent_fields:
            raise ValueError("referent_fields must be non-empty.")
        overlap = set(self.match) & set(self.differ)
        if overlap:
            raise ValueError(
                "Absent-object match and differ fields overlap: "
                f"{sorted(overlap)}."
            )


class BoundCounterfactualInstruction(InstructionWrapper):
    """Instruction with pre-resolved actors that are absent from the scene."""

    def __init__(
        self,
        instruction: InstructionWrapper,
        actors: Mapping[str, InstructionActor],
    ) -> None:
        super().__init__(
            instruction.template,
            template_mode=cast(
                Literal["fixed", "variants"],
                instruction.template_mode,
            ),
            actor_description_mode=cast(
                Literal["raw", "seen", "unseen"],
                instruction.actor_description_mode,
            ),
            attribute_name=cast(
                InstructionAttributeName | None,
                instruction.attribute_name,
            ),
            strict=instruction.strict,
        )
        self._actors = dict(actors)

    def resolved_actors(self) -> dict[str, InstructionActor]:
        """Return a copy of the non-spawned actors used for rendering."""
        return dict(self._actors)


def _format_attribute_values(values: frozenset[str]) -> str:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} and {ordered[1]}"
    return ", ".join(ordered[:-1]) + f", and {ordered[-1]}"


def _actor_from_absent_spec(
    spec: "RigidObjectSpec",
    *,
    attribute_name: InstructionAttributeName | None,
) -> InstructionActor:
    """Build a literal actor description without requiring a caption file."""
    if spec.uuid is None or spec.category is None:
        raise InstructionRenderError(
            "Absent-object instruction actors require uuid and category."
        )
    category = spec.category.replace("_", " ")
    description = category
    attribute_value = None
    if attribute_name is not None:
        values = frozenset(
            value.strip().lower()
            for value in spec.attributes.get(attribute_name, ())
            if value.strip()
        )
        if not values:
            raise InstructionRenderError(
                f"Asset {spec.uuid!r} has no {attribute_name!r} value"
            )
        if attribute_name != "color" and len(values) != 1:
            raise InstructionRenderError(
                f"Asset {spec.uuid!r} has multiple {attribute_name!r} "
                f"values: {sorted(values)}"
            )
        attribute_value = _format_attribute_values(values)
        description = f"{attribute_value} {category}"
    return InstructionActor(
        uuid=spec.uuid,
        description=description,
        raw_description=description,
        category=category,
        seen_descriptions=[description],
        unseen_descriptions=[description],
        attribute_name=attribute_name,
        attribute_value=attribute_value,
    )


def bind_instruction_assets(
    instruction: InstructionWrapper,
    actor_specs: Mapping[str, "RigidObjectSpec"],
) -> InstructionWrapper:
    """Return an instruction bound to non-spawned rigid actors."""
    actors = {
        actor_name: _actor_from_absent_spec(
            spec,
            attribute_name=(
                cast(
                    InstructionAttributeName | None,
                    instruction.attribute_name,
                )
                if actor_name == "actor1"
                else None
            ),
        )
        for actor_name, spec in actor_specs.items()
    }
    return BoundCounterfactualInstruction(instruction, actors)


def counterfactual_instruction_actors(
    task: Any,
) -> dict[str, InstructionActor] | None:
    """Return counterfactual actors, or None for a normal instruction."""
    condition = counterfactual_condition_for_task(task)
    if condition is None:
        return None
    if condition != "absent_object":
        return {}
    instruction = task.instruction
    if not isinstance(instruction, BoundCounterfactualInstruction):
        raise RuntimeError(
            "Absent-object instruction was not resolved during task build."
        )
    return instruction.resolved_actors()


def resolve_instruction_assets(
    *,
    task: Any,
    resolver: "AssetResolver | None",
) -> dict[str, RigidObjectSpec]:
    """Resolve non-spawned rigid actors needed by the task instruction."""
    instruction = task.instruction
    if (
        instruction is None
        or counterfactual_condition_from_template(instruction.template)
        != "absent_object"
    ):
        return {}
    if resolver is None:
        raise ValueError(
            "Absent-object instruction resolution requires an AssetResolver."
        )

    from robo_orchard_sim.orchard_env.assets import RigidObjectSpec

    present_uuids: list[str] = []
    for spec in task.assets.flatten().values():
        if not isinstance(spec, RigidObjectSpec):
            raise NotImplementedError(
                "Absent-object counterfactual evaluation currently supports "
                "rigid-object scenes only."
            )
        if spec.uuid is None:
            raise ValueError(
                "Absent-object evaluation requires registry-backed rigid "
                f"object '{spec.scene_name}' with a UUID."
            )
        present_uuids.append(spec.uuid)

    actor_spec = resolver.resolve_absent_referent(
        present_uuids=list(dict.fromkeys(present_uuids)),
        rule=AbsentObjectRule.from_instruction(instruction),
    )
    if not isinstance(actor_spec, RigidObjectSpec):
        raise TypeError(
            "Absent-object resolution must produce a RigidObjectSpec, got "
            f"{type(actor_spec).__name__}."
        )
    return {"actor1": actor_spec}
