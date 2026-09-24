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

"""Tests for single-task data-synthesis validation boundaries."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

from robo_orchard_sim.pipeline.data_synthesis.single_task import (
    TaskDataSynthesisRunner,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionActor,
    InstructionWrapper,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.selector import create_selector


class CandidateTask:
    """Task-facing candidate provider with a multi-entry original pool."""

    roles = {"pick": SimpleNamespace(cardinality="one")}

    def get_role_candidates(self, role_id, *, swap=False):
        return [TargetRef("objects/apple"), TargetRef("objects/banana")]


def _scene_actor(
    *,
    category: str,
    actor_type: str,
    uuid: str,
    attributes: dict[str, tuple[str, ...]],
):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            category=category,
            actor_type=actor_type,
            uuid=uuid,
            attributes=attributes,
        )
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 10, 100])
def test_synthesis_binding_swap_disabled_always_preserves_first_candidate(
    seed,
):
    registry = TaskDataSynthesisRunner.build_role_registry(
        runtime_task=CandidateTask(), seed=seed, swap_enabled=False
    )

    assert registry.resolve_one("pick") == TargetRef("objects/apple")


@pytest.mark.parametrize("seed", [0, 42, 101])
def test_synthesis_binding_reordered_roles_preserve_seeded_targets(
    seed,
):
    roles = {
        "pick": SimpleNamespace(cardinality="one"),
        "place": SimpleNamespace(cardinality="one"),
        "all": SimpleNamespace(cardinality="many"),
    }
    pool = [
        TargetRef("cabinet", "door", "open"),
        TargetRef("cabinet", "door", "close"),
        TargetRef("cabinet", "drawer", "open"),
    ]
    expected = {}
    for role_id, spec in roles.items():
        expected[role_id] = (
            create_selector("random", seed=seed, role_id=role_id).select(
                pool, 1, 0
            )[0]
            if spec.cardinality == "one"
            else pool
        )
    for reverse in (False, True):
        task = SimpleNamespace(
            roles=dict(reversed(roles.items())) if reverse else roles,
            get_role_candidates=lambda role_id, swap: pool,
        )
        registry = TaskDataSynthesisRunner.build_role_registry(
            runtime_task=task, seed=seed, swap_enabled=True
        )
        assert registry.bindings() == expected


def test_build_validator_fixed_horizon_validator_raises_value_error():
    runner = object.__new__(TaskDataSynthesisRunner)
    validator = SimpleNamespace(fixed_horizon=True)
    task = SimpleNamespace(
        roles={},
        build_validator=lambda context: validator,
    )
    embodiment = SimpleNamespace(get_robot_info_cfgs=lambda: {})

    with pytest.raises(ValueError, match="not supported by data synthesis"):
        runner.build_validator(
            runtime_task=task,
            embodiment=embodiment,
        )


@pytest.mark.parametrize(
    ("template_name", "actors", "expected_text", "expected"),
    [
        (
            "pick_default",
            {
                "actor1": InstructionActor(
                    uuid="u-apple",
                    category="apple",
                    description="red apple",
                    raw_description="red apple",
                )
            },
            "Pick up red apple",
            {"pick": "red apple"},
        ),
        (
            "place_a2b_default",
            {
                "actor1": InstructionActor(
                    uuid="u-apple",
                    category="apple",
                    description="red apple",
                    raw_description="red apple",
                ),
                "actor2": InstructionActor(
                    uuid="u-bowl",
                    category="bowl",
                    description="blue bowl",
                    raw_description="blue bowl",
                ),
            },
            "Pick up red apple and place in blue bowl",
            {"pick": "red apple", "place": "blue bowl"},
        ),
        (
            "pick_attribute",
            {
                "actor1": InstructionActor(
                    uuid="u-apple",
                    category="apple",
                    description="apple",
                    raw_description="apple",
                    attribute_name="color",
                    attribute_value="red",
                )
            },
            "Pick red apple",
            {"pick": "red apple"},
        ),
        (
            "spatial_pick_default",
            {
                "obj": InstructionActor(
                    uuid="u-tomato",
                    category="tomato",
                    description="tomato",
                    raw_description="tomato",
                ),
                "ref_obj": InstructionActor(
                    uuid="u-apple",
                    category="apple",
                    description="apple",
                    raw_description="apple",
                ),
                "spatial_relation": "to the left of",
            },
            "Pick up the tomato to the left of the apple.",
            {"pick": "tomato"},
        ),
    ],
    ids=[
        "pick-category",
        "place-a2b",
        "pick-attribute",
        "spatial-direction",
    ],
)
def test_render_actor_descriptions_registered_template_returns_role_values(
    template_name: str,
    actors: Mapping[str, object],
    expected_text: str,
    expected: dict[str, str],
) -> None:
    wrapper = InstructionWrapper(template_name, template_mode="fixed")

    text, descriptions = wrapper.render_with_actor_descriptions(actors=actors)

    assert (text, descriptions) == (expected_text, expected)


def test_render_actor_descriptions_seen_mode_reuses_rendered_value() -> None:
    wrapper = InstructionWrapper(
        "pick_default",
        template_mode="fixed",
        actor_description_mode="seen",
    )
    actor = InstructionActor(
        uuid="u-apple",
        category="apple",
        description="stale description",
        raw_description="apple",
        seen_descriptions=["red apple", "green apple"],
    )

    text, descriptions = wrapper.render_with_actor_descriptions(
        actors={"actor1": actor},
        actor_description_seed=7,
    )

    assert text == f"Pick up {descriptions['pick']}"
    assert descriptions["pick"] in {"red apple", "green apple"}


def test_episode_metadata_role_binding_records_description_on_bound_actor(
    monkeypatch,
):
    monkeypatch.setattr(
        "robo_orchard_sim.utils.env_utils.bbox_of",
        lambda cfg: None,
    )
    runner = object.__new__(TaskDataSynthesisRunner)
    role_registry = RoleRegistry(num_envs=1)
    role_registry.bind_one(0, "pick", TargetRef("distractor_apple"))
    context = SimpleNamespace(
        role_registry=role_registry,
        init_state_of=lambda name, env_idx: torch.zeros(7),
        final_state_of=lambda name, env_idx: torch.ones(7),
    )
    env = SimpleNamespace(
        scene={
            "pick_apple": _scene_actor(
                category="apple",
                actor_type="rigid_object",
                uuid="pick-apple",
                attributes={},
            ),
            "distractor_apple": _scene_actor(
                category="apple",
                actor_type="rigid_object",
                uuid="distractor-apple",
                attributes={},
            ),
        }
    )

    metadata = runner.build_episode_metadata(
        env=env,
        context=context,
        scene_names=["pick_apple", "distractor_apple"],
        validator_output=SimpleNamespace(success=True, progress=1.0),
        instruction_actor_descriptions={"pick": "red apple"},
    )

    assert {
        name: actor_metadata["description"]
        for name, actor_metadata in metadata["actors"].items()
        if "description" in actor_metadata
    } == {"distractor_apple": "red apple"}
