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

"""Tests for counterfactual instruction metadata."""

from types import SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pytest

from robo_orchard_sim.asset_manager.registry.types import AssetMeta
from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
    AssetResolutionError,
    AssetResolver,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionActor,
    InstructionWrapper,
)
from robo_orchard_sim.task_components.instructions.counterfactual import (
    AbsentObjectRule,
    counterfactual_condition_from_template,
    resolve_instruction_assets,
)
from robo_orchard_sim.task_components.instructions.registry import (
    get_instruction_template,
)


def _meta(
    uuid: str,
    *,
    category: str = "mug",
    shape: frozenset[str] = frozenset({"round"}),
    material: frozenset[str] = frozenset({"ceramic"}),
) -> AssetMeta:
    return AssetMeta(
        uuid=uuid,
        asset_id=uuid,
        relative_path=f"{uuid}/metadata.json",
        domain="test",
        super_category="objects",
        category=category,
        name=category,
        description=category,
        color=frozenset({"green"}),
        shape=shape,
        material=material,
        real_height=0.1,
        real_mass=0.1,
        min_height=0.0,
        max_height=1.0,
        min_mass=0.0,
        max_mass=1.0,
        usd_path=f"/tmp/{uuid}.usd",
        urdf_path="",
        interaction_path="",
        caption_path=f"/tmp/{uuid}.json",
        tags=frozenset({"is_graspable"}),
    )


class _Registry:
    def __init__(self, metas: list[AssetMeta]) -> None:
        self.metas = {meta.uuid: meta for meta in metas}
        self.built_uuids: list[str] = []

    def __iter__(self):
        return iter(self.metas.values())

    def __len__(self) -> int:
        return len(self.metas)

    def get_meta(self, uuid: str) -> AssetMeta:
        return self.metas[uuid]

    def build_spec(self, meta, *, name=None, role, config=None):
        del config
        self.built_uuids.append(meta.uuid)
        return SimpleNamespace(
            name=name or meta.asset_id,
            role=role,
            uuid=meta.uuid,
        )


@pytest.mark.parametrize(
    ("template", "condition", "prompt"),
    [
        ("empty", "empty", ""),
        ("generic_object", "generic_object", "Pick an object"),
        (
            "place_a2b_generic_object",
            "generic_object",
            "Pick up an object and place it into a container",
        ),
        ("absent_object", "absent_object", "Pick up missing"),
    ],
)
def test_counterfactual_template_fixed_mode_returns_condition_and_prompt(
    template: str,
    condition: str,
    prompt: str,
) -> None:
    instruction = InstructionWrapper(template, template_mode="fixed")
    actors = (
        {
            "actor1": InstructionActor(
                uuid="missing",
                description="missing",
                raw_description="missing",
            )
        }
        if template == "absent_object"
        else None
    )

    assert (
        counterfactual_condition_from_template(template),
        instruction.render(actors=actors),
    ) == (condition, prompt)


@pytest.mark.parametrize(
    "template",
    ["generic_object", "absent_object", "place_a2b_generic_object"],
)
def test_counterfactual_template_variants_have_no_trailing_period(
    template: str,
) -> None:
    payload = get_instruction_template(template)

    assert all(not variant.endswith(".") for variant in payload["variants"])


@pytest.mark.parametrize(
    ("attribute_name", "expected"),
    [
        (None, ((), ("category",), ("category",))),
        ("color", (("category",), ("color",), ("category", "color"))),
        ("shape", (("category",), ("shape",), ("category", "shape"))),
        (
            "material",
            (("category",), ("material",), ("category", "material")),
        ),
    ],
)
def test_absent_object_rule_attribute_returns_expected_semantics(
    attribute_name,
    expected,
) -> None:
    rule = AbsentObjectRule.from_instruction(
        InstructionWrapper("absent_object", attribute_name=attribute_name)
    )

    assert (rule.match, rule.differ, rule.referent_fields) == expected


def test_absent_instruction_without_asset_resolver_raises_value_error():
    task = SimpleNamespace(instruction=InstructionWrapper("absent_object"))

    with pytest.raises(ValueError, match="requires an AssetResolver"):
        resolve_instruction_assets(task=task, resolver=None)


def test_absent_referent_category_present_elsewhere_selects_missing_category():
    apple = _meta("apple", category="apple")
    banana = _meta("banana", category="banana")
    duplicate_banana = _meta("banana-copy", category="banana")
    orange = _meta("orange", category="orange")
    registry = _Registry([apple, banana, duplicate_banana, orange])
    resolver = AssetResolver(
        cast(Any, registry),
        rng=np.random.default_rng(0),
    )

    resolver.resolve_absent_referent(
        present_uuids=[apple.uuid, banana.uuid],
        rule=AbsentObjectRule(
            differ=("category",),
            referent_fields=("category",),
        ),
    )

    assert registry.built_uuids == [orange.uuid]


@pytest.mark.parametrize("attribute_name", ["shape", "material"])
def test_absent_referent_multivalued_attribute_skips_unrenderable_candidate(
    attribute_name: Literal["shape", "material"],
) -> None:
    present = _meta("present")
    multi = (
        _meta("multi", shape=frozenset({"square", "triangle"}))
        if attribute_name == "shape"
        else _meta("multi", material=frozenset({"metal", "wood"}))
    )
    single = (
        _meta("single", shape=frozenset({"cylinder"}))
        if attribute_name == "shape"
        else _meta("single", material=frozenset({"metal"}))
    )
    registry = _Registry([present, multi, single])
    resolver = AssetResolver(
        cast(Any, registry),
        rng=np.random.default_rng(0),
    )

    resolver.resolve_absent_referent(
        present_uuids=[present.uuid],
        rule=AbsentObjectRule.from_instruction(
            InstructionWrapper(
                "absent_object",
                attribute_name=attribute_name,
            )
        ),
    )

    assert registry.built_uuids == ["single"]


@pytest.mark.parametrize("attribute_name", ["shape", "material"])
def test_absent_referent_only_multivalued_attribute_raises_resolution_error(
    attribute_name: Literal["shape", "material"],
) -> None:
    present = _meta("present")
    multi = (
        _meta("multi", shape=frozenset({"square", "triangle"}))
        if attribute_name == "shape"
        else _meta("multi", material=frozenset({"metal", "wood"}))
    )
    resolver = AssetResolver(
        cast(Any, _Registry([present, multi])),
        rng=np.random.default_rng(0),
    )

    with pytest.raises(AssetResolutionError, match="no renderable"):
        resolver.resolve_absent_referent(
            present_uuids=[present.uuid],
            rule=AbsentObjectRule.from_instruction(
                InstructionWrapper(
                    "absent_object",
                    attribute_name=attribute_name,
                )
            ),
        )
