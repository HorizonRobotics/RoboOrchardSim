## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import dataclasses
import json
from itertools import count
from pathlib import Path

import pytest
from google.protobuf.struct_pb2 import Struct
from mcap_protobuf.writer import Writer

from robo_orchard_sim.task_components.instructions import (
    extract_instruction_actor_uuids_from_mcap,
    registry as instruction_registry,
    render_instruction_from_mcap,
    render_instructions_from_mcaps,
)
from robo_orchard_sim.task_components.instructions.base import (
    InstructionActor,
    InstructionRenderError,
    InstructionWrapper,
    render_instruction_from_registry,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_meta_mcap(path: Path, payload: dict) -> None:
    struct_message = Struct()
    struct_message.update(payload)
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.write_message(
            topic="/meta_data",
            message=struct_message,
            log_time=1,
            publish_time=1,
        )
        writer.finish()


_TEMPLATE_COUNTER = count()


def _register_template(payload: dict) -> str:
    name = f"test_template_{next(_TEMPLATE_COUNTER)}"
    instruction_registry.INSTRUCTION_TEMPLATE_REGISTRY[name] = payload
    return name


@dataclasses.dataclass(slots=True)
class FakeAssetMeta:
    uuid: str
    category: str
    description: str
    caption_path: str


class FakeAssetRegistry:
    """Minimal AssetRegistry stand-in for InstructionActor.from_registry."""

    def __init__(self, metas: list[FakeAssetMeta]) -> None:
        self._metas = {meta.uuid: meta for meta in metas}

    def get_meta(self, uuid: str) -> FakeAssetMeta:
        return self._metas[uuid]


@dataclasses.dataclass(slots=True)
class FakeRigidObjectCfg:
    uuid: str | None
    category: str | None
    caption_path: str | None
    attributes: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass(slots=True)
class FakeRigidObject:
    cfg: FakeRigidObjectCfg


def _build_asset_meta(tmp_path, *, uuid: str, category: str) -> FakeAssetMeta:
    asset_dir = tmp_path / category
    asset_dir.mkdir(parents=True, exist_ok=True)
    return FakeAssetMeta(
        uuid=uuid,
        category=category,
        description=category,
        caption_path=str(asset_dir / "caption_candidates.json"),
    )


class TestInstructionActorFromRegistry:
    def test_from_registry_candidates_field_populates_seen_descriptions(
        self, tmp_path
    ):
        meta = _build_asset_meta(
            tmp_path,
            uuid="u-apple-candidates",
            category="apple",
        )
        _write_json(
            tmp_path / "apple" / "caption_candidates.json",
            {
                "uuid": "u-apple-candidates",
                "raw": "apple",
                "candidates": ["red apple", "green apple"],
                "unseen": ["fruit", "produce"],
            },
        )
        registry = FakeAssetRegistry([meta])

        actor = InstructionActor.from_registry(
            "u-apple-candidates",
            registry,
            actor_description_mode="seen",
            actor_description_seed=2,
        )

        assert actor.seen_descriptions == ["red apple", "green apple"]
        assert actor.description in {"red apple", "green apple"}

    def test_from_registry_raw_mode_returns_raw_description(self, tmp_path):
        meta = _build_asset_meta(
            tmp_path,
            uuid="u-apple-1",
            category="apple",
        )
        _write_json(
            tmp_path / "apple" / "caption_candidates.json",
            {
                "uuid": "u-apple-1",
                "raw": "fresh apple",
                "seen": ["red apple", "green apple"],
                "unseen": ["fruit", "produce"],
            },
        )
        registry = FakeAssetRegistry([meta])

        actor = InstructionActor.from_registry(
            "u-apple-1",
            registry,
            actor_description_mode="raw",
        )

        assert actor.uuid == "u-apple-1"
        assert actor.category == "apple"
        assert actor.description == "fresh apple"
        assert actor.raw_description == "fresh apple"
        assert actor.seen_descriptions == ["red apple", "green apple"]
        assert actor.unseen_descriptions == ["fruit", "produce"]

    def test_from_registry_seen_mode_uses_seeded_candidate(self, tmp_path):
        meta = _build_asset_meta(
            tmp_path,
            uuid="u-apple-2",
            category="apple",
        )
        _write_json(
            tmp_path / "apple" / "caption_candidates.json",
            {
                "uuid": "u-apple-2",
                "raw": "apple",
                "seen": ["red apple", "green apple"],
                "unseen": ["fruit", "produce"],
            },
        )
        registry = FakeAssetRegistry([meta])

        actor1 = InstructionActor.from_registry(
            "u-apple-2",
            registry,
            actor_description_mode="seen",
            actor_description_seed=3,
        )
        actor2 = InstructionActor.from_registry(
            "u-apple-2",
            registry,
            actor_description_mode="seen",
            actor_description_seed=3,
        )

        assert actor1.description == actor2.description
        assert actor1.description in {"red apple", "green apple"}

    def test_from_registry_seen_mode_falls_back_to_candidates(self, tmp_path):
        meta = _build_asset_meta(
            tmp_path,
            uuid="u-apple-candidates-mode",
            category="apple",
        )
        _write_json(
            tmp_path / "apple" / "caption_candidates.json",
            {
                "uuid": "u-apple-candidates-mode",
                "raw": "apple",
                "candidates": ["red apple", "green apple"],
            },
        )
        registry = FakeAssetRegistry([meta])

        actor1 = InstructionActor.from_registry(
            "u-apple-candidates-mode",
            registry,
            actor_description_mode="seen",
            actor_description_seed=3,
        )
        actor2 = InstructionActor.from_registry(
            "u-apple-candidates-mode",
            registry,
            actor_description_mode="seen",
            actor_description_seed=3,
        )

        assert actor1.description == actor2.description
        assert actor1.description in {"red apple", "green apple"}

    def test_from_registry_unseen_mode_uses_seeded_candidate(self, tmp_path):
        meta = _build_asset_meta(
            tmp_path,
            uuid="u-apple-3",
            category="apple",
        )
        _write_json(
            tmp_path / "apple" / "caption_candidates.json",
            {
                "uuid": "u-apple-3",
                "raw": "apple",
                "seen": ["red apple", "green apple"],
                "unseen": ["fruit", "produce"],
            },
        )
        registry = FakeAssetRegistry([meta])

        actor1 = InstructionActor.from_registry(
            "u-apple-3",
            registry,
            actor_description_mode="unseen",
            actor_description_seed=5,
        )
        actor2 = InstructionActor.from_registry(
            "u-apple-3",
            registry,
            actor_description_mode="unseen",
            actor_description_seed=5,
        )

        assert actor1.description == actor2.description
        assert actor1.description in {"fruit", "produce"}

    def test_from_registry_missing_optional_fields_uses_fallbacks(
        self, tmp_path
    ):
        meta = _build_asset_meta(
            tmp_path,
            uuid="u-4",
            category="apple",
        )
        _write_json(
            tmp_path / "apple" / "caption_candidates.json",
            {"uuid": "u-4"},
        )
        registry = FakeAssetRegistry([meta])

        actor = InstructionActor.from_registry("u-4", registry)

        assert actor.uuid == "u-4"
        assert actor.category == "apple"
        assert actor.raw_description == "apple"
        assert actor.description == "apple"
        assert actor.seen_descriptions == []
        assert actor.unseen_descriptions == []

    def test_from_registry_uuid_mismatch_raises(self, tmp_path):
        meta = _build_asset_meta(
            tmp_path,
            uuid="u-5",
            category="apple",
        )
        _write_json(
            tmp_path / "apple" / "caption_candidates.json",
            {
                "uuid": "u-other",
                "raw": "alice",
                "seen": [],
                "unseen": [],
            },
        )
        registry = FakeAssetRegistry([meta])

        with pytest.raises(InstructionRenderError, match="uuid mismatch"):
            InstructionActor.from_registry("u-5", registry)

    def test_from_registry_invalid_caption_payload_raises(self, tmp_path):
        meta = _build_asset_meta(
            tmp_path,
            uuid="u-6",
            category="apple",
        )

        registry = FakeAssetRegistry([meta])

        with pytest.raises(
            InstructionRenderError,
            match="Invalid caption payload",
        ):
            InstructionActor.from_registry("u-6", registry)


class TestInstructionActorFromRigidObject:
    def test_from_rigid_object_raw_mode_returns_raw_description(
        self, tmp_path
    ):
        caption_path = tmp_path / "caption.json"
        _write_json(
            caption_path,
            {
                "uuid": "u-rigid-1",
                "raw": "fresh apple",
                "seen": ["red apple"],
                "unseen": ["fruit"],
            },
        )
        rigid_object = FakeRigidObject(
            cfg=FakeRigidObjectCfg(
                uuid="u-rigid-1",
                category="apple",
                caption_path=str(caption_path),
            )
        )

        actor = InstructionActor.from_rigid_object(rigid_object)

        assert actor.uuid == "u-rigid-1"
        assert actor.category == "apple"
        assert actor.description == "fresh apple"
        assert actor.seen_descriptions == ["red apple"]
        assert actor.unseen_descriptions == ["fruit"]

    def test_from_rigid_object_unset_caption_path_raises(self):
        rigid_object = FakeRigidObject(
            cfg=FakeRigidObjectCfg(
                uuid="u-rigid-2",
                category="apple",
                caption_path=None,
            )
        )

        with pytest.raises(
            InstructionRenderError,
            match="caption_path is required",
        ):
            InstructionActor.from_rigid_object(rigid_object)

    def test_from_rigid_object_unset_uuid_falls_back_to_unknown(
        self, tmp_path
    ):
        caption_path = tmp_path / "caption.json"
        _write_json(
            caption_path,
            {
                "raw": "apple",
                "seen": [],
                "unseen": [],
            },
        )
        rigid_object = FakeRigidObject(
            cfg=FakeRigidObjectCfg(
                uuid=None,
                category=None,
                caption_path=str(caption_path),
            )
        )

        actor = InstructionActor.from_rigid_object(rigid_object)

        assert actor.uuid == "unknown"
        assert actor.category == "unknown"
        assert actor.raw_description == "apple"

    def test_from_rigid_object_with_attribute_color_sets_attribute_value(
        self,
        tmp_path,
    ):
        caption_path = tmp_path / "caption_candidates.json"
        _write_json(
            caption_path,
            {
                "uuid": "u-peach-color",
                "raw": "peach",
                "seen": ["yellow peach"],
            },
        )
        rigid_object = FakeRigidObject(
            cfg=FakeRigidObjectCfg(
                uuid="u-peach-color",
                category="peach",
                caption_path=str(caption_path),
                attributes={"color": ("yellow",)},
            )
        )

        actor = InstructionActor.from_rigid_object_with_attribute(
            rigid_object,
            attribute_name="color",
        )

        assert actor.attribute_name == "color"
        assert actor.attribute_value == "yellow"

    def test_from_rigid_object_with_attribute_two_colors_joins_values(
        self,
        tmp_path,
    ):
        caption_path = tmp_path / "caption_candidates.json"
        _write_json(
            caption_path,
            {
                "uuid": "u-cup-color",
                "raw": "cup",
            },
        )
        rigid_object = FakeRigidObject(
            cfg=FakeRigidObjectCfg(
                uuid="u-cup-color",
                category="cup",
                caption_path=str(caption_path),
                attributes={"color": ("black", "white")},
            )
        )

        actor = InstructionActor.from_rigid_object_with_attribute(
            rigid_object,
            attribute_name="color",
        )

        assert actor.attribute_value == "black and white"

    def test_from_rigid_object_with_attribute_three_colors_formats_phrase(
        self,
        tmp_path,
    ):
        caption_path = tmp_path / "caption_candidates.json"
        _write_json(
            caption_path,
            {
                "uuid": "u-peach-colorful",
                "raw": "peach",
            },
        )
        rigid_object = FakeRigidObject(
            cfg=FakeRigidObjectCfg(
                uuid="u-peach-colorful",
                category="peach",
                caption_path=str(caption_path),
                attributes={"color": ("yellow", "green", "red")},
            )
        )

        actor = InstructionActor.from_rigid_object_with_attribute(
            rigid_object,
            attribute_name="color",
        )

        assert actor.attribute_value == "green, red, and yellow"

    def test_from_rigid_object_with_attribute_missing_value_raises(
        self,
        tmp_path,
    ):
        caption_path = tmp_path / "caption_candidates.json"
        _write_json(caption_path, {"uuid": "u-plate-shape", "raw": "plate"})
        rigid_object = FakeRigidObject(
            cfg=FakeRigidObjectCfg(
                uuid="u-plate-shape",
                category="plate",
                caption_path=str(caption_path),
            )
        )

        with pytest.raises(InstructionRenderError, match="no 'shape' value"):
            InstructionActor.from_rigid_object_with_attribute(
                rigid_object,
                attribute_name="shape",
            )

    def test_from_rigid_object_with_attribute_multi_shape_raises(
        self,
        tmp_path,
    ):
        caption_path = tmp_path / "caption_candidates.json"
        _write_json(caption_path, {"uuid": "u-bowl-shape", "raw": "bowl"})
        rigid_object = FakeRigidObject(
            cfg=FakeRigidObjectCfg(
                uuid="u-bowl-shape",
                category="bowl",
                caption_path=str(caption_path),
                attributes={"shape": ("round", "deep")},
            )
        )

        with pytest.raises(
            InstructionRenderError,
            match="multiple 'shape' values",
        ):
            InstructionActor.from_rigid_object_with_attribute(
                rigid_object,
                attribute_name="shape",
            )


class TestInstructionWrapper:
    def test_placeholders_from_fixed_mode_returns_declared_fields(self):
        template = _register_template(
            {
                "fixed": "Grab {item} for {name} ({actor.uuid})",
                "variants": ["ignored-variant-{name}"],
            }
        )
        wrapper = InstructionWrapper(template, template_mode="variants")
        assert wrapper.template == template
        assert not hasattr(wrapper, "placeholders")
        assert not hasattr(wrapper, "validate")
        assert not hasattr(wrapper, "render_partial")

    def test_init_with_unknown_template_raises_value_error(self):
        with pytest.raises(
            InstructionRenderError,
            match="Unknown instruction template",
        ):
            InstructionWrapper("missing_template")

    def test_render_with_mapping_actors_renders_text(self):
        template = _register_template(
            {
                "fixed": "Grab {item} using {arm} for {name}.",
                "variants": [],
            }
        )
        wrapper = InstructionWrapper(template)
        text = wrapper.render(
            actors={"name": "alice", "arm": "left", "item": "apple"},
        )
        assert text == "Grab apple using left for alice."

    def test_render_spatial_pick_default_relation_context_returns_instruction(
        self,
    ):
        wrapper = InstructionWrapper(
            "spatial_pick_default",
            template_mode="fixed",
        )

        text = wrapper.render(
            actors={
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
        )

        assert text == "Pick up the tomato to the left of the apple."

    def test_render_without_explicit_template_mode_prefers_variants(
        self,
    ):
        template = _register_template(
            {
                "fixed": "fixed-{name}",
                "variants": ["variant-{name}"],
            }
        )
        wrapper = InstructionWrapper(template)

        text = wrapper.render(
            actors={"name": "alice"},
            template_seed=1,
        )

        assert text == "variant-alice"

    def test_render_with_dataclass_actor1_renders_nested_fields(self):
        template = _register_template(
            {
                "fixed": (
                    "Seen {actor.seen_descriptions} by {category} ({uuid})."
                ),
                "variants": [],
            }
        )
        wrapper = InstructionWrapper(template)
        text = wrapper.render(
            actors={
                "actor1": InstructionActor(
                    uuid="actor-1",
                    description="apple",
                    raw_description="apple",
                    seen_descriptions=["apple"],
                    unseen_descriptions=[],
                )
            }
        )
        assert text == "Seen ['apple'] by unknown (actor-1)."

    def test_render_with_actor1_populates_aliases(self):
        template = _register_template(
            {
                "fixed": ("{actor.uuid}|{actor1.uuid}|{category}|{uuid}"),
                "variants": [],
            }
        )
        wrapper = InstructionWrapper(template)
        actor = InstructionActor(
            category="alice",
            uuid="a-1",
            description="alice",
            raw_description="alice",
            seen_descriptions=[],
            unseen_descriptions=[],
        )

        text = wrapper.render(actors={"actor1": actor})

        assert text == "a-1|a-1|alice|a-1"

    def test_render_strict_missing_placeholder_raises_error(self):
        template = _register_template(
            {"fixed": "Move {item} with {arm}.", "variants": []}
        )
        wrapper = InstructionWrapper(template)
        with pytest.raises(
            InstructionRenderError,
            match="Unresolved placeholders: arm",
        ):
            wrapper.render(actors={"item": "cup"})

    def test_render_with_multiple_actors_mapping_uses_named_actors(self):
        template = _register_template(
            {
                "fixed": (
                    "ActorA {actor1.category}:"
                    "{actor1.seen_descriptions[0]} -> "
                    "ActorB {actor2.category}:"
                    "{actor2.unseen_descriptions[0]}"
                ),
                "variants": [],
            }
        )
        wrapper = InstructionWrapper(template)
        text = wrapper.render(
            actors={
                "actor1": InstructionActor(
                    category="alice",
                    uuid="a-1",
                    description="alice",
                    raw_description="alice",
                    seen_descriptions=["apple"],
                    unseen_descriptions=["orange"],
                ),
                "actor2": InstructionActor(
                    category="bob",
                    uuid="b-2",
                    description="bob",
                    raw_description="bob",
                    seen_descriptions=["cup"],
                    unseen_descriptions=["plate"],
                ),
            }
        )
        assert text == "ActorA alice:apple -> ActorB bob:plate"

    def test_render_with_template_mode_seed_is_deterministic(self):
        template = _register_template(
            {
                "fixed": "fixed-{name}",
                "variants": ["variant-a-{name}", "variant-b-{name}"],
            }
        )
        wrapper1 = InstructionWrapper(template, template_mode="variants")
        wrapper2 = InstructionWrapper(template, template_mode="variants")

        text1 = wrapper1.render(
            actors={"name": "alice"},
            template_seed=9,
        )
        text2 = wrapper2.render(
            actors={"name": "alice"},
            template_seed=9,
        )

        assert text1 == text2
        assert text1 in {"variant-a-alice", "variant-b-alice"}

    def test_render_with_actor_description_mode_raw_uses_category(self):
        template = _register_template(
            {
                "fixed": "name={actor.category},desc={actor.description}",
                "variants": [],
            }
        )
        wrapper = InstructionWrapper(template)
        actor = InstructionActor(
            category="raw-name",
            uuid="u-raw",
            description="raw-name",
            raw_description="raw-desc",
            seen_descriptions=["seen-desc"],
            unseen_descriptions=["unseen-desc"],
        )
        text = wrapper.render(actors={"actor1": actor})
        assert text == "name=raw-name,desc=raw-desc"

    def test_render_with_actor_description_mode_seen_requires_seed(self):
        template = _register_template(
            {"fixed": "desc={actor.description}", "variants": []}
        )
        wrapper = InstructionWrapper(template, actor_description_mode="seen")
        actor = InstructionActor(
            category="raw-name",
            uuid="u-seed",
            description="raw-name",
            raw_description="raw-name",
            seen_descriptions=["seen-a"],
            unseen_descriptions=["unseen-a"],
        )

        with pytest.raises(
            InstructionRenderError,
            match="actor_description_seed is required",
        ):
            wrapper.render(actors={"actor1": actor})

    def test_render_with_actor_description_mode_empty_candidates_raises(self):
        template = _register_template(
            {"fixed": "desc={actor.description}", "variants": []}
        )
        wrapper = InstructionWrapper(template, actor_description_mode="seen")
        actor = InstructionActor(
            category="raw-name",
            uuid="u-empty",
            description="raw-name",
            raw_description="raw-name",
            seen_descriptions=[],
            unseen_descriptions=[],
        )

        with pytest.raises(
            InstructionRenderError,
            match="description 'seen' is empty",
        ):
            wrapper.render(
                actors={"actor1": actor},
                actor_description_seed=1,
            )

    def test_render_uses_wrapper_default_actor_description_mode(self):
        template = _register_template(
            {"fixed": "desc={actor.description}", "variants": []}
        )
        wrapper = InstructionWrapper(template, actor_description_mode="seen")
        actor = InstructionActor(
            uuid="u-default-seen",
            description="raw-name",
            raw_description="raw-name",
            seen_descriptions=["seen-a"],
            unseen_descriptions=["unseen-a"],
        )

        text = wrapper.render(
            actors={"actor1": actor},
            actor_description_seed=1,
        )

        assert text == "desc=seen-a"

    def test_render_resolves_actors_with_actor_description_mode(self):
        template = _register_template(
            {
                "fixed": "a1={actor1.description},a2={actor2.description}",
                "variants": [],
            }
        )
        wrapper = InstructionWrapper(template, actor_description_mode="seen")

        text = wrapper.render(
            actors={
                "actor1": InstructionActor(
                    uuid="u-extra-1",
                    description="raw-a",
                    raw_description="raw-a",
                    seen_descriptions=["seen-a"],
                    unseen_descriptions=["unseen-a"],
                ),
                "actor2": InstructionActor(
                    uuid="u-extra-2",
                    description="raw-b",
                    raw_description="raw-b",
                    seen_descriptions=["seen-b"],
                    unseen_descriptions=["unseen-b"],
                ),
            },
            actor_description_seed=1,
        )

        assert text == "a1=seen-a,a2=seen-b"


def test_render_with_actor_store_and_multiple_actors(tmp_path):
    caption_path_1 = tmp_path / "obj-1.json"
    caption_path_2 = tmp_path / "obj-2.json"
    _write_json(
        caption_path_1,
        {
            "category": "plate",
            "uuid": "obj-1",
            "raw": "plate",
            "seen": ["small plate", "white plate"],
            "unseen": ["ceramic dish"],
        },
    )
    _write_json(
        caption_path_2,
        {
            "category": "cup",
            "uuid": "obj-2",
            "raw": "cup",
            "seen": ["red cup", "blue cup"],
            "unseen": ["glass cup"],
        },
    )

    registry = FakeAssetRegistry(
        [
            FakeAssetMeta(
                uuid="obj-1",
                category="plate",
                description="plate",
                caption_path=str(caption_path_1),
            ),
            FakeAssetMeta(
                uuid="obj-2",
                category="cup",
                description="cup",
                caption_path=str(caption_path_2),
            ),
        ]
    )

    template = _register_template(
        {
            "fixed": (
                "Pick {actor1.description} ({actor1.uuid}) "
                "then pass to {actor2.description} ({actor2.uuid})"
            ),
            "variants": [],
        }
    )

    actor1 = InstructionActor.from_registry(
        "obj-1",
        registry,
        actor_description_mode="seen",
        actor_description_seed=0,
    )
    actor2 = InstructionActor.from_registry(
        "obj-2",
        registry,
        actor_description_mode="seen",
        actor_description_seed=0,
    )
    wrapper = InstructionWrapper(template, actor_description_mode="seen")

    text1 = wrapper.render(
        actors={"actor1": actor1, "actor2": actor2},
        actor_description_seed=0,
    )
    text2 = wrapper.render(
        actors={"actor1": actor1, "actor2": actor2},
        actor_description_seed=0,
    )

    assert text1 == text2
    assert text1 in {
        "Pick white plate (obj-1) then pass to blue cup (obj-2)",
        "Pick small plate (obj-1) then pass to red cup (obj-2)",
    }


def test_render_instruction_from_registry_known_uuid_renders_template(
    tmp_path,
):
    meta = _build_asset_meta(
        tmp_path,
        uuid="u-apple-render",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "u-apple-render",
            "raw": "apple",
            "candidates": ["red apple", "green apple"],
            "unseen": ["fruit"],
        },
    )
    registry = FakeAssetRegistry([meta])
    template = _register_template(
        {
            "variants": [
                "Pick the {actor.description}",
                "Grab the {actor.description}",
            ]
        }
    )

    rendered1 = render_instruction_from_registry(
        template_name=template,
        actor_uuids={"actor1": "u-apple-render"},
        registry=registry,
        actor_description_mode="seen",
        template_seed=4,
        actor_description_seed=7,
    )
    rendered2 = render_instruction_from_registry(
        template_name=template,
        actor_uuids={"actor1": "u-apple-render"},
        registry=registry,
        actor_description_mode="seen",
        template_seed=4,
        actor_description_seed=7,
    )

    assert rendered1 == rendered2
    assert rendered1 in {
        "Pick the red apple",
        "Pick the green apple",
        "Grab the red apple",
        "Grab the green apple",
    }


def test_render_instruction_from_registry_defaults_to_raw_description(
    tmp_path,
):
    meta = _build_asset_meta(
        tmp_path,
        uuid="u-apple-default-mode",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "u-apple-default-mode",
            "raw": "fresh apple",
            "seen": ["red apple"],
        },
    )
    registry = FakeAssetRegistry([meta])
    template = _register_template({"fixed": "Use {actor1.description}"})

    rendered = render_instruction_from_registry(
        template_name=template,
        actor_uuids={"actor1": "u-apple-default-mode"},
        registry=registry,
    )

    assert rendered == "Use fresh apple"


def test_render_instruction_from_registry_fixed_mode_uses_fixed_template(
    tmp_path,
):
    meta = _build_asset_meta(
        tmp_path,
        uuid="u-apple-fixed",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "u-apple-fixed",
            "raw": "apple",
            "candidates": ["red apple"],
            "unseen": ["fruit"],
        },
    )
    registry = FakeAssetRegistry([meta])
    template = _register_template(
        {
            "fixed": "Fixed {actor.description}",
            "variants": ["Variant {actor.description}"],
        }
    )

    rendered = render_instruction_from_registry(
        template_name=template,
        actor_uuids={"actor1": "u-apple-fixed"},
        registry=registry,
        template_mode="fixed",
        actor_description_mode="seen",
        actor_description_seed=3,
    )

    assert rendered == "Fixed red apple"


def test_render_instruction_from_registry_actor_description_mode_raw_uses_raw_description(  # noqa: E501
    tmp_path,
):
    meta = _build_asset_meta(
        tmp_path,
        uuid="u-apple-raw-mode",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "u-apple-raw-mode",
            "raw": "fresh apple",
            "candidates": ["red apple"],
            "unseen": ["fruit"],
        },
    )
    registry = FakeAssetRegistry([meta])
    template = _register_template({"fixed": "Use {actor.description}"})

    rendered = render_instruction_from_registry(
        template_name=template,
        actor_uuids={"actor1": "u-apple-raw-mode"},
        registry=registry,
        actor_description_mode="raw",
    )

    assert rendered == "Use fresh apple"


def test_render_instruction_from_registry_actor_description_mode_seen_falls_back_to_candidates(  # noqa: E501
    tmp_path,
):
    meta = _build_asset_meta(
        tmp_path,
        uuid="u-apple-candidates-render",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "u-apple-candidates-render",
            "raw": "apple",
            "candidates": ["red apple"],
        },
    )
    registry = FakeAssetRegistry([meta])
    template = _register_template({"fixed": "Use {actor.description}"})

    rendered = render_instruction_from_registry(
        template_name=template,
        actor_uuids={"actor1": "u-apple-candidates-render"},
        registry=registry,
        actor_description_mode="seen",
        actor_description_seed=0,
    )

    assert rendered == "Use red apple"


def test_render_instruction_from_registry_actor_description_mode_candidates_raises_value_error(  # noqa: E501
    tmp_path,
):
    meta = _build_asset_meta(
        tmp_path,
        uuid="u-apple-candidates-invalid",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "u-apple-candidates-invalid",
            "raw": "apple",
            "candidates": ["red apple"],
        },
    )
    registry = FakeAssetRegistry([meta])
    template = _register_template({"fixed": "Use {actor.description}"})

    with pytest.raises(
        InstructionRenderError,
        match="Unsupported actor_description_mode",
    ):
        render_instruction_from_registry(
            template_name=template,
            actor_uuids={"actor1": "u-apple-candidates-invalid"},
            registry=registry,
            actor_description_mode="candidates",
            actor_description_seed=0,
        )


def test_extract_instruction_actor_uuids_from_mcap_pick_and_place_returns_actor_mapping(  # noqa: E501
    tmp_path,
):
    mcap_path = tmp_path / "episode.mcap"
    _write_meta_mcap(
        mcap_path,
        {
            "actors": {
                "apple_001": {
                    "actor_type": "pick",
                    "actor_uuid": "uuid-apple",
                },
                "plate_001": {
                    "actor_type": "place",
                    "actor_uuid": "uuid-plate",
                },
            }
        },
    )

    template = _register_template(
        {"fixed": "Move {actor1.description} to {actor2.description}"}
    )

    actor_uuids = extract_instruction_actor_uuids_from_mcap(
        str(mcap_path),
        template_name=template,
    )

    assert actor_uuids == {
        "actor1": "uuid-apple",
        "actor2": "uuid-plate",
    }


def test_extract_instruction_actor_uuids_from_mcap_pick_only_template_does_not_require_place(  # noqa: E501
    tmp_path,
):
    mcap_path = tmp_path / "episode_pick_only.mcap"
    _write_meta_mcap(
        mcap_path,
        {
            "actors": {
                "apple_001": {
                    "actor_type": "pick",
                    "actor_uuid": "uuid-apple",
                },
            }
        },
    )

    actor_uuids = extract_instruction_actor_uuids_from_mcap(
        str(mcap_path),
        template_name="pick_default",
    )

    assert actor_uuids == {"actor1": "uuid-apple"}


def test_extract_instruction_actor_uuids_from_mcap_variants_only_template_returns_actor_mapping(  # noqa: E501
    tmp_path,
):
    mcap_path = tmp_path / "episode_variants_only.mcap"
    _write_meta_mcap(
        mcap_path,
        {
            "actors": {
                "apple_001": {
                    "actor_type": "pick",
                    "actor_uuid": "uuid-apple",
                },
            }
        },
    )
    template = _register_template({"variants": ["Grab {actor1.description}."]})

    actor_uuids = extract_instruction_actor_uuids_from_mcap(
        str(mcap_path),
        template_name=template,
    )

    assert actor_uuids == {"actor1": "uuid-apple"}


def test_render_instructions_from_mcaps_batch_returns_one_instruction_per_mcap(
    tmp_path,
):
    apple_meta = _build_asset_meta(
        tmp_path,
        uuid="uuid-apple",
        category="apple",
    )
    plate_meta = _build_asset_meta(
        tmp_path,
        uuid="uuid-plate",
        category="plate",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "uuid-apple",
            "raw": "apple",
            "candidates": ["red apple"],
            "unseen": ["fruit"],
        },
    )
    _write_json(
        tmp_path / "plate" / "caption_candidates.json",
        {
            "uuid": "uuid-plate",
            "raw": "plate",
            "candidates": ["white plate"],
            "unseen": ["dish"],
        },
    )
    registry = FakeAssetRegistry([apple_meta, plate_meta])
    template = _register_template(
        {
            "fixed": (
                "Pick the {actor1.description} and place it on "
                "the {actor2.description}"
            )
        }
    )
    mcap_path_1 = tmp_path / "episode_1.mcap"
    mcap_path_2 = tmp_path / "episode_2.mcap"
    payload = {
        "actors": {
            "apple_001": {
                "actor_type": "pick",
                "actor_uuid": "uuid-apple",
            },
            "plate_001": {
                "actor_type": "place",
                "actor_uuid": "uuid-plate",
            },
        }
    }
    _write_meta_mcap(mcap_path_1, payload)
    _write_meta_mcap(mcap_path_2, payload)

    single_rendered = render_instruction_from_mcap(
        mcap_path=str(mcap_path_1),
        template_name=template,
        registry=registry,
        actor_description_mode="seen",
        template_seed=1,
        actor_description_seed=2,
    )
    rendered = render_instructions_from_mcaps(
        mcap_paths=[str(mcap_path_1), str(mcap_path_2)],
        template_name=template,
        registry=registry,
        actor_description_mode="seen",
        template_seed=1,
        actor_description_seed=2,
    )

    assert single_rendered == (
        "Pick the red apple and place it on the white plate"
    )
    assert rendered == [
        {
            "instruction": single_rendered,
            "mcap_path": str(mcap_path_1),
        },
        {
            "instruction": single_rendered,
            "mcap_path": str(mcap_path_2),
        },
    ]


def test_render_instructions_from_mcaps_actor_description_mode_raw_uses_raw_description(  # noqa: E501
    tmp_path,
):
    apple_meta = _build_asset_meta(
        tmp_path,
        uuid="uuid-apple-raw",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "uuid-apple-raw",
            "raw": "fresh apple",
            "candidates": ["red apple"],
            "unseen": ["fruit"],
        },
    )
    registry = FakeAssetRegistry([apple_meta])
    mcap_path = tmp_path / "episode_raw.mcap"
    _write_meta_mcap(
        mcap_path,
        {
            "actors": {
                "apple_001": {
                    "actor_type": "pick",
                    "actor_uuid": "uuid-apple-raw",
                },
            }
        },
    )
    template = _register_template({"fixed": "Use {actor1.description}"})

    rendered = render_instructions_from_mcaps(
        mcap_paths=[str(mcap_path)],
        template_name=template,
        registry=registry,
        actor_description_mode="raw",
    )

    assert rendered == [
        {
            "instruction": "Use fresh apple",
            "mcap_path": str(mcap_path),
        }
    ]


def test_render_instructions_from_mcaps_actor_description_mode_seen_falls_back_to_candidates(  # noqa: E501
    tmp_path,
):
    apple_meta = _build_asset_meta(
        tmp_path,
        uuid="uuid-apple-candidates-mode",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "uuid-apple-candidates-mode",
            "raw": "apple",
            "candidates": ["red apple"],
            "unseen": ["fruit"],
        },
    )
    registry = FakeAssetRegistry([apple_meta])
    mcap_path = tmp_path / "episode_candidates.mcap"
    _write_meta_mcap(
        mcap_path,
        {
            "actors": {
                "apple_001": {
                    "actor_type": "pick",
                    "actor_uuid": "uuid-apple-candidates-mode",
                },
            }
        },
    )
    template = _register_template({"fixed": "Use {actor1.description}"})

    rendered = render_instructions_from_mcaps(
        mcap_paths=[str(mcap_path)],
        template_name=template,
        registry=registry,
        actor_description_mode="seen",
        actor_description_seed=0,
    )

    assert rendered == [
        {
            "instruction": "Use red apple",
            "mcap_path": str(mcap_path),
        }
    ]


def test_render_instructions_from_mcaps_explicit_fixed_template_mode_uses_fixed(  # noqa: E501
    tmp_path,
):
    apple_meta = _build_asset_meta(
        tmp_path,
        uuid="uuid-apple-fixed-mode",
        category="apple",
    )
    _write_json(
        tmp_path / "apple" / "caption_candidates.json",
        {
            "uuid": "uuid-apple-fixed-mode",
            "raw": "apple",
            "candidates": ["red apple"],
        },
    )
    registry = FakeAssetRegistry([apple_meta])
    mcap_path = tmp_path / "episode_fixed_mode.mcap"
    _write_meta_mcap(
        mcap_path,
        {
            "actors": {
                "apple_001": {
                    "actor_type": "pick",
                    "actor_uuid": "uuid-apple-fixed-mode",
                },
            }
        },
    )
    template = _register_template(
        {
            "fixed": "Fixed {actor1.description}",
            "variants": ["Variant {actor1.description}"],
        }
    )

    rendered = render_instructions_from_mcaps(
        mcap_paths=[str(mcap_path)],
        template_name=template,
        registry=registry,
        template_mode="fixed",
        actor_description_mode="seen",
        actor_description_seed=0,
    )

    assert rendered == [
        {
            "instruction": "Fixed red apple",
            "mcap_path": str(mcap_path),
        }
    ]
