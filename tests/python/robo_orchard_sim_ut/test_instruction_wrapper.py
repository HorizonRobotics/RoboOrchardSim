## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import dataclasses
import json
from itertools import count

import pytest

from robo_orchard_sim.tasks.instructions import (
    registry as instruction_registry,
)
from robo_orchard_sim.tasks.instructions.base import (
    InstructionActor,
    InstructionRenderError,
    InstructionWrapper,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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

    def test_from_registry_missing_required_fields_raises(self, tmp_path):
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

        with pytest.raises(
            InstructionRenderError,
            match="missing required fields",
        ):
            InstructionActor.from_registry("u-4", registry)

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
