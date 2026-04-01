## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import json

import pytest

from robo_orchard_sim.tasks.instructions.base import (
    Actor,
    ActorDescription,
    ActorStoreAdapter,
    InstructionRenderError,
    InstructionWrapper,
)


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_template_json(tmp_path, payload: dict) -> str:
    path = tmp_path / "template.json"
    _write_json(path, payload)
    return str(path)


class TestInstructionWrapper:
    def test_placeholders_from_raw_mode(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {
                "raw": "Grab {item} for {name} ({actor.uuid})",
                "seen": ["ignored-seen-{name}"],
                "unseen": ["ignored-unseen-{name}"],
            },
        )
        wrapper = InstructionWrapper(template_json)
        assert wrapper.placeholders() == {"item", "actor.uuid", "name"}

    def test_placeholders_from_seen_mode_use_selected_template(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {
                "raw": "raw-{name}",
                "seen": ["seen-a-{name}", "seen-b-{item}"],
                "unseen": ["unseen-{name}"],
            },
        )
        wrapper = InstructionWrapper(template_json, template_mode="seen")
        placeholders = wrapper.placeholders(seed=0)
        assert placeholders in ({"name"}, {"item"})

    def test_load_template_from_invalid_json_source_raises(self, tmp_path):
        broken = tmp_path / "broken_template.json"
        broken.write_text("{invalid-json", encoding="utf-8")
        with pytest.raises(InstructionRenderError, match="Invalid template"):
            InstructionWrapper(str(broken))

    def test_render_with_mapping_actor_and_extra(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {
                "raw": "Grab {item} using {arm} for {name}.",
                "seen": [],
                "unseen": [],
            },
        )
        wrapper = InstructionWrapper(template_json)
        text = wrapper.render(
            actors={"name": "alice", "arm": "left"},
            extra={"item": "apple"},
        )
        assert text == "Grab apple using left for alice."

    def test_render_with_dataclass_actor(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {
                "raw": (
                    "Seen {actor.description_pool.seen} "
                    "by {category} ({uuid})."
                ),
                "seen": [],
                "unseen": [],
            },
        )
        wrapper = InstructionWrapper(template_json)
        text = wrapper.render(
            actors=Actor(
                category="bob",
                uuid="actor-1",
                description_pool=ActorDescription(seen=["apple"], unseen=[]),
            )
        )
        assert text == "Seen ['apple'] by bob (actor-1)."

    def test_build_context_with_sequence_actors(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {"raw": "{actor1.uuid}-{actor2.uuid}", "seen": [], "unseen": []},
        )
        wrapper = InstructionWrapper(template_json)
        actor1 = Actor(
            category="alice",
            uuid="a-1",
            description_pool=ActorDescription(seen=[], unseen=[]),
        )
        actor2 = Actor(
            category="bob",
            uuid="b-2",
            description_pool=ActorDescription(seen=[], unseen=[]),
        )

        context = wrapper.build_context(actors=[actor1, actor2])

        assert context["actor"] == actor1
        assert context["actor1"] == actor1
        assert context["actor2"] == actor2
        assert context["actors"]["actor1"] == actor1
        assert context["actors"]["actor2"] == actor2
        assert context["uuid"] == "a-1"

    def test_validate_raise_when_missing_placeholder(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {"raw": "Move {item} with {arm}.", "seen": [], "unseen": []},
        )
        wrapper = InstructionWrapper(template_json)
        with pytest.raises(
            InstructionRenderError,
            match="Unresolved placeholders: item",
        ):
            wrapper.validate(actors={"arm": "left"})

    def test_render_partial_keeps_unresolved_fields(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {
                "raw": "Move {item!r:^10} with {arm}.",
                "seen": [],
                "unseen": [],
            },
        )
        wrapper = InstructionWrapper(template_json)
        text = wrapper.render_partial(actors={"arm": "left"})
        assert text == "Move {item!r:^10} with left."

    def test_render_with_multiple_actors_mapping(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {
                "raw": (
                    "ActorA {actor1.category}:"
                    "{actor1.description_pool.seen[0]} -> "
                    "ActorB {actor2.category}:"
                    "{actor2.description_pool.unseen[0]}"
                ),
                "seen": [],
                "unseen": [],
            },
        )
        wrapper = InstructionWrapper(template_json)
        text = wrapper.render(
            actors={
                "actor1": Actor(
                    category="alice",
                    uuid="a-1",
                    description_pool=ActorDescription(
                        seen=["apple"],
                        unseen=["orange"],
                    ),
                ),
                "actor2": Actor(
                    category="bob",
                    uuid="b-2",
                    description_pool=ActorDescription(
                        seen=["cup"],
                        unseen=["plate"],
                    ),
                ),
            }
        )
        assert text == "ActorA alice:apple -> ActorB bob:plate"

    def test_render_with_template_mode_seed_is_deterministic(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {
                "raw": "raw-{name}",
                "seen": ["seen-a-{name}", "seen-b-{name}"],
                "unseen": ["unseen-a-{name}", "unseen-b-{name}"],
            },
        )
        wrapper1 = InstructionWrapper(template_json, template_mode="unseen")
        wrapper2 = InstructionWrapper(template_json, template_mode="unseen")

        text1 = wrapper1.render(actors={"name": "alice"}, seed=9)
        text2 = wrapper2.render(actors={"name": "alice"}, seed=9)

        assert text1 == text2
        assert text1 in {"unseen-a-alice", "unseen-b-alice"}

    def test_render_with_obj_desc_mode_raw(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {
                "raw": "name={actor.category},desc={actor.description}",
                "seen": [],
                "unseen": [],
            },
        )
        wrapper = InstructionWrapper(template_json)
        actor = Actor(
            category="raw-name",
            uuid="u-raw",
            description_pool=ActorDescription(
                seen=["seen-desc"],
                unseen=["unseen-desc"],
            ),
        )
        text = wrapper.render(actors=actor, obj_desc_mode="raw")
        assert text == "name=raw-name,desc=raw-name"

    def test_render_with_obj_desc_mode_seen_requires_seed(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {"raw": "desc={actor.description}", "seen": [], "unseen": []},
        )
        wrapper = InstructionWrapper(template_json)
        actor = Actor(
            category="raw-name",
            uuid="u-seed",
            description_pool=ActorDescription(
                seen=["seen-a"],
                unseen=["unseen-a"],
            ),
        )

        with pytest.raises(
            InstructionRenderError,
            match="obj_desc_seed is required",
        ):
            wrapper.render(actors=actor, obj_desc_mode="seen")

    def test_render_with_obj_desc_mode_empty_candidates_raises(self, tmp_path):
        template_json = _write_template_json(
            tmp_path,
            {"raw": "desc={actor.description}", "seen": [], "unseen": []},
        )
        wrapper = InstructionWrapper(template_json)
        actor = Actor(
            category="raw-name",
            uuid="u-empty",
            description_pool=ActorDescription(seen=[], unseen=[]),
        )

        with pytest.raises(
            InstructionRenderError,
            match="description 'seen' is empty",
        ):
            wrapper.render(actors=actor, obj_desc_mode="seen", obj_desc_seed=1)


class TestActorStoreAdapter:
    def test_build_index_fallback_to_stem_by_default(self, tmp_path):
        _write_json(
            tmp_path / "obj-1.json",
            {"category": "plate", "seen": [], "unseen": []},
        )
        store = ActorStoreAdapter(root=tmp_path)
        assert "obj-1" in store
        assert store.get_path("obj-1") == (tmp_path / "obj-1.json").resolve()

    def test_build_index_no_fallback_to_stem(self, tmp_path):
        _write_json(
            tmp_path / "obj-1.json",
            {"category": "plate", "seen": [], "unseen": []},
        )
        store = ActorStoreAdapter(root=tmp_path, fallback_to_stem=False)
        assert "obj-1" not in store
        assert len(store) == 0

    def test_build_index_raise_on_duplicate_uuid(self, tmp_path):
        _write_json(tmp_path / "a.json", {"uuid": "dup", "category": "x"})
        nested = tmp_path / "nested"
        nested.mkdir()
        _write_json(nested / "b.json", {"uuid": "dup", "category": "y"})

        with pytest.raises(ValueError, match="Duplicate uuid"):
            ActorStoreAdapter(root=tmp_path)

    def test_get_actor_json_returns_none_for_missing_or_invalid(
        self, tmp_path
    ):
        _write_json(tmp_path / "ok.json", {"uuid": "ok", "category": "a"})
        broken = tmp_path / "broken.json"
        broken.write_text("{bad-json", encoding="utf-8")
        store = ActorStoreAdapter(root=tmp_path)

        assert store.get_actor_json("missing") is None
        assert store.get_actor_json("broken") is None

    def test_get_actor_json_raise_for_non_mapping_payload(self, tmp_path):
        _write_json(tmp_path / "arr.json", {"uuid": "arr"})
        (tmp_path / "arr.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(AttributeError, match="get"):
            ActorStoreAdapter(root=tmp_path)


class TestActorFromUuid:
    def test_from_uuid_with_mapping_payload(self, tmp_path):
        payload = {
            "category": "alice",
            "uuid": "u-1",
            "seen": ["apple"],
            "unseen": ["orange"],
        }
        _write_json(tmp_path / "u-1.json", payload)

        store = ActorStoreAdapter(root=tmp_path)
        actor = Actor.from_uuid("u-1", store=store)

        assert actor.category == "alice"
        assert actor.uuid == "u-1"
        assert actor.description_pool.seen == ["apple"]
        assert actor.description_pool.unseen == ["orange"]

    def test_from_uuid_raise_on_missing_payload(self, tmp_path):
        store = ActorStoreAdapter(root=tmp_path)
        with pytest.raises(InstructionRenderError, match="Actor not found"):
            Actor.from_uuid("u-3", store=store)

    def test_from_uuid_raise_on_missing_required_fields(self, tmp_path):
        _write_json(tmp_path / "u-4.json", {"uuid": "u-4"})
        store = ActorStoreAdapter(root=tmp_path)
        with pytest.raises(
            InstructionRenderError,
            match="missing required fields",
        ):
            Actor.from_uuid("u-4", store=store)

    def test_from_uuid_raise_on_uuid_mismatch(self, tmp_path):
        class _Store:
            def get_actor_json(self, uuid: str):
                return {
                    "uuid": "u-other",
                    "category": "alice",
                    "seen": [],
                    "unseen": [],
                }

        store = _Store()
        with pytest.raises(InstructionRenderError, match="uuid mismatch"):
            Actor.from_uuid("u-5", store=store)


def test_render_with_actor_store_and_multiple_actors(tmp_path):
    _write_json(
        tmp_path / "obj-1.json",
        {
            "category": "plate",
            "uuid": "obj-1",
            "seen": ["small plate", "white plate"],
            "unseen": ["ceramic dish"],
        },
    )
    _write_json(
        tmp_path / "obj-2.json",
        {
            "category": "cup",
            "uuid": "obj-2",
            "seen": ["red cup", "blue cup"],
            "unseen": ["glass cup"],
        },
    )

    template_json = _write_template_json(
        tmp_path,
        {
            "raw": (
                "Pick {actor1.description} ({actor1.uuid}) "
                "then pass to {actor2.description} ({actor2.uuid})"
            ),
            "seen": [],
            "unseen": [],
        },
    )

    store = ActorStoreAdapter(root=tmp_path)
    actor1 = Actor.from_uuid("obj-1", store=store)
    actor2 = Actor.from_uuid("obj-2", store=store)
    wrapper = InstructionWrapper(template_json)

    text1 = wrapper.render(
        actors={"actor1": actor1, "actor2": actor2},
        obj_desc_mode="seen",
        obj_desc_seed=0,
    )
    text2 = wrapper.render(
        actors={"actor1": actor1, "actor2": actor2},
        obj_desc_mode="seen",
        obj_desc_seed=0,
    )

    assert text1 == text2
    assert text1 in {
        "Pick white plate (obj-1) then pass to blue cup (obj-2)",
        "Pick small plate (obj-1) then pass to red cup (obj-2)",
    }
