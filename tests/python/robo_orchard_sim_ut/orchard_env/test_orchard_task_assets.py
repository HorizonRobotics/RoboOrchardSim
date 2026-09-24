## Copyright (c) 2026 Horizon Robotics. All Rights Reserved.

"""Unit tests for the TaskAssets build-time container."""

from robo_orchard_sim.orchard_env.assets import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets


def _obj(name: str) -> RigidObjectSpec:
    return RigidObjectSpec(name=name, usd_path=f"/tmp/{name}.usd")


def _assets() -> TaskAssets:
    return TaskAssets(
        role_candidates={
            "pick": [_obj(f"pick_{i}") for i in range(5)],
            "distractors_pick": [_obj(f"dpick_{i}") for i in range(3)],
        }
    )


def test_flatten_returns_all_candidates_keyed_by_scene_name():
    flattened = _assets().flatten()

    assert len(flattened) == 8
    assert "pick_0" in flattened
    assert "dpick_2" in flattened


def test_all_scene_names_and_flatten_agree_on_order():
    assets = _assets()

    assert list(assets.flatten()) == assets.all_scene_names()


def test_by_role_returns_candidates_for_that_role():
    assets = _assets()

    assert [s.scene_name for s in assets.by_role("pick")] == [
        f"pick_{i}" for i in range(5)
    ]
    assert assets.by_role("nonexistent") == []


def test_from_resolved_treats_single_and_multi_roles_alike():
    resolved = {
        "pick": [_obj("pick_0"), _obj("pick_1")],
        "anchor": _obj("anchor_object"),
    }

    assets = TaskAssets.from_resolved(resolved)

    assert [s.scene_name for s in assets.by_role("pick")] == [
        "pick_0",
        "pick_1",
    ]
    assert [s.scene_name for s in assets.by_role("anchor")] == [
        "anchor_object"
    ]
