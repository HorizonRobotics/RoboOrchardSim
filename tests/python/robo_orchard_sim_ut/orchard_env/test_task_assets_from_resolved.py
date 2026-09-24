## Copyright (c) 2026 Horizon Robotics. All Rights Reserved.

"""TaskAssets.from_resolved keeps the grouping the task YAML declared."""

import pytest

from robo_orchard_sim.orchard_env.assets.object_spec import RigidObjectSpec
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
from robo_orchard_sim.orchard_env.task_templates.place_a2b_task import (
    PlaceA2BTask,
)


def _obj(name: str) -> RigidObjectSpec:
    return RigidObjectSpec(name=name, usd_path=f"/tmp/{name}.usd")


def test_distractor_groups_stay_separate():
    # The YAML distinguishes clutter around the pick object from clutter
    # around the place target; folding both into one bucket would lose
    # which is which.
    assets = TaskAssets.from_resolved(
        {
            "pick": _obj("pick_object"),
            "place": _obj("place_object"),
            "distractors_graspable": [_obj("g_0"), _obj("g_1")],
            "distractors_container": [_obj("c_0")],
        }
    )

    assert [s.scene_name for s in assets.by_role("distractors_graspable")] == [
        "g_0",
        "g_1",
    ]
    assert [s.scene_name for s in assets.by_role("distractors_container")] == [
        "c_0"
    ]


def test_single_spec_and_list_both_become_lists():
    assets = TaskAssets.from_resolved(
        {"pick": _obj("pick_object"), "distractors": [_obj("d_0")]}
    )

    assert [s.scene_name for s in assets.by_role("pick")] == ["pick_object"]
    assert [s.scene_name for s in assets.by_role("distractors")] == ["d_0"]


def test_absent_role_reads_as_empty():
    assets = TaskAssets.from_resolved({"pick": _obj("pick_object")})

    assert assets.by_role("distractors") == []


def test_task_rejects_assets_missing_a_declared_role():
    with pytest.raises(ValueError, match="place"):
        PlaceA2BTask(TaskAssets.from_resolved({"pick": _obj("pick_object")}))
