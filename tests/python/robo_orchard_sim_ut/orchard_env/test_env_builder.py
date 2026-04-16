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

"""Tests for orchard env builder phase 1.5 refactor."""

import types

import pytest
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
)

from robo_orchard_sim.cfg_wrappers.assets_cfg import (
    ArticulationCfg,
    AssetBaseCfg,
)
from robo_orchard_sim.cfg_wrappers.envs.env_cfg import ViewerCfg
from robo_orchard_sim.cfg_wrappers.sim.simulation_cfg import SimulationCfg
from robo_orchard_sim.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.cfg_wrappers.sim.spawners.lights_cfg import (
    DomeLightCfg,
)
from robo_orchard_sim.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.models.scenes.asset_scene import AssetSceneCfg
from robo_orchard_sim.models.scenes.interactive_scene import InteractiveScene
from robo_orchard_sim.orchard_env.assets import (
    ArticulationSpec,
    CustomAssetSpec,
    RigidObjectSpec,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper import (
    DualArmPiperEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.env_builder.builder import EnvBuilder
from robo_orchard_sim.orchard_env.scene.plane_table_scene import (
    PlaneTableScene,
)
from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase
from robo_orchard_sim.orchard_env.tasks.place_a2b_task import (
    PlaceA2BTask,
    PlaceA2BTaskAssets,
)
from robo_orchard_sim.orchard_env.tasks.task_base import TaskBase
from robo_orchard_sim.tasks.validators.base import Validator
from robo_orchard_sim.tasks.validators.checkers import (
    LiftChecker,
    ReachChecker,
    WithinXYChecker,
)


def _make_asset_cfg(name: str) -> AssetBaseCfg:
    return AssetBaseCfg(
        class_type=XFormPrimAsset,
        prim_path=f"/World/{name}",
        spawn=DomeLightCfg(intensity=1.0),
    )


class DummyScene(SceneBase):
    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        return {
            "lights": GroupAssetCfg(dome=_make_asset_cfg("lights_dome")),
            "terrain": GroupAssetCfg(ground=_make_asset_cfg("terrain_ground")),
        }

    def get_sim_cfg(self) -> SimulationCfg:
        return SimulationCfg(dt=0.01)

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.0, 1.0, 1.0), lookat=(0.0, 0.0, 0.0))

    def get_decimation(self) -> int:
        return 2

    def get_num_envs(self) -> int:
        return 4

    def get_env_spacing(self) -> float:
        return 3.0


class IncompleteScene(SceneBase):
    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        return {}

    def get_sim_cfg(self) -> SimulationCfg:
        return SimulationCfg(dt=0.01)

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.0, 1.0, 1.0), lookat=(0.0, 0.0, 0.0))


class DummyEmbodiment(EmbodimentBase):
    def __init__(self):
        super().__init__(
            robot=ArticulationSpec(
                name="dualarm",
                namespace="robots",
                template_cfg=ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/template_robot",
                    spawn=UsdFileCfg(usd_path="/tmp/robot.usd"),
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={}),
                    actuators={},
                ),
            )
        )


class DummyTask(TaskBase):
    def __init__(self):
        super().__init__(
            assets={
                "pick": RigidObjectSpec(
                    name="pick_object",
                    usd_path="/tmp/pick.usd",
                ),
                "place": RigidObjectSpec(
                    name="place_object",
                    usd_path="/tmp/place.usd",
                ),
            }
        )

    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        return super().get_assets_cfg()

    def get_event_cfg(self) -> EventManagerCfg:
        return EventManagerCfg(terms={})

    def get_observation_cfg(self) -> ObservationManagerCfg:
        return ObservationManagerCfg(groups={})

    def get_action_cfg(self) -> ActionManagerCfg:
        return ActionManagerCfg(terms={})

    def build_validator(self) -> Validator:
        return Validator(
            actors=[
                "objects/pick_object",
                "objects/place_object",
            ],
            criteria=[],
            criteria_name=[],
        )


def test_asset_spec_name_rejects_path_separator():
    with pytest.raises(ValueError, match="must not contain '/'"):
        RigidObjectSpec(
            name="bad/name",
            namespace="objects",
            usd_path="/tmp/bad.usd",
        )


def test_asset_spec_with_default_namespace_sets_missing_value_only():
    spec = CustomAssetSpec(
        name="task_light",
        cfg=_make_asset_cfg("task_light"),
    )

    updated = spec.with_default_namespace("lights")
    preserved = updated.with_default_namespace("objects")

    assert updated.namespace == "lights"
    assert updated.scene_name == "lights/task_light"
    assert preserved.namespace == "lights"


def test_env_builder_aggregates_assets_by_namespace_into_asset_scene_cfg():
    env_cfg = EnvBuilder(
        scene=DummyScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    ).build()

    assert isinstance(env_cfg.scene, AssetSceneCfg)
    assert env_cfg.scene.num_envs == 4
    assert env_cfg.scene.env_spacing == 3.0
    assert set(env_cfg.scene.assets) == {
        "lights",
        "objects",
        "robots",
        "terrain",
    }
    assert set(env_cfg.scene.assets["objects"]) == {
        "pick_object",
        "place_object",
    }
    assert set(env_cfg.scene.assets["robots"]) == {"dualarm"}


def test_env_builder_uses_scene_base_default_layout_values():
    env_cfg = EnvBuilder(
        scene=IncompleteScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    ).build()

    assert env_cfg.scene.num_envs == 1
    assert env_cfg.scene.env_spacing == 2.5


def test_interactive_scene_flattens_namespace_assets_without_keyerror(
    monkeypatch,
):
    scene_cfg = AssetSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        assets={
            "objects": GroupAssetCfg(cube=_make_asset_cfg("cube")),
            "robots": GroupAssetCfg(arm=_make_asset_cfg("arm")),
        },
    )
    scene = object.__new__(InteractiveScene)
    scene.cfg = scene_cfg
    scene.__dict__["env_regex_ns"] = "/World/envs/env_.*"
    scene._add_asset = types.MethodType(lambda self, name, cfg: None, scene)
    monkeypatch.setattr(
        InteractiveScene.__mro__[1],
        "_add_entities_from_cfg",
        lambda self: None,
    )

    scene._add_entities_from_cfg()

    assert "objects" in scene.cfg.__dict__["assets"]
    assert "robots" in scene.cfg.__dict__["assets"]


def test_place_a2b_task_build_validator_encodes_task_success_semantics():
    task = PlaceA2BTask(
        assets=PlaceA2BTaskAssets(
            pick=RigidObjectSpec(
                name="pick_object",
                usd_path="/tmp/pick.usd",
            ),
            place=RigidObjectSpec(
                name="place_object",
                usd_path="/tmp/place.usd",
            ),
        )
    )

    validator = task.build_validator()

    assert validator.actors == [
        "objects/pick_object",
        "objects/place_object",
    ]
    assert validator.criteria_name == [
        "reach_pick",
        "lift_pick",
        "reach_place",
        "place_within_xy",
    ]
    assert len(validator.criteria) == 4
    assert isinstance(validator.criteria[0], ReachChecker)
    assert validator.criteria[0].actor_name == "objects/pick_object"

    assert isinstance(validator.criteria[1], tuple)
    checker, deps = validator.criteria[1]
    assert isinstance(checker, LiftChecker)
    assert checker.actor_name == "objects/pick_object"
    assert deps == [0]

    assert isinstance(validator.criteria[2], tuple)
    checker, deps = validator.criteria[2]
    assert isinstance(checker, WithinXYChecker)
    assert checker.actor1 == "objects/pick_object"
    assert checker.actor2 == "objects/place_object"
    assert checker.gripper_checker is None
    assert deps == [1]

    assert isinstance(validator.criteria[3], tuple)
    checker, deps = validator.criteria[3]
    assert isinstance(checker, WithinXYChecker)
    assert checker.actor1 == "objects/pick_object"
    assert checker.actor2 == "objects/place_object"
    assert checker.gripper_checker is not None
    assert deps == [2]


def test_place_a2b_task_resets_pick_place_and_distractors_in_one_pose_event():
    task = PlaceA2BTask(
        assets=PlaceA2BTaskAssets(
            pick=RigidObjectSpec(
                name="pick_object",
                usd_path="/tmp/pick.usd",
            ),
            place=RigidObjectSpec(
                name="place_object",
                usd_path="/tmp/place.usd",
            ),
            distractors=[
                RigidObjectSpec(
                    name="distractor_0",
                    usd_path="/tmp/distractor_0.usd",
                ),
                RigidObjectSpec(
                    name="distractor_1",
                    usd_path="/tmp/distractor_1.usd",
                ),
            ],
        )
    )

    event_cfg = task.get_event_cfg()

    assert list(event_cfg.terms) == ["random_pose_event"]
    pose_event = event_cfg.terms["random_pose_event"]
    assert [asset_cfg.name for asset_cfg in pose_event.asset_cfgs] == [
        "objects/place_object",
        "objects/pick_object",
        "objects/distractor_0",
        "objects/distractor_1",
    ]
    assert pose_event.mode == "random_non_overlap"
    assert pose_event.clear_cross_group_cache is True


def test_articulation_spec_patches_identity_and_preserves_template_cfg():
    template_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/template_name",
        spawn=UsdFileCfg(usd_path="/tmp/robot.usd"),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={"joint1": 0.1},
        ),
        actuators={},
    )
    spec = ArticulationSpec(
        name="robot",
        namespace="robots",
        template_cfg=template_cfg,
        initial_pos=(0.1, 0.2, 0.3),
    )

    cfg = spec.to_isaac_cfg()

    assert isinstance(cfg, ArticulationCfg)
    assert cfg.prim_path == "{ENV_REGEX_NS}/robot"
    assert cfg.spawn.usd_path == "/tmp/robot.usd"
    assert cfg.init_state.pos == (0.1, 0.2, 0.3)
    assert template_cfg.prim_path == "{ENV_REGEX_NS}/template_name"


def test_plane_table_scene_allows_user_defined_custom_asset():
    scene = PlaneTableScene(
        assets=[
            CustomAssetSpec(
                name="light_fill",
                namespace="background",
                cfg=_make_asset_cfg("light_fill"),
            )
        ]
    )

    grouped = scene.get_assets_cfg()

    assert "background" in grouped
    assert "light_fill" in grouped["background"]


def test_place_a2b_task_accepts_multiple_distractor_assets():
    task = PlaceA2BTask(
        assets=PlaceA2BTaskAssets(
            pick=RigidObjectSpec(
                name="pick_object",
                usd_path="/tmp/pick.usd",
            ),
            place=RigidObjectSpec(
                name="place_object",
                usd_path="/tmp/place.usd",
            ),
            distractors=[
                RigidObjectSpec(
                    name="pick_distractor_0",
                    usd_path="/tmp/pick_d0.usd",
                ),
                RigidObjectSpec(
                    name="place_distractor_0",
                    usd_path="/tmp/place_d0.usd",
                ),
                RigidObjectSpec(
                    name="place_distractor_1",
                    usd_path="/tmp/place_d1.usd",
                ),
            ],
        )
    )

    grouped = task.get_assets_cfg()

    assert "objects" in grouped
    assert set(grouped["objects"]) == {
        "pick_object",
        "place_object",
        "pick_distractor_0",
        "place_distractor_0",
        "place_distractor_1",
    }


def test_place_a2b_task_assets_reject_non_object_pick_or_place():
    with pytest.raises(TypeError):
        PlaceA2BTaskAssets(
            pick=CustomAssetSpec(
                name="task_light",
                cfg=_make_asset_cfg("task_light"),
            ),
            place=RigidObjectSpec(
                name="place_object",
                usd_path="/tmp/place.usd",
            ),
        )


def test_place_a2b_assets_defaults_distractors_to_empty():
    assets = PlaceA2BTaskAssets(
        pick=RigidObjectSpec(
            name="pick_object",
            usd_path="/tmp/pick.usd",
        ),
        place=RigidObjectSpec(
            name="place_object",
            usd_path="/tmp/place.usd",
        ),
    )

    assert assets.flatten() == {
        "pick": assets.pick,
        "place": assets.place,
    }


def test_dualarm_piper_embodiment_provides_default_cfg_entries():
    embodiment = DualArmPiperEmbodiment()

    action_cfg = embodiment.get_action_cfg()
    observation_cfg = embodiment.get_observation_cfg()

    assert set(action_cfg.terms) == {
        "left_robot_joint_position",
        "left_robot_gripper_control",
        "right_robot_joint_position",
        "right_robot_gripper_control",
    }
    assert "/robot" in observation_cfg.groups
    assert "/tf" in observation_cfg.groups
    assert "base_link" in observation_cfg.groups["/robot"].terms
    assert "left_robot_tf" in observation_cfg.groups["/tf"].terms
    assert "right_robot_tf" in observation_cfg.groups["/tf"].terms


def test_dualarm_piper_embodiment_can_disable_cameras():
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)

    assets_cfg = embodiment.get_assets_cfg()
    observation_cfg = embodiment.get_observation_cfg()

    assert "cameras" not in assets_cfg
    assert "/camera" not in observation_cfg.groups
