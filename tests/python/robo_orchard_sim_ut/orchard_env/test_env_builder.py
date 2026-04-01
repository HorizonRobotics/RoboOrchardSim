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
from robo_orchard_sim.models.assets.rigid_object import RigidObjectCfg
from robo_orchard_sim.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.models.scenes.asset_scene import AssetSceneCfg
from robo_orchard_sim.models.scenes.interactive_scene import InteractiveScene
from robo_orchard_sim.orchard_env.assets import (
    ArticulationSpec,
    AssetSpec,
    CustomAssetSpec,
    ObjectSpec,
    RigidObjectSpec,
)
from robo_orchard_sim.orchard_env.embodiments.dualarm_piper import (
    DualArmPiperEmbodiment,
)
from robo_orchard_sim.orchard_env.embodiments.embodiment_base import (
    EmbodimentBase,
)
from robo_orchard_sim.orchard_env.env_builder.builder import EnvBuilder
from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
from robo_orchard_sim.orchard_env.scene.plane_table_scene import (
    PlaneTableScene,
)
from robo_orchard_sim.orchard_env.scene.scene_base import SceneBase
from robo_orchard_sim.orchard_env.tasks.place_a2b_task import (
    PlaceA2BRole,
    PlaceA2BTask,
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


def test_rigid_object_spec_scene_name_is_derived_from_namespace_and_name():
    spec = RigidObjectSpec(
        name="pick_object",
        namespace="objects",
        usd_path="/tmp/pick.usd",
    )

    assert isinstance(spec, AssetSpec)
    assert isinstance(spec, ObjectSpec)
    assert spec.namespace == "objects"
    assert spec.name == "pick_object"
    assert spec.scene_name == "objects/pick_object"


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


def test_asset_scene_cfg_accepts_namespace_group_assets():
    scene_cfg = AssetSceneCfg(
        num_envs=1,
        env_spacing=2.0,
        assets={
            "objects": GroupAssetCfg(cube=_make_asset_cfg("cube")),
            "lights": GroupAssetCfg(dome=_make_asset_cfg("dome")),
        },
    )

    assert set(scene_cfg.assets) == {"objects", "lights"}
    assert set(scene_cfg.assets["objects"]) == {"cube"}
    assert set(scene_cfg.assets["lights"]) == {"dome"}


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
        assets={
            PlaceA2BRole.PICK: RigidObjectSpec(
                name="pick_object",
                usd_path="/tmp/pick.usd",
            ),
            PlaceA2BRole.PLACE: RigidObjectSpec(
                name="place_object",
                usd_path="/tmp/place.usd",
            ),
        }
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


def test_rigid_object_spec_exposes_user_facing_object_fields():
    spec = RigidObjectSpec(
        name="plate",
        usd_path="/tmp/plate.usd",
        interaction_path=None,
        namespace=None,
        mass=1.0,
        scale=(1.0, 2.0, 3.0),
        initial_pos=(0.1, 0.2, 0.3),
        initial_rot=(1.0, 0.0, 0.0, 0.0),
    )

    assert spec.name == "plate"
    assert spec.namespace is None
    assert spec.usd_path == "/tmp/plate.usd"
    assert spec.mass == 1.0
    assert spec.scale == (1.0, 2.0, 3.0)


def test_rigid_object_spec_converts_to_rigid_object_cfg():
    spec = RigidObjectSpec(
        name="plate",
        usd_path="/tmp/plate.usd",
        namespace="objects",
    )

    cfg = spec.to_isaac_cfg()

    assert isinstance(cfg, RigidObjectCfg)
    assert cfg.prim_path == "{ENV_REGEX_NS}/plate"
    assert cfg.spawn.usd_path == "/tmp/plate.usd"


def test_articulation_spec_converts_to_articulation_cfg():
    spec = ArticulationSpec(
        name="robot",
        namespace="robots",
        template_cfg=ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/template_robot",
            spawn=UsdFileCfg(usd_path="/tmp/robot.usd"),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={}),
            actuators={},
        ),
    )

    cfg = spec.to_isaac_cfg()

    assert isinstance(cfg, ArticulationCfg)
    assert cfg.prim_path == "{ENV_REGEX_NS}/robot"
    assert cfg.spawn.usd_path == "/tmp/robot.usd"


def test_articulation_spec_patches_template_cfg():
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

    assert cfg.prim_path == "{ENV_REGEX_NS}/robot"
    assert cfg.init_state.pos == (0.1, 0.2, 0.3)
    assert template_cfg.prim_path == "{ENV_REGEX_NS}/template_name"


def test_custom_asset_spec_returns_wrapped_cfg():
    cfg = _make_asset_cfg("custom_light")
    spec = CustomAssetSpec(
        name="light",
        namespace="background",
        cfg=cfg,
    )

    assert spec.to_isaac_cfg() is cfg


def test_embodiment_base_proxies_robot_spec_identity():
    embodiment = DummyEmbodiment()

    assert embodiment.name == "dualarm"
    assert embodiment.namespace == "robots"
    assert embodiment.scene_name == "robots/dualarm"


def test_orchard_env_uses_plane_table_scene_by_default_and_builds_cfg():
    orchard_env = OrchardEnv(
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    )

    assert isinstance(orchard_env.scene, PlaneTableScene)
    assert orchard_env.embodiment.name == "dualarm"
    env_cfg = orchard_env.to_isaac_env_cfg()
    assert isinstance(env_cfg.scene, AssetSceneCfg)
    assert {"background", "objects", "robots"} <= set(env_cfg.scene.assets)


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


def test_place_a2b_task_accepts_mixed_asset_specs_with_role_mapping():
    task = PlaceA2BTask(
        assets={
            PlaceA2BRole.PICK: RigidObjectSpec(
                name="pick_object",
                usd_path="/tmp/pick.usd",
            ),
            PlaceA2BRole.PLACE: RigidObjectSpec(
                name="place_object",
                usd_path="/tmp/place.usd",
            ),
            PlaceA2BRole.OTHER: CustomAssetSpec(
                name="task_light",
                namespace="lights",
                cfg=_make_asset_cfg("task_light"),
            ),
        }
    )

    grouped = task.get_assets_cfg()

    assert "objects" in grouped
    assert set(grouped["objects"]) == {"pick_object", "place_object"}
    assert "lights" in grouped
    assert set(grouped["lights"]) == {"task_light"}


def test_place_a2b_task_asset_specs_use_to_isaac_cfg_entrypoint(monkeypatch):
    converted_names: list[str] = []
    original = RigidObjectSpec.to_isaac_cfg

    def _wrapped(self: RigidObjectSpec):
        converted_names.append(self.name)
        return original(self)

    monkeypatch.setattr(RigidObjectSpec, "to_isaac_cfg", _wrapped)

    task = PlaceA2BTask(
        assets={
            PlaceA2BRole.PICK: RigidObjectSpec(
                name="pick_object",
                usd_path="/tmp/pick.usd",
            ),
            PlaceA2BRole.PLACE: RigidObjectSpec(
                name="place_object",
                usd_path="/tmp/place.usd",
            ),
        }
    )

    grouped = task.get_assets_cfg()

    assert "objects" in grouped
    assert converted_names == ["pick_object", "place_object"]


def test_place_a2b_task_rejects_missing_required_roles():
    with pytest.raises(ValueError, match="must include"):
        PlaceA2BTask(
            assets={
                PlaceA2BRole.PICK: RigidObjectSpec(
                    name="pick_object",
                    usd_path="/tmp/pick.usd",
                )
            }
        )


def test_place_a2b_task_rejects_non_object_required_roles():
    with pytest.raises(
        TypeError,
        match="must be ObjectSpec instances",
    ):
        PlaceA2BTask(
            assets={
                PlaceA2BRole.PICK: CustomAssetSpec(
                    name="task_light",
                    cfg=_make_asset_cfg("task_light"),
                ),
                PlaceA2BRole.PLACE: RigidObjectSpec(
                    name="place_object",
                    usd_path="/tmp/place.usd",
                ),
            }
        )


def test_dualarm_piper_embodiment_provides_robot_action_terms():
    embodiment = DualArmPiperEmbodiment()

    action_cfg = embodiment.get_action_cfg()

    assert set(action_cfg.terms) == {
        "left_robot_joint_position",
        "left_robot_gripper_control",
        "right_robot_joint_position",
        "right_robot_gripper_control",
    }


def test_dualarm_piper_embodiment_provides_robot_and_tf_observation_groups():
    embodiment = DualArmPiperEmbodiment()

    observation_cfg = embodiment.get_observation_cfg()

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
