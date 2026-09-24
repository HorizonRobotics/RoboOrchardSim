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

import pytest
import torch
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
)

from robo_orchard_sim.benchmark.manipulation.place_a2b import (
    PlaceA2BTaskDefinition,
)
from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import (
    ArticulationCfg,
    AssetBaseCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.envs.env_cfg import ViewerCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.simulation_cfg import SimulationCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.lights_cfg import (
    DomeLightCfg,
)
from robo_orchard_sim.ext.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_sim.ext.envs.managers.record import (
    EpisodeRecordControllerCfg,
    NoOpRecordControllerCfg,
    RecordTermBaseCfg,
    StationaryEpisodeRecordControllerCfg,
)
from robo_orchard_sim.ext.envs.managers.record.mcap import (
    McapImageTermCfg,
    McapTFTermCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.ext.models.assets.xform_asset import XFormPrimAsset
from robo_orchard_sim.ext.models.scenes.asset_scene import AssetSceneCfg
from robo_orchard_sim.orchard_env import OrchardEnv
from robo_orchard_sim.orchard_env.assets import (
    ArticulationSpec,
    CustomAssetSpec,
    RigidObjectSpec,
)
from robo_orchard_sim.orchard_env.assets.task_assets import TaskAssets
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
from robo_orchard_sim.orchard_env.task_templates.place_a2b_task import (
    PICK_ROLE,
    PLACE_ROLE,
    PlaceA2BTask,
    PlaceA2BTaskParams,
)
from robo_orchard_sim.orchard_env.task_templates.task_base import TaskBase
from robo_orchard_sim.orchard_env.task_templates.task_params import (
    PoseRangeConfig,
    TaskPoseResetConfig,
)
from robo_orchard_sim.task_components.role_registry import (
    RoleRegistry,
    TargetRef,
)
from robo_orchard_sim.task_components.validators.base import (
    GripperRange,
    Validator,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
    ValidatorRobotContext,
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
            assets=TaskAssets(
                role_candidates={
                    "pick": [
                        RigidObjectSpec(
                            name="pick_object",
                            usd_path="/tmp/pick.usd",
                        )
                    ],
                    "place": [
                        RigidObjectSpec(
                            name="place_object",
                            usd_path="/tmp/place.usd",
                        )
                    ],
                }
            )
        )

    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        return super().get_assets_cfg()

    def get_role_candidates(self, role_id: str, *, swap: bool = False) -> list:
        del role_id, swap
        return []

    def get_event_cfg(self) -> EventManagerCfg:
        return EventManagerCfg(terms={})

    def get_observation_cfg(self) -> ObservationManagerCfg:
        return ObservationManagerCfg(groups={})

    def get_action_cfg(self) -> ActionManagerCfg:
        return ActionManagerCfg(terms={})

    def build_validator(
        self,
        context=None,
    ) -> Validator:
        del context
        return Validator(
            criteria=[],
            criteria_name=[],
        )


class RecordScene(DummyScene):
    def get_record_terms(self) -> dict[str, RecordTermBaseCfg]:
        return {
            "scene_rgb": McapImageTermCfg(
                topic="/scene/camera/rgb",
                fps=5.0,
                key="camera/scene/rgb",
                frame_id="scene_camera",
                mode="rgb",
            )
        }


class RecordEmbodiment(DummyEmbodiment):
    def get_record_terms(self) -> dict[str, RecordTermBaseCfg]:
        return {
            "embodiment_rgb": McapImageTermCfg(
                topic="/embodiment/camera/rgb",
                fps=10.0,
                key="camera/embodiment/rgb",
                frame_id="embodiment_camera",
                mode="rgb",
            )
        }


class RecordTask(DummyTask):
    def get_record_terms(self) -> dict[str, RecordTermBaseCfg]:
        return {
            "task_rgb": McapImageTermCfg(
                topic="/task/camera/rgb",
                fps=15.0,
                key="camera/task/rgb",
                frame_id="task_camera",
                mode="rgb",
            )
        }


class _DummyObjectData:
    def __init__(self, position: tuple[float, float, float]):
        self.root_pos_w = torch.tensor([position], dtype=torch.float32)
        self.root_quat_w = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        self.root_state_w = torch.zeros((1, 13), dtype=torch.float32)
        self.root_state_w[:, :3] = self.root_pos_w
        self.root_state_w[:, 3] = 1.0
        self.root_lin_vel_w = torch.zeros((1, 3), dtype=torch.float32)
        self.root_ang_vel_w = torch.zeros((1, 3), dtype=torch.float32)
        self.default_root_state = torch.zeros((1, 13), dtype=torch.float32)


class _DummyObject:
    def __init__(
        self,
        position: tuple[float, float, float],
        prim_path: str,
    ):
        self.data = _DummyObjectData(position)
        self.cfg = type("Cfg", (), {"prim_path": prim_path})()

    def set_position(self, position: tuple[float, float, float]) -> None:
        new_position = torch.tensor(position, dtype=torch.float32)
        self.data.root_pos_w[0] = new_position
        self.data.root_state_w[0, :3] = new_position


class _DummyRobotData:
    def __init__(self):
        self.body_com_pos_w = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.body_pos_w = self.body_com_pos_w
        self.body_quat_w = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 6])
        self.joint_pos = torch.zeros((1, 4), dtype=torch.float32)


class _DummyRobot:
    def __init__(self):
        self.data = _DummyRobotData()

    def find_bodies(self, name: str):
        body_names = [
            "left_link6",
            "right_link6",
            "left_link7",
            "left_link8",
            "right_link7",
            "right_link8",
        ]
        return [body_names.index(name)], [name]

    def find_joints(self, name: str):
        joint_names = [
            "left_joint7",
            "left_joint8",
            "right_joint7",
            "right_joint8",
        ]
        return [joint_names.index(name)], [name]

    def set_gripper_positions(
        self,
        left: float,
        right: float,
        *,
        left_mirror: float | None = None,
        right_mirror: float | None = None,
    ) -> None:
        self.data.joint_pos[0] = torch.tensor(
            [
                left,
                left_mirror if left_mirror is not None else -left,
                right,
                right_mirror if right_mirror is not None else -right,
            ],
            dtype=torch.float32,
        )


class _DummyValidatorScene(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = object()


class _DummyContactSensor:
    def __init__(self) -> None:
        self.cfg = type("Cfg", (), {"filter_prim_paths_expr": []})()
        self.contact_physx_view = type(
            "ContactView",
            (),
            {"filter_paths": [[]]},
        )()
        self.data = type(
            "Data", (), {"force_matrix_w": None, "contact_pos_w": None}
        )()

    def _initialize_impl(self) -> None:
        paths = list(self.cfg.filter_prim_paths_expr)
        self.contact_physx_view.filter_paths = [paths]
        self.data.force_matrix_w = torch.zeros(
            (1, 1, len(paths), 3),
            dtype=torch.float32,
        )
        self.data.contact_pos_w = torch.full_like(
            self.data.force_matrix_w, torch.nan
        )

    def set_force(
        self,
        prim_path: str,
        force: tuple[float, float, float],
        position: tuple[float, float, float],
    ) -> None:
        column = self.contact_physx_view.filter_paths[0].index(prim_path)
        self.data.force_matrix_w[0, 0, column] = torch.tensor(force)
        self.data.contact_pos_w[0, 0, column] = torch.tensor(position)


class _DummyValidatorEnv:
    def __init__(self, scene: _DummyValidatorScene):
        self.num_envs = 1
        self.scene = scene
        self.allow_xy_match = False


def _make_validator_env() -> _DummyValidatorEnv:
    scene = _DummyValidatorScene(
        {
            "objects/pick_object": _DummyObject(
                position=(0.6, 0.0, 0.0),
                prim_path="/World/envs/env_.*/pick_object",
            ),
            "objects/place_object": _DummyObject(
                position=(0.5, 0.0, 0.0),
                prim_path="/World/envs/env_.*/place_object",
            ),
            "robots/dualarm_piper": _DummyRobot(),
            "sensors/gripper_contact_0_0": _DummyContactSensor(),
            "sensors/gripper_contact_0_1": _DummyContactSensor(),
            "sensors/gripper_contact_1_0": _DummyContactSensor(),
            "sensors/gripper_contact_1_1": _DummyContactSensor(),
        }
    )
    return _DummyValidatorEnv(scene=scene)


def test_asset_spec_name_with_path_separator_raises_value_error():
    with pytest.raises(ValueError, match="must not contain '/'"):
        RigidObjectSpec(
            name="bad/name",
            namespace="objects",
            usd_path="/tmp/bad.usd",
        )


def test_asset_spec_with_default_namespace_missing_sets_namespace_and_path():
    spec = CustomAssetSpec(
        name="task_light",
        cfg=_make_asset_cfg("task_light"),
    )

    updated = spec.with_default_namespace("lights")

    assert updated.namespace == "lights"
    assert updated.scene_name == "lights/task_light"


def test_asset_spec_with_default_namespace_existing_namespace_preserved():
    spec = CustomAssetSpec(
        name="task_light",
        cfg=_make_asset_cfg("task_light"),
    )
    updated = spec.with_default_namespace("lights")

    preserved = updated.with_default_namespace("objects")

    assert preserved.namespace == "lights"


def test_env_builder_build_groups_assets_by_namespace():
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


def _make_place_a2b_task() -> PlaceA2BTask:
    return PlaceA2BTask(
        assets=TaskAssets(
            role_candidates={
                "pick": [
                    RigidObjectSpec(
                        name="pick_object", usd_path="/tmp/pick.usd"
                    )
                ],
                "place": [
                    RigidObjectSpec(
                        name="place_object", usd_path="/tmp/place.usd"
                    )
                ],
            }
        )
    )


def test_place_a2b_task_build_validator_reports_task_progress_order(
    monkeypatch,
):
    monkeypatch.setattr(
        "robo_orchard_sim.task_components.validators.utils.is_object_center_in_obb",
        lambda *_args, **_kwargs: env.allow_xy_match,
    )
    env = _make_validator_env()
    task = _make_place_a2b_task()
    role_registry = RoleRegistry()
    role_registry.bind_one(
        0,
        PICK_ROLE,
        TargetRef("objects/pick_object"),
    )
    role_registry.bind_one(
        0,
        PLACE_ROLE,
        TargetRef("objects/place_object"),
    )
    context = ValidatorContext(
        robot=ValidatorRobotContext(
            robot_name="robots/dualarm_piper",
            ee_links=("left_link6", "right_link6"),
            tcp_offsets=(
                ("left_link6", (0.0, 0.0, 0.0)),
                ("right_link6", (0.0, 0.0, 0.0)),
            ),
            gripper_joints=(
                GripperRange(name="left_joint7", open_val=0.05, close_val=0.0),
                GripperRange(
                    name="right_joint7", open_val=0.05, close_val=0.0
                ),
            ),
            gripper_body_groups=(
                ("left_link7", "left_link8"),
                ("right_link7", "right_link8"),
            ),
        ),
        role_registry=role_registry,
    )
    context.capture_init_states(
        env,
        ["objects/pick_object", "objects/place_object"],
    )
    validator = task.build_validator(context=context)
    pick_object = env.scene["objects/pick_object"]
    robot = env.scene["robots/dualarm_piper"]

    initial = validator.evaluate(env)
    pick_object.set_position((0.02, 0.0, 0.0))
    for _ in range(15):
        reached = validator.evaluate(env)
    target_path = pick_object.cfg.prim_path.replace("env_.*", "env_0")
    first_sensor = env.scene["sensors/gripper_contact_0_0"]
    second_sensor = env.scene["sensors/gripper_contact_0_1"]
    first_sensor.set_force(target_path, (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    contacted = validator.evaluate(env)
    second_sensor.set_force(target_path, (1.0, 0.0, 0.0), (0.04, 0.0, 0.0))
    grasped = validator.evaluate(env)
    pick_object.set_position((0.02, 0.0, 0.04))
    lifted = validator.evaluate(env)
    env.allow_xy_match = True
    matched_xy = validator.evaluate(env)
    robot.set_gripper_positions(left=0.05, right=0.05)
    completed = validator.evaluate(env)

    assert list(initial.metrics["criteria_reached"]) == [
        "reach_pick",
        "contact_pick",
        "grasp_pick",
        "lift_pick",
        "reach_place",
        "place_within_xy",
    ]
    assert (
        initial.progress,
        reached.progress,
        contacted.progress,
        grasped.progress,
        lifted.progress,
        matched_xy.progress,
        completed.progress,
        completed.success,
    ) == (0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0, True)


def _place_a2b_assets(
    *,
    pick_name: str = "pick_object",
    place_name: str = "place_object",
    distractors: list | None = None,
) -> TaskAssets:
    role_candidates: dict = {
        "pick": [
            RigidObjectSpec(name=pick_name, usd_path=f"/tmp/{pick_name}.usd")
        ],
        "place": [
            RigidObjectSpec(name=place_name, usd_path=f"/tmp/{place_name}.usd")
        ],
    }
    if distractors:
        role_candidates["distractors"] = distractors
    return TaskAssets(role_candidates=role_candidates)


def test_place_a2b_task_event_cfg_resets_all_objects_via_single_pose_event():
    task = PlaceA2BTask(
        assets=_place_a2b_assets(
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
    assert {asset_cfg.name for asset_cfg in pose_event.asset_cfgs} == {
        "objects/place_object",
        "objects/pick_object",
        "objects/distractor_0",
        "objects/distractor_1",
    }
    assert pose_event.mode == "random_non_overlap"


def test_place_a2b_task_operable_scene_names_include_distractors():
    task = PlaceA2BTask(
        assets=_place_a2b_assets(
            pick_name="pick",
            place_name="place",
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

    assert task.get_operable_scene_names() == [
        "objects/pick",
        "objects/place",
        "objects/distractor_0",
        "objects/distractor_1",
    ]


def test_articulation_spec_to_isaac_cfg_patches_prim_path_and_pos():
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


def test_articulation_spec_to_isaac_cfg_does_not_mutate_template_cfg():
    template_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/template_name",
        spawn=UsdFileCfg(usd_path="/tmp/robot.usd"),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={}),
        actuators={},
    )
    spec = ArticulationSpec(
        name="robot",
        namespace="robots",
        template_cfg=template_cfg,
    )

    spec.to_isaac_cfg()

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
        assets=_place_a2b_assets(
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


def test_place_a2b_task_uses_injected_pose_range_for_pose_reset():
    task = PlaceA2BTask(
        assets=TaskAssets(
            role_candidates={
                "pick": [
                    RigidObjectSpec(
                        name="pick_object", usd_path="/tmp/pick.usd"
                    )
                ],
                "place": [
                    RigidObjectSpec(
                        name="place_object", usd_path="/tmp/place.usd"
                    )
                ],
            }
        ),
        params=PlaceA2BTaskParams(
            pose_reset=TaskPoseResetConfig(
                mode="drop",
                pose_range=PoseRangeConfig(
                    x=(0.1, 0.2),
                    y=(-0.2, 0.4),
                    z=(0.01, 0.02),
                    roll=(0.0, 0.1),
                    pitch=(-0.1, 0.1),
                    yaw=(-1.0, 1.5),
                ),
                min_separation=0.07,
            ),
        ),
    )

    event_cfg = task.get_event_cfg()

    pose_event = event_cfg.terms["random_pose_event"]
    assert pose_event.mode == "drop"
    assert pose_event.pose_range == {
        "x": (0.1, 0.2),
        "y": (-0.2, 0.4),
        "z": (0.01, 0.02),
        "roll": (0.0, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-1.0, 1.5),
    }
    assert pose_event.min_separation == 0.07


def test_place_a2b_task_rejects_a_role_filled_by_a_non_object_asset():
    with pytest.raises(TypeError, match="pick"):
        PlaceA2BTask(
            assets=TaskAssets(
                role_candidates={
                    "pick": [
                        CustomAssetSpec(
                            name="task_light",
                            cfg=_make_asset_cfg("task_light"),
                        )
                    ],
                    "place": [
                        RigidObjectSpec(
                            name="place_object",
                            usd_path="/tmp/place.usd",
                        )
                    ],
                }
            )
        )


def test_env_builder_merges_record_cfg_fragments_into_env_cfg(tmp_path):
    env_cfg = EnvBuilder(
        scene=RecordScene(),
        embodiment=RecordEmbodiment(),
        task=RecordTask(),
        record_file_path=str(tmp_path),
        record_controller=EpisodeRecordControllerCfg(),
    ).build()

    assert env_cfg.records is not None
    assert env_cfg.records.file_path == str(tmp_path)
    assert isinstance(env_cfg.records.controller, EpisodeRecordControllerCfg)
    assert set(env_cfg.records.terms) == {
        "scene_rgb",
        "embodiment_rgb",
        "task_rgb",
    }


def test_env_builder_retimes_record_terms_to_the_scene_clock(tmp_path):
    class FastScene(RecordScene):
        def __init__(self):
            super().__init__(render_fps=15, step_fps=60)

    env_cfg = EnvBuilder(
        scene=FastScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
        record_file_path=str(tmp_path),
    ).build()

    assert env_cfg.records is not None
    terms = env_cfg.records.terms
    # Images follow the render clock; everything else that samples per
    # step follows the simulation clock, whatever the term declared.
    assert terms["scene_rgb"].fps == 15.0
    assert terms["object_tf_term"].fps == 60.0
    # Episode metadata is written once and never consults fps.
    assert terms["meta_dict_term"].fps == 1.0


def test_env_builder_uses_noop_controller_when_configured(tmp_path):
    env_cfg = EnvBuilder(
        scene=RecordScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
        record_file_path=str(tmp_path),
        record_controller=NoOpRecordControllerCfg(),
    ).build()

    assert env_cfg.records is not None
    assert isinstance(env_cfg.records.controller, NoOpRecordControllerCfg)


def test_env_builder_records_task_owned_terms_by_default():
    env_cfg = EnvBuilder(
        scene=DummyScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    ).build()

    assert env_cfg.records is not None
    # Every task records episode metadata, and one that owns objects
    # also records their per-frame poses; both come from ``TaskBase``.
    assert set(env_cfg.records.terms) == {"meta_dict_term", "object_tf_term"}
    assert isinstance(env_cfg.records.controller, NoOpRecordControllerCfg)


def test_orchard_env_defaults_recording_to_noop_controller():
    orchard_env = OrchardEnv(
        scene=RecordScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    )

    env_cfg = orchard_env.to_isaac_env_cfg()

    assert env_cfg.records is not None
    assert env_cfg.records.file_path == "logs/records"
    assert isinstance(env_cfg.records.controller, NoOpRecordControllerCfg)


def test_orchard_env_configure_recording_uses_episode_controller_by_default():
    orchard_env = OrchardEnv(
        scene=RecordScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    ).configure_recording()

    env_cfg = orchard_env.to_isaac_env_cfg()

    assert env_cfg.records is not None
    assert env_cfg.records.file_path == "logs/records"
    assert isinstance(env_cfg.records.controller, EpisodeRecordControllerCfg)


def test_orchard_env_disable_recording_restores_noop_controller(tmp_path):
    orchard_env = OrchardEnv(
        scene=RecordScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    ).configure_recording(
        file_path=str(tmp_path),
        controller=EpisodeRecordControllerCfg(),
    )
    orchard_env.disable_recording()

    env_cfg = orchard_env.to_isaac_env_cfg()

    assert env_cfg.records is not None
    assert env_cfg.records.file_path == str(tmp_path)
    assert isinstance(env_cfg.records.controller, NoOpRecordControllerCfg)


def test_place_a2b_task_definition_builds_record_cfg_when_enabled(tmp_path):
    # PlaceA2BTaskDefinition.build() requires an AssetResolver, which in
    # turn needs a real asset library. This test only exercises the
    # recording cfg path, so compose the env manually with fake asset
    # specs (same scene + embodiment resolution that build() would use).
    orchard_env = OrchardEnv(
        scene=PlaceA2BTaskDefinition.resolve_scene(),
        embodiment=PlaceA2BTaskDefinition.resolve_embodiment(),
        task=_make_place_a2b_task(),
    ).configure_recording(
        file_path=str(tmp_path),
        controller=StationaryEpisodeRecordControllerCfg(),
    )

    env_cfg = orchard_env.to_isaac_env_cfg()

    assert env_cfg.records is not None
    assert env_cfg.records.file_path == str(tmp_path)
    assert isinstance(
        env_cfg.records.controller,
        StationaryEpisodeRecordControllerCfg,
    )
    assert "static_camera_rgb" in env_cfg.records.terms
    assert "left_hand_camera_rgb" in env_cfg.records.terms
    assert "right_hand_camera_rgb" in env_cfg.records.terms
    assert "vis_camera_rgb" in env_cfg.records.terms


def test_place_a2b_task_rejects_assets_missing_a_declared_role():
    with pytest.raises(ValueError, match="place"):
        PlaceA2BTask(
            assets=TaskAssets(
                role_candidates={
                    "pick": [
                        RigidObjectSpec(
                            name="pick_object",
                            usd_path="/tmp/pick.usd",
                        )
                    ],
                }
            )
        )


def test_task_assets_flatten_is_keyed_by_scene_name():
    assets = _place_a2b_assets().with_default_namespace("objects")

    assert set(assets.flatten()) == {
        "objects/pick_object",
        "objects/place_object",
    }


def test_dualarm_piper_embodiment_get_action_cfg_returns_arm_gripper_terms():
    embodiment = DualArmPiperEmbodiment()

    action_cfg = embodiment.get_action_cfg()

    assert set(action_cfg.terms) == {
        "left_robot_joint_position",
        "left_robot_gripper_control",
        "right_robot_joint_position",
        "right_robot_gripper_control",
    }


def test_dualarm_piper_embodiment_robot_cfg_preserves_joint_defaults():
    embodiment = DualArmPiperEmbodiment(enable_cameras=False)

    robot_cfg = embodiment.get_assets_cfg()["robots"]["dualarm_piper"]

    assert robot_cfg.init_state.joint_pos["left_joint7"] == 0.05
    assert robot_cfg.init_state.joint_pos["left_joint8"] == -0.05
    assert robot_cfg.init_state.joint_pos["right_joint7"] == 0.05
    assert robot_cfg.init_state.joint_pos["right_joint8"] == -0.05


def test_dualarm_piper_embodiment_observation_cfg_returns_robot_tf_groups():
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


def test_dualarm_piper_camera_enabled_tf_obs_includes_camera_tf_terms():
    embodiment = DualArmPiperEmbodiment(enable_cameras=True)

    camera_names = set(embodiment.get_assets_cfg()["cameras"])
    tf_terms = set(embodiment.get_observation_cfg().groups["/tf"].terms)

    assert {f"{camera_name}_tf" for camera_name in camera_names} <= tf_terms


def test_dualarm_piper_camera_enabled_camera_obs_includes_camera_terms():
    embodiment = DualArmPiperEmbodiment(enable_cameras=True)

    camera_names = set(embodiment.get_assets_cfg()["cameras"])
    camera_obs_terms = set(
        embodiment.get_observation_cfg().groups["/camera"].terms
    )

    assert {
        f"{camera_name}_term" for camera_name in camera_names
    } <= camera_obs_terms


def test_dualarm_piper_camera_enabled_record_includes_camera_tf_terms():
    embodiment = DualArmPiperEmbodiment(enable_cameras=True)

    camera_names = set(embodiment.get_assets_cfg()["cameras"])
    record_terms = set(embodiment.get_record_terms())

    assert {
        f"{camera_name}_tf" for camera_name in camera_names
    } <= record_terms


def test_dualarm_piper_camera_tf_record_terms_use_runtime_frame_names():
    embodiment = DualArmPiperEmbodiment(enable_cameras=True)

    record_terms = embodiment.get_record_terms()

    for term_name in (
        "static_camera_tf",
        "left_hand_camera_tf",
        "right_hand_camera_tf",
        "vis_camera_tf",
    ):
        term_cfg = record_terms[term_name]
        assert isinstance(term_cfg, McapTFTermCfg)
        assert term_cfg.parent_frame is None
        assert term_cfg.child_frame is None


def test_dualarm_piper_camera_image_record_terms_use_runtime_frame_names():
    embodiment = DualArmPiperEmbodiment(enable_cameras=True)

    record_terms = embodiment.get_record_terms()

    expected_frame_ids = {
        "static_camera_rgb": "cameras/static_camera",
        "left_hand_camera_rgb": "cameras/left_hand_camera",
        "right_hand_camera_rgb": "cameras/right_hand_camera",
        "vis_camera_rgb": "cameras/vis_camera",
    }

    for term_name, expected_frame_id in expected_frame_ids.items():
        term_cfg = record_terms[term_name]
        assert isinstance(term_cfg, McapImageTermCfg)
        assert term_cfg.frame_id == expected_frame_id


def test_dualarm_piper_image_record_terms_cover_rgb_depth_calibration():
    embodiment = DualArmPiperEmbodiment(enable_cameras=True)

    record_terms = embodiment.get_record_terms()

    expected_terms = {
        "static_camera_rgb": (
            "/observation/cameras/static_camera/color_image/image_raw",
            "cameras/static_camera",
            "rgb",
        ),
        "static_camera_depth": (
            "/observation/cameras/static_camera/depth_image/image_raw",
            "cameras/static_camera",
            "depth",
        ),
        "static_camera_color_calib": (
            "/observation/cameras/static_camera/color_image/camera_info",
            "cameras/static_camera",
            "calibration",
        ),
        "static_camera_depth_calib": (
            "/observation/cameras/static_camera/depth_image/camera_info",
            "cameras/static_camera",
            "calibration",
        ),
        "left_hand_camera_rgb": (
            "/observation/cameras/left_hand_camera/color_image/image_raw",
            "cameras/left_hand_camera",
            "rgb",
        ),
        "left_hand_camera_depth": (
            "/observation/cameras/left_hand_camera/depth_image/image_raw",
            "cameras/left_hand_camera",
            "depth",
        ),
        "left_hand_camera_color_calib": (
            "/observation/cameras/left_hand_camera/color_image/camera_info",
            "cameras/left_hand_camera",
            "calibration",
        ),
        "left_hand_camera_depth_calib": (
            "/observation/cameras/left_hand_camera/depth_image/camera_info",
            "cameras/left_hand_camera",
            "calibration",
        ),
        "right_hand_camera_rgb": (
            "/observation/cameras/right_hand_camera/color_image/image_raw",
            "cameras/right_hand_camera",
            "rgb",
        ),
        "right_hand_camera_depth": (
            "/observation/cameras/right_hand_camera/depth_image/image_raw",
            "cameras/right_hand_camera",
            "depth",
        ),
        "right_hand_camera_color_calib": (
            "/observation/cameras/right_hand_camera/color_image/camera_info",
            "cameras/right_hand_camera",
            "calibration",
        ),
        "right_hand_camera_depth_calib": (
            "/observation/cameras/right_hand_camera/depth_image/camera_info",
            "cameras/right_hand_camera",
            "calibration",
        ),
        "vis_camera_rgb": (
            "/observation/cameras/vis_camera/color_image/image_raw",
            "cameras/vis_camera",
            "rgb",
        ),
        "vis_camera_depth": (
            "/observation/cameras/vis_camera/depth_image/image_raw",
            "cameras/vis_camera",
            "depth",
        ),
        "vis_camera_color_calib": (
            "/observation/cameras/vis_camera/color_image/camera_info",
            "cameras/vis_camera",
            "calibration",
        ),
        "vis_camera_depth_calib": (
            "/observation/cameras/vis_camera/depth_image/camera_info",
            "cameras/vis_camera",
            "calibration",
        ),
    }

    for term_name, (
        expected_topic,
        expected_frame_id,
        expected_mode,
    ) in expected_terms.items():
        term_cfg = record_terms[term_name]
        assert isinstance(term_cfg, McapImageTermCfg)
        assert term_cfg.topic == expected_topic
        assert term_cfg.frame_id == expected_frame_id
        assert term_cfg.mode == expected_mode


# ----- layout_builder integration -----------------------------------------


def _layout_builder_with_one_role(role: str = "pick"):
    """Build a LayoutBuilder containing one episode for ``role``."""
    from robo_orchard_sim.orchard_env.layout.builder import LayoutBuilder
    from robo_orchard_sim.orchard_env.layout.loader import (
        Layout,
        LayoutObject,
        LayoutSequence,
    )

    layout = Layout(
        objects={
            role: LayoutObject(
                category="apple",
                position=(0.0, 0.0, 0.0),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
        },
        raw={},
    )
    return LayoutBuilder(
        layouts=LayoutSequence(entries=[layout], raw=[]),
        scene_name_by_role={role: "objects/pick"},
    )


def test_env_builder_with_layout_builder_injects_layout_reset():
    """Layout-mode env_cfg.events contains the layout_reset term."""
    builder = _layout_builder_with_one_role()
    env_cfg = EnvBuilder(
        scene=DummyScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
        layout_builder=builder,
    ).build()

    assert "layout_reset" in env_cfg.events.terms


def test_env_builder_without_layout_builder_uses_task_event_cfg():
    """Sampler-mode env_cfg.events comes from the task (no layout_reset)."""
    env_cfg = EnvBuilder(
        scene=DummyScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    ).build()

    assert "layout_reset" not in env_cfg.events.terms


def test_layout_builder_apply_to_drops_pose_reset_keeps_others():
    """Pose-reset terms are shadowed by layout; other terms survive."""
    from unittest.mock import MagicMock

    from robo_orchard_sim.ext.envs.managers.events.pose_reset import (
        PoseResetTermCfg,
    )

    builder = _layout_builder_with_one_role()
    pose_term = MagicMock(spec=PoseResetTermCfg)
    light_term = MagicMock()
    task_event_cfg = EventManagerCfg(
        terms={
            "random_pose_event": pose_term,
            "light_randomization": light_term,
        }
    )

    merged = builder.apply_to(task_event_cfg)

    assert set(merged.terms) == {"light_randomization", "layout_reset"}


def test_orchard_env_num_episodes_with_layout_builder():
    """OrchardEnv.num_episodes reads layout_builder.num_episodes."""
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

    env = OrchardEnv(
        scene=DummyScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
        layout_builder=_layout_builder_with_one_role(),
    )
    assert env.num_episodes == 1


def test_orchard_env_num_episodes_without_layout_builder_is_none():
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

    env = OrchardEnv(
        scene=DummyScene(),
        embodiment=DummyEmbodiment(),
        task=DummyTask(),
    )
    assert env.num_episodes is None
