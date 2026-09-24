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

"""Default atomic action plans for place-a2b tasks."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast

from robo_orchard_sim.task_components.trajs_gen.base_executor import ObjectInfo
from robo_orchard_sim.task_components.trajs_gen.executors import (
    BackToDefaultExecutorCfg,
    MoveExecutorCfg,
    PickExecutorCfg,
    PlaceExecutorCfg,
)
from robo_orchard_sim.task_components.trajs_gen.manipulator_resolver import (
    BoundManipulatorResolver,
    PredicateManipulatorResolver,
)
from robo_orchard_sim.task_components.trajs_gen.pose_generator import (
    MoveByDisplacementCfg,
    MoveByJointOffsetCfg,
)

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.task_components.role_registry import RoleRegistry
    from robo_orchard_sim.task_components.trajs_gen.base_executor import (
        BaseExecutorCfg,
    )

_DUALARM_PIPER_ROBOT_NAMES = {"dualarm_piper", "dualarm_piperx"}
_PANDA_ROBOT_NAMES = {"franka_panda", "panda_droid"}


def build_task_atomic_action_plan(
    orchard_env: "OrchardEnv",
    *,
    role_registry: RoleRegistry | None = None,
) -> list[BaseExecutorCfg]:
    """Build the default atomic action plan for place-a2b."""
    task = cast(Any, orchard_env.task)
    pick_obj = (
        task.pick_object.scene_name
        if role_registry is None
        else role_registry.resolve_one("pick").scene_name
    )
    place_obj = (
        task.place_object.scene_name
        if role_registry is None
        else role_registry.resolve_one("place").scene_name
    )
    robot_name = cast(Any, orchard_env.embodiment).name
    if robot_name in _DUALARM_PIPER_ROBOT_NAMES:
        return _build_dualarm_piper_action_plan(
            orchard_env, pick_obj=pick_obj, place_obj=place_obj
        )
    if robot_name in _PANDA_ROBOT_NAMES:
        return _build_franka_panda_action_plan(
            orchard_env, pick_obj=pick_obj, place_obj=place_obj
        )

    robot_infos = orchard_env.embodiment.get_robot_info_cfgs()
    available = ", ".join(sorted(robot_infos))
    raise ValueError(
        "No place_a2b action plan for robot "
        f"{robot_name!r}. Available manipulators: {available or '<none>'}."
    )


def _build_dualarm_piper_action_plan(
    orchard_env: "OrchardEnv",
    *,
    pick_obj: str,
    place_obj: str,
) -> list[BaseExecutorCfg]:
    """Build the dual-arm Piper place-a2b action plan."""
    left_arm = orchard_env.embodiment.get_robot_info_cfg("left_arm")
    right_arm = orchard_env.embodiment.get_robot_info_cfg("right_arm")

    def pick_object_is_on_left(env: Any) -> bool:
        object_data = env.scene[pick_obj].data.root_pos_w
        return bool(object_data[0, 1].item() > 0.0)

    selector = PredicateManipulatorResolver(
        predicate=pick_object_is_on_left,
        true_robot_info=left_arm,
        false_robot_info=right_arm,
    )
    arm = BoundManipulatorResolver(binding_key="pick-move", selector=selector)

    return [
        PickExecutorCfg(
            robot_info=arm,
            pick_object_info=ObjectInfo(
                name=pick_obj, mode="passive", action="pick", part="body"
            ),
            pre_grasp=MoveByJointOffsetCfg(
                joint_id_idxs=[1],
                joint_offsets=[-0.15],
            ),
            grasp_mode="Top-down",
            priority=0,
        ),
        MoveExecutorCfg(
            robot_info=arm,
            target=MoveByJointOffsetCfg(
                joint_id_idxs=[1],
                joint_offsets=[-0.15],
            ),
            gripper_state="CLOSED",
            priority=0,
        ),
        PlaceExecutorCfg(
            robot_info=arm,
            pick_object_info=ObjectInfo(
                name=pick_obj, mode="active", action="place", part="body"
            ),
            place_object_info=ObjectInfo(
                name=place_obj, mode="passive", action="place", part="body"
            ),
            pre_place_cfg=MoveByJointOffsetCfg(
                joint_id_idxs=[1],
                joint_offsets=[-0.2],
            ),
            constrain="free",
            priority=0,
        ),
        MoveExecutorCfg(
            robot_info=arm,
            target=MoveByJointOffsetCfg(
                joint_id_idxs=[1],
                joint_offsets=[-0.15],
            ),
            priority=0,
        ),
        BackToDefaultExecutorCfg(
            robot_info=arm,
            priority=0,
        ),
    ]


def _build_franka_panda_action_plan(
    orchard_env: "OrchardEnv",
    *,
    pick_obj: str,
    place_obj: str,
) -> list[BaseExecutorCfg]:
    """Build the Franka Panda place-a2b action plan."""
    arm = orchard_env.embodiment.get_robot_info_cfg("main_arm")

    return [
        PickExecutorCfg(
            robot_info=arm,
            pick_object_info=ObjectInfo(
                name=pick_obj, mode="passive", action="pick", part="body"
            ),
            pre_grasp=MoveByDisplacementCfg(
                distance=0.15,
                direction="z",
                frame="world",
            ),
            grasp_mode="Top-down",
            priority=0,
        ),
        MoveExecutorCfg(
            robot_info=arm,
            target=MoveByDisplacementCfg(
                distance=0.15,
                direction="z",
                frame="world",
            ),
            gripper_state="CLOSED",
            priority=0,
        ),
        PlaceExecutorCfg(
            robot_info=arm,
            pick_object_info=ObjectInfo(
                name=pick_obj, mode="active", action="place", part="body"
            ),
            place_object_info=ObjectInfo(
                name=place_obj, mode="passive", action="place", part="body"
            ),
            pre_place_cfg=MoveByDisplacementCfg(
                distance=0.15,
                direction="z",
                frame="world",
            ),
            constrain="free",
            priority=0,
        ),
        MoveExecutorCfg(
            robot_info=arm,
            target=MoveByDisplacementCfg(
                distance=0.15,
                direction="z",
                frame="world",
            ),
            priority=0,
        ),
        BackToDefaultExecutorCfg(
            robot_info=arm,
            priority=0,
        ),
    ]
