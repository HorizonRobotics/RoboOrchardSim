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

from robo_orchard_sim.tasks.trajs_gen.base_executor import ObjectInfo
from robo_orchard_sim.tasks.trajs_gen.executors import (
    BackToDefaultExecutorCfg,
    PickExecutorCfg,
    PlaceExecutorCfg,
)
from robo_orchard_sim.tasks.trajs_gen.manipulator_resolver import (
    BoundManipulatorResolver,
    PredicateManipulatorResolver,
)
from robo_orchard_sim.tasks.trajs_gen.pose_generator import (
    MoveByJointOffsetCfg,
)

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.tasks.trajs_gen.base_executor import BaseExecutorCfg


def build_task_atomic_action_plan(
    orchard_env: "OrchardEnv",
) -> list[BaseExecutorCfg]:
    """Build the default atomic action plan for place-a2b."""
    task = cast(Any, orchard_env.task)
    pick_obj = task.pick_object.scene_name
    place_obj = task.place_object.scene_name
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
                joint_offsets=[-0.15],
            ),
            constrain="free",
            priority=0,
        ),
        BackToDefaultExecutorCfg(
            robot_info=arm,
            priority=0,
        ),
    ]
