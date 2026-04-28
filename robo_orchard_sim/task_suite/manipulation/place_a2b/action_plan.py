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
from robo_orchard_sim.tasks.trajs_gen.executors.pick import PickExecutorCfg

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.tasks.trajs_gen.base_executor import BaseExecutorCfg


def build_task_atomic_action_plan(
    orchard_env: "OrchardEnv",
) -> list[BaseExecutorCfg]:
    """Build the default atomic action plan for place-a2b."""
    task = cast(Any, orchard_env.task)
    pick_obj = task.pick_object.scene_name
    left_arm = orchard_env.embodiment.get_robot_info_cfg("left_arm")
    right_arm = orchard_env.embodiment.get_robot_info_cfg("right_arm")

    return [
        PickExecutorCfg(
            robot_info=left_arm,
            pick_object_info=ObjectInfo(
                name=pick_obj, mode="active", action="pick", part="gripper"
            ),
            priority=1,
        ),
        PickExecutorCfg(
            robot_info=right_arm,
            pick_object_info=ObjectInfo(
                name=pick_obj, mode="active", action="pick", part="gripper"
            ),
            priority=2,
        ),
        # MoveExecutorCfg(
        #     robot_info=left_arm,
        #     target_pose=[0.38, -0.25, 0.25, 0.265, 0.092, 0.963, 0.0374],
        #     priority=0,
        # ),
        # MoveExecutorCfg(
        #     robot_info=right_arm,
        #     target_pose=[0.38, 0.25, 0.25, 0.245, -0.097, 0.963, -0.044],
        #     priority=0,
        # ),
    ]
