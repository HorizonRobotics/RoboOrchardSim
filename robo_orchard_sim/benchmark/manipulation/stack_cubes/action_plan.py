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

"""Default atomic action plans for the stack-cubes task."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, cast

from robo_orchard_sim.task_components.trajs_gen.base_executor import (
    ObjectInfo,
)
from robo_orchard_sim.task_components.trajs_gen.executors import (
    BackToDefaultExecutorCfg,
    MoveExecutorCfg,
    PickExecutorCfg,
    PlaceExecutorCfg,
)
from robo_orchard_sim.task_components.trajs_gen.pose_generator import (
    MoveByDisplacementCfg,
)

if TYPE_CHECKING:
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
    from robo_orchard_sim.task_components.role_registry import RoleRegistry
    from robo_orchard_sim.task_components.trajs_gen.base_executor import (
        BaseExecutorCfg,
    )

_PANDA_ROBOT_NAMES = {"franka_panda", "panda_droid"}
_MEMBERS_ROLE = "members"

# Vertical clearance for approach, lift, and retreat. Matches the value
# the place-a2b panda plan uses.
_CLEARANCE_M = 0.15

# How far a grasp may tilt off vertical. This is about the shape of the
# demonstration, not whether the stack stands up: the synthesized
# trajectories are what a policy imitates, so the wrist should approach
# the way a person would rather than reach in sideways. The executor
# default of 45 degrees produces visibly contorted approaches here.
_MAX_GRASP_TILT_DEG = 10.0


def build_task_atomic_action_plan(
    orchard_env: "OrchardEnv",
    *,
    role_registry: "RoleRegistry | None" = None,
) -> list[BaseExecutorCfg]:
    """Build the default atomic action plan for stack-cubes."""
    robot_name = cast(Any, orchard_env.embodiment).name
    if robot_name in _PANDA_ROBOT_NAMES:
        return _build_franka_panda_action_plan(
            orchard_env,
            role_registry=role_registry,
        )

    robot_infos = orchard_env.embodiment.get_robot_info_cfgs()
    available = ", ".join(sorted(robot_infos))
    raise ValueError(
        "No stack_cubes action plan for robot "
        f"{robot_name!r}. Available manipulators: {available or '<none>'}."
    )


def _resolve_stack_order(
    role_registry: "RoleRegistry | None",
) -> list[str]:
    """Return member scene names ordered bottom to top for this episode."""
    if role_registry is None:
        raise ValueError(
            "Stack-cubes needs a role_registry before the action plan "
            "is built."
        )
    targets = role_registry.resolve_many(_MEMBERS_ROLE)
    return [target.scene_name for target in targets]


def _vertical_clearance() -> MoveByDisplacementCfg:
    """Return a straight-up displacement in the world frame."""
    return MoveByDisplacementCfg(
        distance=_CLEARANCE_M,
        direction="z",
        frame="world",
    )


def _build_franka_panda_action_plan(
    orchard_env: "OrchardEnv",
    *,
    role_registry: "RoleRegistry | None",
) -> list[BaseExecutorCfg]:
    """Build the Franka Panda stack-cubes action plan."""
    order = _resolve_stack_order(role_registry)
    arm = orchard_env.embodiment.get_robot_info_cfg("main_arm")

    plan: list[BaseExecutorCfg] = []
    for support, carried in zip(order, order[1:], strict=False):
        plan.append(
            PickExecutorCfg(
                robot_info=arm,
                pick_object_info=ObjectInfo(
                    name=carried,
                    mode="passive",
                    action="pick",
                    part="body",
                ),
                pre_grasp=_vertical_clearance(),
                grasp_mode="Top-down",
                top_down_max_tilt_deg=_MAX_GRASP_TILT_DEG,
                priority=0,
            )
        )
        plan.append(
            MoveExecutorCfg(
                robot_info=arm,
                target=_vertical_clearance(),
                gripper_state="CLOSED",
                priority=0,
            )
        )
        plan.append(
            PlaceExecutorCfg(
                robot_info=arm,
                pick_object_info=ObjectInfo(
                    name=carried,
                    mode="active",
                    action="place",
                    part="body",
                ),
                place_object_info=ObjectInfo(
                    name=support,
                    mode="passive",
                    action="place",
                    part="body",
                ),
                pre_place_cfg=_vertical_clearance(),
                constrain="align_dir_axis",
                candidate_selection="joint_distance",
                axis_rotation_candidates=4,
                priority=0,
            )
        )
        plan.append(
            MoveExecutorCfg(
                robot_info=arm,
                target=_vertical_clearance(),
                priority=0,
            )
        )
    plan.append(
        BackToDefaultExecutorCfg(
            robot_info=arm,
            priority=0,
        )
    )
    return plan


__all__ = ["build_task_atomic_action_plan"]
