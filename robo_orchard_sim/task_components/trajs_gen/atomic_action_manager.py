# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

"""Public interfaces for the atomic action manager.

This file only defines the external contract for configuration, action
registration, and runtime I/O. Detailed planning and execution logic will be
implemented later behind the same interfaces.
"""

from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import torch
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType_co,
)
from typing_extensions import Literal, TypeAlias

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    ResolvedManipulatorProfile,
)
from robo_orchard_sim.task_components.trajs_gen.base_executor import (
    BaseExecutor,
    BaseExecutorCfg,
    DebugTargetPose,
)
from robo_orchard_sim.task_components.trajs_gen.manipulator_resolver import (
    ManipulatorBindingContext,
)

AtomicActionLifecycle = Literal[
    "PENDING",
    "RUNNING",
    "COMPLETED",
    "FAILED",
]

AtomicActionOutput: TypeAlias = dict[str, UnifiedJointCommand]
"""Atomic action output keyed by robot name.

Each value is a :class:`UnifiedJointCommand` that merges all active
manipulator actions belonging to the same robot.
"""


@dataclass
class AtomicActionStatus:
    """Runtime status of one registered atomic action."""

    action_type: str
    effector: str
    status: AtomicActionLifecycle
    priority: int
    success: bool | None = None


@dataclass
class AtomicActionManagerState:
    """Observable state returned by :meth:`AtomicActionManager.step`.

    Attributes:
        current_priority: Lowest priority among unfinished actions.
        running_actions: Runtime status keyed by
            ``"{robot_name}/{manipulator_name}"``.
        pending_count: Number of actions not yet completed.
        env_busy: Per-env busy flag with shape ``[num_envs]``.
    """

    current_priority: int | None
    running_actions: dict[str, AtomicActionStatus]
    pending_count: int
    env_busy: torch.Tensor


class ActionStatusLogger:
    """Throttle status logging for atomic actions."""

    def __init__(self, running_interval: int = 50) -> None:
        self.running_interval = running_interval
        self._last_status_by_action_key: dict[str, AtomicActionLifecycle] = {}

    def collect(
        self,
        *,
        running_actions: dict[str, AtomicActionStatus],
        step_idx: int,
    ) -> list[str]:
        """Return formatted status lines for the current step."""
        log_lines: list[str] = []
        for action_key, action_status in sorted(running_actions.items()):
            status = action_status.status
            previous_status = self._last_status_by_action_key.get(action_key)
            if status == "RUNNING":
                if step_idx % self.running_interval != 0:
                    continue
            elif previous_status == status:
                continue

            log_lines.append(
                f"[{action_key}]:{action_status.action_type}---{status}"
            )

        self._last_status_by_action_key = {
            action_key: action_status.status
            for action_key, action_status in running_actions.items()
        }
        return log_lines


@dataclass
class _RegisteredAction:
    executor: BaseExecutor
    priority: int
    action_type: str
    status: AtomicActionLifecycle = "PENDING"
    success: bool | None = None
    manipulator_name: str | None = None
    action_key: str | None = None


class _TrajectoryPlayer:
    """Step batched per-env trajectories one control tick at a time."""

    def __init__(
        self,
        trajectories: list[torch.Tensor],
        num_envs: int,
        device: torch.device,
    ) -> None:
        if len(trajectories) != num_envs:
            raise ValueError(
                "Atomic action executor must return one trajectory per env: "
                f"expected {num_envs}, got {len(trajectories)}."
            )
        self._device = device
        self._queues = [
            deque(self._split_steps(trajectory.to(device=device)))
            for trajectory in trajectories
        ]
        self._last = [
            torch.zeros_like(queue[0]) if queue else torch.empty(0)
            for queue in self._queues
        ]

    @property
    def is_complete(self) -> bool:
        return all(len(queue) == 0 for queue in self._queues)

    def next_action(self) -> tuple[torch.Tensor | None, torch.Tensor]:
        env_busy = torch.tensor(
            [len(queue) > 0 for queue in self._queues],
            dtype=torch.bool,
            device=self._device,
        )
        if not bool(torch.any(env_busy).item()):
            return None, env_busy

        action = []
        for index, queue in enumerate(self._queues):
            if queue:
                item = queue.popleft()
                self._last[index] = item
            else:
                item = self._last[index]
            action.append(item)
        return torch.stack(action, dim=0), env_busy

    def _split_steps(self, trajectory: torch.Tensor) -> list[torch.Tensor]:
        if trajectory.dim() == 0:
            raise ValueError("Atomic action trajectory must not be scalar.")
        if trajectory.dim() == 1:
            return [trajectory]
        return [step for step in trajectory]


@dataclass
class _ActiveAction:
    registered: _RegisteredAction
    player: _TrajectoryPlayer
    action_key: str
    resolved: ResolvedManipulatorProfile


class AtomicActionManager:
    """Runtime manager for configured atomic actions.

    External usage:
        1. Construct the manager with ``AtomicActionManager(cfg)``.
        2. Register actions via :meth:`register` with a list of
           :class:`BaseExecutorCfg`.
        3. Call :meth:`get_action` every control tick.
    """

    cfg: "AtomicActionManagerCfg"

    def __init__(self, cfg: "AtomicActionManagerCfg"):
        self.cfg = cfg
        self._registered_actions: list[_RegisteredAction] = []
        self._active_actions: dict[str, _ActiveAction] = {}
        self._manipulator_context = ManipulatorBindingContext()
        self._debug_visualizer: Any | None = None

    def register(self, atomic_actions: list[BaseExecutorCfg]) -> None:
        for action_cfg in atomic_actions:
            executor = action_cfg()
            action_type = (
                action_cfg.action_type
                if action_cfg.action_type
                else self._infer_action_type(executor)
            )
            self._registered_actions.append(
                _RegisteredAction(
                    executor=executor,
                    priority=action_cfg.priority,
                    action_type=action_type,
                )
            )

    def clear(self, clear_planner_instances: bool = False) -> None:
        """Remove all registered actions and reset runtime state.

        Use this between task segments to discard completed actions
        before registering a new plan on the same manager instance.

        Args:
            clear_planner_instances: Whether to also discard cached planner
                instances. Defaults to ``False`` so planner setup can be reused
                across independent environments in a long-running process.
        """
        self._registered_actions.clear()
        self._active_actions.clear()
        self._manipulator_context.reset(
            clear_planner_instances=clear_planner_instances
        )
        if self._debug_visualizer is not None:
            self._debug_visualizer.clear()

    def reset_sequence(self) -> None:
        """Reset sequence-scoped manipulator binding decisions."""
        self._manipulator_context.reset()
        self._active_actions.clear()
        for registered in self._registered_actions:
            registered.status = "PENDING"
            registered.success = None
            registered.manipulator_name = None
            registered.action_key = None
            registered.executor.reset()

    def get_action(
        self,
        env: Any,
    ) -> tuple[AtomicActionOutput, AtomicActionManagerState]:
        """Advance one manager tick and return per-robot joint commands.

        Returns:
            tuple[AtomicActionOutput, AtomicActionManagerState]:
                - ``AtomicActionOutput``:
                  a ``dict[str, UnifiedJointCommand]`` keyed by robot
                  name.  All active manipulator actions for the same
                  robot are merged into one :class:`UnifiedJointCommand`.
                  The dict is empty when no manipulators are active.
                - ``AtomicActionManagerState``:
                  runtime state summary for debugging and orchestration.
        """
        num_envs: int = env.num_envs
        device: torch.device = env.device
        per_robot_actions: dict[str, list[UnifiedJointCommand]] = defaultdict(
            list
        )
        running_actions: dict[str, AtomicActionStatus] = {}
        env_busy = torch.zeros(num_envs, dtype=torch.bool, device=device)

        tick_priority = self._current_priority()
        if tick_priority is not None:
            self._start_ready_actions(env, priority=tick_priority)

        for manipulator_name, active in list(self._active_actions.items()):
            action, busy = active.player.next_action()
            env_busy |= busy
            if action is not None:
                resolved = active.resolved
                joint_names = (
                    resolved.joint_names + resolved.gripper_joint_names
                )
                per_robot_actions[resolved.robot_name].append(
                    UnifiedJointCommand(
                        values=action,
                        joint_names=joint_names,
                    )
                )

            registered = active.registered
            if active.player.is_complete:
                registered.status = "COMPLETED"
                registered.success = True
                del self._active_actions[manipulator_name]
            else:
                registered.status = "RUNNING"

            running_actions[active.action_key] = self._status_for(
                registered,
                manipulator_name=manipulator_name,
            )

        for registered in self._registered_actions:
            if registered.status != "FAILED":
                continue
            manipulator_name = registered.manipulator_name
            action_key = registered.action_key
            if (
                manipulator_name is None
                or action_key is None
                or action_key in running_actions
            ):
                continue
            running_actions[action_key] = self._status_for(
                registered,
                manipulator_name=manipulator_name,
            )

        output: AtomicActionOutput = {
            robot_name: UnifiedJointCommand.merge(*actions)
            for robot_name, actions in per_robot_actions.items()
        }

        status = AtomicActionManagerState(
            current_priority=tick_priority,
            running_actions=running_actions,
            pending_count=self._pending_count(),
            env_busy=env_busy,
        )
        return output, status

    def _start_ready_actions(self, env: Any, priority: int) -> None:
        for registered in self._registered_actions:
            if registered.status != "PENDING":
                continue
            if registered.priority != priority:
                continue

            resolved = registered.executor.cfg.resolve_manipulator_info(
                env,
                context=self._manipulator_context,
            )
            if resolved.manipulator_name in self._active_actions:
                continue

            trajectories = registered.executor.plan(
                env,
                context=self._manipulator_context,
            )
            resolved = trajectories.resolved_manipulator
            if self.cfg.debug_vis:
                self._visualize_debug_target_poses(
                    resolved=resolved,
                    target_poses=trajectories.debug_target_poses,
                )
            manipulator_name = resolved.manipulator_name
            action_key = f"{resolved.robot_name}/{resolved.manipulator_name}"
            registered.manipulator_name = manipulator_name
            registered.action_key = action_key
            registered.success = trajectories.success
            if not trajectories.success:
                registered.status = "FAILED"
                continue

            if manipulator_name in self._active_actions:
                continue

            registered.status = "RUNNING"
            self._active_actions[manipulator_name] = _ActiveAction(
                registered=registered,
                player=_TrajectoryPlayer(
                    trajectories.trajectories,
                    num_envs=env.num_envs,
                    device=env.device,
                ),
                action_key=action_key,
                resolved=resolved,
            )

    def _infer_action_type(self, executor: BaseExecutor) -> str:
        class_name = type(executor).__name__
        if class_name.endswith("Executor"):
            class_name = class_name[: -len("Executor")]
        return class_name.lower()

    def _current_priority(self) -> int | None:
        unfinished = [
            registered.priority
            for registered in self._registered_actions
            if registered.status in ("PENDING", "RUNNING")
        ]
        if not unfinished:
            return None
        return min(unfinished)

    def _pending_count(self) -> int:
        return sum(
            registered.status in ("PENDING", "RUNNING")
            for registered in self._registered_actions
        )

    def _status_for(
        self,
        registered: _RegisteredAction,
        manipulator_name: str,
    ) -> AtomicActionStatus:
        return AtomicActionStatus(
            action_type=registered.action_type,
            effector=manipulator_name,
            status=registered.status,
            priority=registered.priority,
            success=registered.success,
        )

    def _get_debug_visualizer(self) -> Any:
        if self._debug_visualizer is not None:
            return self._debug_visualizer

        from robo_orchard_sim.task_components.trajs_gen.debug_vis import (
            AtomicActionDebugVisualizer,
        )

        self._debug_visualizer = AtomicActionDebugVisualizer(
            marker_scale=self.cfg.debug_vis_marker_scale,
        )
        return self._debug_visualizer

    def _visualize_debug_target_poses(
        self,
        *,
        resolved: ResolvedManipulatorProfile,
        target_poses: tuple[DebugTargetPose, ...],
    ) -> None:
        visualizer = self._get_debug_visualizer()
        for target_pose in target_poses:
            visualizer.visualize_pose(
                marker_name=self._debug_marker_name(
                    resolved=resolved,
                    target_name=target_pose.name,
                ),
                pose_w=target_pose.pose_w,
            )

    def _debug_marker_name(
        self,
        *,
        resolved: ResolvedManipulatorProfile,
        target_name: str,
    ) -> str:
        return "_".join(
            (
                resolved.robot_name,
                resolved.manipulator_name,
                target_name,
            )
        ).replace("/", "_")


class AtomicActionManagerCfg(ClassConfig):
    """Configuration for :class:`AtomicActionManager`."""

    class_type: ClassType_co[AtomicActionManager] = AtomicActionManager
    debug_vis: bool = False
    debug_vis_marker_scale: tuple[float, float, float] = (0.1, 0.1, 0.1)

    def __call__(self, **kwargs: Any) -> AtomicActionManager:
        return self.class_type(self, **kwargs)
