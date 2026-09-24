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

"""Base class for robot embodiment providers."""

from __future__ import annotations
from collections.abc import Mapping

import torch
from robo_orchard_core.envs.managers.actions.action_manager import (
    ActionManagerCfg,
)
from robo_orchard_core.envs.managers.events import EventManagerCfg
from robo_orchard_core.envs.managers.observations.observation_manager import (
    ObservationManagerCfg,
)

from robo_orchard_sim.contracts.joint_command import (
    UnifiedJointCommand,
)
from robo_orchard_sim.contracts.policy_binding import PolicyBindingSchema
from robo_orchard_sim.ext.envs.managers.actions.articulation import (
    joint_base as _joint_base,
)
from robo_orchard_sim.ext.envs.managers.record import RecordTermBaseCfg
from robo_orchard_sim.ext.models.assets.asset_cfg import GroupAssetCfg
from robo_orchard_sim.orchard_env.assets import ArticulationSpec
from robo_orchard_sim.orchard_env.embodiments.embodiment_profile import (
    RobotInfoCfg,
)
from robo_orchard_sim.task_components.validators.contact_binding import (
    GRIPPER_CONTACT_NAMESPACE,
    GRIPPER_CONTACT_SENSOR_PREFIX,
)

ArticulationJointActionTermCfg = _joint_base.ArticulationJointActionTermCfg


class EmbodimentBase:
    """Abstract base for robot embodiment configuration providers."""

    def __init__(self, robot: ArticulationSpec):
        self.robot = robot.with_default_namespace("robots")

    @property
    def name(self) -> str:
        """Return the embodiment robot name."""
        return self.robot.name

    @property
    def namespace(self) -> str | None:
        """Return the embodiment robot namespace."""
        return self.robot.namespace

    @property
    def scene_name(self) -> str:
        """Return the scene-unique robot reference."""
        return self.robot.scene_name

    def get_assets_cfg(self) -> dict[str, GroupAssetCfg]:
        """Return embodiment-owned assets grouped by namespace."""
        namespace = self.robot.namespace
        assert namespace is not None
        assets = {
            namespace: GroupAssetCfg(
                **{
                    self.robot.name: self.robot.to_isaac_cfg(),
                }
            )
        }
        contact_sensors = self.build_gripper_contact_assets()
        if contact_sensors is not None:
            assets[GRIPPER_CONTACT_NAMESPACE] = contact_sensors
        return assets

    def get_observation_cfg(self) -> ObservationManagerCfg:
        """Return embodiment observation cfg fragment."""
        return ObservationManagerCfg(groups={})

    def get_action_cfg(self) -> ActionManagerCfg:
        """Return embodiment action cfg fragment."""
        return ActionManagerCfg(terms={})

    def translate_joint_command_to_env_action(
        self,
        action: UnifiedJointCommand,
    ) -> dict[str, torch.Tensor]:
        """Translate canonical joint command into action-manager inputs.

        The default implementation derives term names and joint specs from
        subclass-provided articulation joint action configs.
        """
        return self._translate_joint_command_by_action_cfg(
            action=action,
            action_cfg=self.get_action_cfg(),
        )

    def _translate_joint_command_by_action_cfg(
        self,
        action: UnifiedJointCommand,
        action_cfg: ActionManagerCfg,
    ) -> dict[str, torch.Tensor]:
        """Translate a joint command using articulation joint action terms."""
        translated: dict[str, torch.Tensor] = {}
        for term_name, term_cfg in action_cfg.terms.items():
            if not isinstance(term_cfg, ArticulationJointActionTermCfg):
                continue
            joint_specs = term_cfg.asset_cfg.joint_names
            if not joint_specs:
                continue
            wildcard_specs = [spec for spec in joint_specs if "*" in spec]
            if wildcard_specs:
                raise ValueError(
                    "Wildcard joint specs are not supported by default "
                    "joint-command translation. Use explicit joint names or "
                    f"range specs for action term '{term_name}'. Unsupported "
                    f"specs: {', '.join(wildcard_specs)}."
                )
            term_action = action.select_if_present(*joint_specs)
            if term_action is not None:
                translated[term_name] = term_action
        return translated

    def get_event_cfg(self) -> EventManagerCfg:
        """Return embodiment event cfg fragment."""
        return EventManagerCfg(terms={})

    def get_record_terms(self) -> Mapping[str, RecordTermBaseCfg]:
        """Return embodiment record term fragments."""
        return {}

    def get_robot_info_cfgs(self) -> Mapping[str, RobotInfoCfg]:
        """Return robot metadata keyed by manipulator name."""
        return {}

    def gripper_body_groups(self) -> tuple[tuple[str, ...], ...]:
        """Return unique per-manipulator finger-body groups."""
        groups: list[tuple[str, ...]] = []
        for robot_info in self.get_robot_info_cfgs().values():
            profile = robot_info.manipulator_profile
            if (
                profile is not None
                and profile.gripper_body_names
                and profile.gripper_body_names not in groups
            ):
                groups.append(profile.gripper_body_names)
        return tuple(groups)

    def build_gripper_contact_assets(self) -> GroupAssetCfg | None:
        """Build one filtered contact sensor for each configured finger."""
        from robo_orchard_sim.ext.models.sensors.contact_sensor import (
            ContactSensorCfg,
        )

        groups = self.gripper_body_groups()
        if not groups:
            return None
        sensors = {}
        for manipulator_idx, body_group in enumerate(groups):
            if len(body_group) != 2:
                raise ValueError(
                    "Opposing contact needs exactly two gripper bodies per "
                    f"manipulator, got {list(body_group)}."
                )
            for finger_idx, body_name in enumerate(body_group):
                other_body_name = body_group[1 - finger_idx]
                sensor_name = (
                    f"{GRIPPER_CONTACT_SENSOR_PREFIX}"
                    f"{manipulator_idx}_{finger_idx}"
                )
                sensors[sensor_name] = ContactSensorCfg(
                    track_contact_points=True,
                    prim_path=("{ENV_REGEX_NS}/" + f"{self.name}/{body_name}"),
                    filter_prim_paths_expr=[
                        "{ENV_REGEX_NS}/" + f"{self.name}/{other_body_name}"
                    ],
                )
        return GroupAssetCfg(**sensors)

    def get_robot_info_cfg(self, manipulator_name: str) -> RobotInfoCfg:
        """Fetch one robot-info config or raise a descriptive error."""
        robot_info_cfgs = self.get_robot_info_cfgs()
        if manipulator_name not in robot_info_cfgs:
            available = ", ".join(sorted(robot_info_cfgs))
            raise KeyError(
                f"Manipulator '{manipulator_name}' is not defined for "
                f"embodiment '{self.scene_name}'. Available manipulators: "
                f"{available or '<none>'}."
            )
        return robot_info_cfgs[manipulator_name]

    def get_policy_binding_schema(self) -> PolicyBindingSchema:
        """Return the canonical policy binding schema for this embodiment."""
        raise NotImplementedError(
            f"Embodiment {self.__class__.__name__} does not define a "
            "policy binding schema."
        )
