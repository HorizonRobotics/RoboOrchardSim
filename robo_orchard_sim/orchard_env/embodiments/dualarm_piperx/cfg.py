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

from robo_orchard_sim.ext.cfg_wrappers.actuators_cfg import ImplicitActuatorCfg
from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.converters import (
    urdf_converter_cfg as _urdf_converter_cfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UrdfFileCfg
from robo_orchard_sim.ext.models.assets.asset_cfg import ORCHARD_ASSET

UrdfConverterCfg = _urdf_converter_cfg.UrdfConverterCfg

__all__ = ["DUALARM_PIPERX_CFG"]

_PIPERX_ASSET_ROOT = f"{ORCHARD_ASSET}/ROBOTS/piper_x_description"
_PIPERX_URDF_NAME = "piper_x_description_dualarm_dark.urdf"

urdf_spawn_cfg = UrdfFileCfg(
    asset_path=f"{_PIPERX_ASSET_ROOT}/{_PIPERX_URDF_NAME}",
    usd_dir="/tmp/IsaacLab/piperx/usd",
    usd_file_name="piper_x_description_dualarm_dark",
    fix_base=True,
    merge_fixed_joints=False,
    collider_type="convex_decomposition",
    semantic_tags=[("class", "dualarm_piperx")],
    activate_contact_sensors=False,
    rigid_props=RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
        fix_root_link=True,
    ),
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
            natural_frequency=30.0
        )
    ),
)

DUALARM_PIPERX_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/dualarm_piperx",
    spawn=urdf_spawn_cfg,
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_joint1": -0.319669,
            "left_joint2": 0.898663,
            "left_joint3": -1.405843,
            "left_joint4": 1.392390,
            "left_joint5": 0.030785,
            "left_joint6": 0.05,
            "left_joint7": 0.05,
            "left_joint8": -0.05,
            "right_joint1": 0.319669,
            "right_joint2": 0.898663,
            "right_joint3": -1.405843,
            "right_joint4": 1.392390,
            "right_joint5": 0.030785,
            "right_joint6": 0.05,
            "right_joint7": 0.05,
            "right_joint8": -0.05,
        },
    ),
    actuators=dict(
        arm=ImplicitActuatorCfg(
            joint_names_expr=["left_joint[1-6]", "right_joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=2000.0,
            damping=300.0,
        ),
        gripper=ImplicitActuatorCfg(
            joint_names_expr=[
                "left_joint7",
                "left_joint8",
                "right_joint7",
                "right_joint8",
            ],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.25,
            stiffness=2e3,
            damping=1e2,
        ),
    ),
    soft_joint_pos_limit_factor=1.0,
)
