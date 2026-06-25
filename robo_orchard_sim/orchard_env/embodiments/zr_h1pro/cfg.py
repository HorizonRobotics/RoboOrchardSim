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

from robo_orchard_sim.ext.cfg_wrappers.actuators_cfg import ImplicitActuatorCfg
from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.converters import (
    urdf_converter_cfg as _urdf_converter_cfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import (
    UrdfFileCfg,
    UsdFileCfg,
)
from robo_orchard_sim.ext.models.assets.asset_cfg import ORCHARD_ASSET

UrdfConverterCfg = _urdf_converter_cfg.UrdfConverterCfg

__all__ = ["ZR_H1PRO_CFG"]

urdf_spawn_cfg = UrdfFileCfg(
    asset_path=f"{ORCHARD_ASSET}/ROBOTS/ZR_H1PRO/ZR_H1PRO-1.1.03.H.2025.08.15_URDF_V1/ZR_H1_V3.0_25.08.17/urdf/ZR_H1_V3.0_25.08.17.urdf",
    usd_dir="/tmp/IsaacLab/zr_h1pro/usd",
    usd_file_name="ZR_H1_V3.0_25.08.17",
    fix_base=False,
    merge_fixed_joints=False,
    collider_type="convex_decomposition",
    semantic_tags=[("class", "h1_pro")],
    activate_contact_sensors=False,
    rigid_props=RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        fix_root_link=False,
    ),
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
            natural_frequency=30.0
        )
    ),
)

usd_spawn_cfg = UsdFileCfg(
    usd_path=f"{ORCHARD_ASSET}/ROBOTS/ZR_H1PRO/ZR_H1PRO-1.1.03.H.2025.08.15_URDF_V1/ZR_H1_V3.0_25.08.17/usd/Collected_ZERITH_robot0211/ZERITH_robot0224.usd",
    activate_contact_sensors=False,
    rigid_props=RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    ),
    articulation_props=ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        fix_root_link=False,
    ),
    semantic_tags=[("class", "h1_pro")],
)

ZR_H1PRO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/zr_h1pro",
    spawn=usd_spawn_cfg,
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
    actuators=dict(
        body=ImplicitActuatorCfg(
            joint_names_expr=[
                "body_pitch_joint",
                "body_yaw_joint",
                "neck_yaw_joint",
                "neck_pitch_joint",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=2000.0,
            damping=300.0,
        ),
        arm=ImplicitActuatorCfg(
            joint_names_expr=["left_.*joint", "right_.*joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=2000.0,
            damping=300.0,
        ),
        gripper=ImplicitActuatorCfg(
            joint_names_expr=[
                "left_jaw_left_finger_joint",
                "left_jaw_right_finger_joint",
                "right_jaw_left_finger_joint",
                "right_jaw_right_finger_joint",
            ],
            effort_limit_sim=2000.0,
            velocity_limit_sim=0.25,
            stiffness=1e4,
            damping=1e2,
        ),
        daogui=ImplicitActuatorCfg(
            joint_names_expr=["daogui_joint"],
            effort_limit_sim=5000.0,
            velocity_limit_sim=2.0,
            stiffness=4e3,
            damping=1e3,
        ),
    ),
    soft_joint_pos_limit_factor=1.0,
)
