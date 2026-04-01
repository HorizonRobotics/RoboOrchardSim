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

from robo_orchard_sim.cfg_wrappers.actuators_cfg import ImplicitActuatorCfg
from robo_orchard_sim.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.cfg_wrappers.sim.converters.urdf_converter_cfg import (
    UrdfConverterCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners import (
    UrdfFileCfg,
    UsdFileCfg,
)
from robo_orchard_sim.models.assets.asset_cfg import ORCHARD_ASSET

__all__ = ["DUALARM_PIPER_CFG"]

urdf_spawn_cfg = UrdfFileCfg(
    asset_path=f"{ORCHARD_ASSET}/ROBOTS/dualarm_piper/piper_description_dualarm_new_textured_large_stroke.urdf",
    usd_dir="/tmp/IsaacLab/piper/usd",
    usd_file_name="piper_description_dualarm_new_textured_large_stroke",
    fix_base=True,
    merge_fixed_joints=False,
    collider_type="convex_decomposition",
    semantic_tags=[("class", "dualarm_piper")],
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

usd_spawn_cfg = UsdFileCfg(
    usd_path=f"{ORCHARD_ASSET}/ROBOTS/dualarm_piper/usd/piper_description_dualarm_new_textured_large_stroke.usd",
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
    semantic_tags=[("class", "dualarm_piper")],
)

DUALARM_PIPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/dualarm_piper",
    spawn=usd_spawn_cfg,
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fl_joint1": 0.0,
            "fl_joint2": 0.0,
            "fl_joint3": 0.0,
            "fl_joint4": 0.0,
            "fl_joint5": 0.0,
            "fl_joint6": 0.0,
            "fl_joint7": 0.4,
            "fl_joint8": -0.4,
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
