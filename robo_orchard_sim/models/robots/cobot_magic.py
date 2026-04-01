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

__all__ = ["COBOT_MAGIC_CFG"]

# add friction to gripper
USD_SPAWN = UsdFileCfg(
    usd_path=f"{ORCHARD_ASSET}/ROBOTS/mobile_aloha_sim-2.0.0/aloha_new_description/usd/aloha_new_wo_material_color_friction.usd",
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
    semantic_tags=[("class", "cobot_magic")],
    # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0), # noqa: E501
)

# without firction
URDF_SPAWN = UrdfFileCfg(
    asset_path=f"{ORCHARD_ASSET}/ROBOTS/mobile_aloha_sim-2.0.0/aloha_new_description/urdf/aloha_new_wo_material_color.urdf",
    usd_dir="/tmp/IsaacLab/cobot_magic/usd",
    usd_file_name="aloha_new_wo_material_color.usd",
    fix_base=True,
    merge_fixed_joints=False,
    collider_type="convex_decomposition",
    semantic_tags=[("class", "cobot_magiic")],
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

COBOT_MAGIC_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/cobot_magic",
    spawn=USD_SPAWN,
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.0}),
    actuators=dict(
        arm=ImplicitActuatorCfg(
            joint_names_expr=[
                "fl_joint[1-6]",
                "fr_joint[1-6]",
                "bl_joint[1-6]",
                "br_joint[1-6]",
            ],
            effort_limit_sim=100.0,
            velocity_limit_sim=3.0,
            stiffness=10000.0,
            damping=6.0,
        ),
        gripper=ImplicitActuatorCfg(
            joint_names_expr=[
                "fl_joint7",
                "fl_joint8",
                "fr_joint7",
                "fr_joint8",
                "bl_joint7",
                "bl_joint8",
                "br_joint7",
                "br_joint8",
            ],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.25,
            stiffness=2e3,
            damping=1e2,
        ),
        wheels=ImplicitActuatorCfg(
            joint_names_expr=[".*wheel"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.25,
            stiffness=2e3,
            damping=1e2,
        ),
    ),
    soft_joint_pos_limit_factor=1.0,
)
