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
from robo_orchard_sim.cfg_wrappers.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.models.assets.asset_cfg import ORCHARD_ASSET

__all__ = ["GALAXEA_A1_GRIPPER_CFG"]


# configuration of galaxea A1 robot
GALAXEA_A1_GRIPPER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/galaxea_a1_with_gripper",
    spawn=UsdFileCfg(
        usd_path=f"{ORCHARD_ASSET}/ROBOTS/GALAXEA/A1_with_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "arm_joint1": 0.0,
            "arm_joint2": 0.0,
            "arm_joint3": 0.0,
            "arm_joint4": 0.0,
            "arm_joint5": 0.0,
            "arm_joint6": 0.0,
            "gripper_axis1": 0.0,
            "gripper_axis2": 0.0,
        },
        pos=(-0.021132780158325766, -0.7327480564695673, 0.015064131873759656),
        rot=(
            0.9998959215482877,
            -0.00941013310753452,
            -0.005014801766955938,
            -0.00971839643342437,
        ),
    ),
    actuators=dict(
        arm=ImplicitActuatorCfg(
            joint_names_expr=["arm_joint[1-6]"],
            effort_limit=100.0,
            velocity_limit=3.0,
            stiffness=10000.0,
            damping=6.0,
        ),
        gripper=ImplicitActuatorCfg(
            joint_names_expr=["gripper_axis1", "gripper_axis2"],
            effort_limit=200.0,
            velocity_limit=0.25,
            stiffness=2e3,
            damping=1e2,
        ),
    ),
    soft_joint_pos_limit_factor=1.0,
)
