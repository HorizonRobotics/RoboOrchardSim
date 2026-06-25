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


# Note: The following import statements should be placed after
# launching the Isaac Sim application.

from robo_orchard_sim.ext.cfg_wrappers.actuators_cfg import ImplicitActuatorCfg
from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import ArticulationCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.ext.models.assets.asset_cfg import NV_ISAAC_DIR

__all__ = [
    "FRANKA_FR3_CFG",
    "FRANKA_FR3_HIGH_PD_CFG",
]

# Configuration of Franka Emika Panda robot with orchard interface insteard of isaac lab interface # noqa: E501
# original franka config is from  isaaclab_assets.robots.franka
FRANKA_FR3_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/franka_fr3",
    spawn=UsdFileCfg(
        usd_path=f"{NV_ISAAC_DIR}/Robots/Franka/FR3/fr3.usd",
        activate_contact_sensors=False,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        semantic_tags=[("class", "franka_fr3")],
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0), # noqa: E501
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.569,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.810,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3.037,
            "fr3_joint7": 0.741,
            "fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=100.0,
            velocity_limit_sim=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_FR3_HIGH_PD_CFG = FRANKA_FR3_CFG.copy()
FRANKA_FR3_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_shoulder"].stiffness = 400.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_shoulder"].damping = 80.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_forearm"].stiffness = 400.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_forearm"].damping = 80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
