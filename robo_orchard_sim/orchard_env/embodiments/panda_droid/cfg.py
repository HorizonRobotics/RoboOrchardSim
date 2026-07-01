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
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    ArticulationRootPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import UsdFileCfg
from robo_orchard_sim.ext.models.assets.asset_cfg import ORCHARD_ASSET

__all__ = [
    "PANDA_DROID_CFG",
    "PANDA_DROID_HIGH_PD_CFG",
]


PANDA_DROID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/panda_droid",
    spawn=UsdFileCfg(
        usd_path=(
            f"{ORCHARD_ASSET}/ROBOTS/FRANKA/franka_panda_robotiq_flange.usd"
        ),
        semantic_tags=[("class", "panda_droid")],
        activate_contact_sensors=False,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.6283,
            "panda_joint3": 0.0,
            "panda_joint4": -2.5133,
            "panda_joint5": 0.0,
            "panda_joint6": 1.8850,
            "panda_joint7": 0.0,
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "robotiq_85_gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            velocity_limit_sim=5.0,
            stiffness=None,
            damping=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Panda Droid robot from the USD asset."""


PANDA_DROID_HIGH_PD_CFG = PANDA_DROID_CFG.copy()
PANDA_DROID_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
PANDA_DROID_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
PANDA_DROID_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
PANDA_DROID_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
PANDA_DROID_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
"""Configuration of Panda Droid robot with stiffer arm PD control."""
