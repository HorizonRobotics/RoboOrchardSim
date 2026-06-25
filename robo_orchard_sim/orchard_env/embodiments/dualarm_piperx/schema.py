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

"""Canonical policy bindings for dual-arm PiperX embodiments."""

from robo_orchard_sim.contracts.policy_binding import (
    CameraBinding,
    ManipulatorBinding,
    PolicyBindingSchema,
)


def build_dualarm_piperx_policy_binding_schema(
    embodiment_type: str,
) -> PolicyBindingSchema:
    """Build the dual-arm PiperX policy binding schema."""
    return PolicyBindingSchema(
        schema_version="1",
        embodiment_type=embodiment_type,
        camera_slots={
            "left_wrist": CameraBinding(
                obs_term="left_hand_camera_term",
                rgb=True,
                depth=True,
                intrinsic=True,
                pose=True,
            ),
            "right_wrist": CameraBinding(
                obs_term="right_hand_camera_term",
                rgb=True,
                depth=True,
                intrinsic=True,
                pose=True,
            ),
            "base": CameraBinding(
                obs_term="static_camera_term",
                rgb=True,
                depth=True,
                intrinsic=True,
                pose=True,
            ),
        },
        manipulator_slots={
            "left_arm": ManipulatorBinding(
                joint_position_obs_key="left_joint_position",
                arm_joint_name_specs=("left_joint[1-6]",),
                gripper_joint_name_specs=("left_joint[7-8]",),
                gripper_policy_representation="first_joint",
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            ),
            "right_arm": ManipulatorBinding(
                joint_position_obs_key="right_joint_position",
                arm_joint_name_specs=("right_joint[1-6]",),
                gripper_joint_name_specs=("right_joint[7-8]",),
                gripper_policy_representation="first_joint",
                gripper_decode_coupling="mirrored",
                gripper_policy_scale=2.0,
            ),
        },
    )
