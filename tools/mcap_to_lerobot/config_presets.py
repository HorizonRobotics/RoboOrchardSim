# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Built-in EmbodimentConfig presets for dualarm_piperx and franka_panda."""

from __future__ import annotations
from pathlib import Path

from tools.mcap_to_lerobot.config import (
    CameraTopics,
    EmbodimentConfig,
    JointGroupSpec,
)

# ---------------------------------------------------------------------------
# dualarm_piperx
# ---------------------------------------------------------------------------
_PIPERX_CAMERAS = {
    "static_camera": CameraTopics(
        color_image="/observation/cameras/static_camera/color_image/image_raw",
        depth_image="/observation/cameras/static_camera/depth_image/image_raw",
        color_info="/observation/cameras/static_camera/color_image/camera_info",
        depth_info="/observation/cameras/static_camera/depth_image/camera_info",
        tf="/observation/cameras/static_camera/color_image/tf",
    ),
    "left_hand_camera": CameraTopics(
        color_image="/observation/cameras/left_hand_camera/color_image/image_raw",
        depth_image="/observation/cameras/left_hand_camera/depth_image/image_raw",
        color_info="/observation/cameras/left_hand_camera/color_image/camera_info",
        depth_info="/observation/cameras/left_hand_camera/depth_image/camera_info",
        tf="/observation/cameras/left_hand_camera/color_image/tf",
    ),
    "right_hand_camera": CameraTopics(
        color_image="/observation/cameras/right_hand_camera/color_image/image_raw",
        depth_image="/observation/cameras/right_hand_camera/depth_image/image_raw",
        color_info="/observation/cameras/right_hand_camera/color_image/camera_info",
        depth_info="/observation/cameras/right_hand_camera/depth_image/camera_info",
        tf="/observation/cameras/right_hand_camera/color_image/tf",
    ),
    # "vis_camera": CameraTopics(
    #     color_image="/observation/cameras/vis_camera/color_image/image_raw",
    #     depth_image="/observation/cameras/vis_camera/depth_image/image_raw",
    #     color_info="/observation/cameras/vis_camera/color_image/camera_info",
    #     depth_info="/observation/cameras/vis_camera/depth_image/camera_info",
    #     tf="/observation/cameras/vis_camera/color_image/tf",
    # ),
}

DUALARM_PIPERX_CONFIG = EmbodimentConfig(
    name="dualarm_piperx",
    robot_type="dualarm_piperx",
    fps=30,
    arms=[
        JointGroupSpec(
            obs_topic="/robot/left/joint_states",
            action_arm_topic="/action/robot_state/left_joint/joint_states",
            action_gripper_topic="/action/robot_state/left_gripper/joint_states",
            arm_dim=6,
            gripper_joint_name="joint1",
            gripper_scale=2.0,
            obs_joint_names=[
                "left_joint1",
                "left_joint2",
                "left_joint3",
                "left_joint4",
                "left_joint5",
                "left_joint6",
                "left_gripper",
            ],
            action_arm_joint_names=[
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_gripper",
            ],
        ),
        JointGroupSpec(
            obs_topic="/robot/right/joint_states",
            action_arm_topic="/action/robot_state/right_joint/joint_states",
            action_gripper_topic="/action/robot_state/right_gripper/joint_states",
            arm_dim=6,
            gripper_joint_name="joint1",
            gripper_scale=2.0,
            obs_joint_names=[
                "right_joint1",
                "right_joint2",
                "right_joint3",
                "right_joint4",
                "right_joint5",
                "right_joint6",
                "right_gripper",
            ],
            action_arm_joint_names=[
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
                "right_arm_gripper",
            ],
        ),
    ],
    cameras=_PIPERX_CAMERAS,
    master_clock_topic="/action/robot_state/left_joint/joint_states",
    # 5e-3 rad ~= 0.29 deg, comfortable margin above observation joint noise
    # while still flagging any deliberate hand-off motion as "moving".
    static_threshold=5e-3,
    head_time_to_filter_s=None,
    tail_time_to_filter_s=None,
    urdf_path=Path(
        "/horizon-bucket/robot_lab/assets/ROBOTS/"
        "piper_x_description/piper_x_description_dualarm_dark.urdf"
        # "/home/users/ziang.li-labs/robo_orchard_sim/assets/"
        # "piper_x_description/piper_x_description_dualarm_dark_with_ee.urdf"
    ),
    # ---- viz overlay ----
    # state layout: [L_arm×6, L_gripper, R_arm×6, R_gripper] = 14 dims
    viz_state_to_joints=[
        [("left_joint1", 1.0)],
        [("left_joint2", 1.0)],
        [("left_joint3", 1.0)],
        [("left_joint4", 1.0)],
        [("left_joint5", 1.0)],
        [("left_joint6", 1.0)],
        [("left_joint7", 0.5), ("left_joint8", 0.5)],
        [("right_joint1", 1.0)],
        [("right_joint2", 1.0)],
        [("right_joint3", 1.0)],
        [("right_joint4", 1.0)],
        [("right_joint5", 1.0)],
        [("right_joint6", 1.0)],
        [("right_joint7", 0.5), ("right_joint8", 0.5)],
    ],
    viz_ee_links=["left_ee_link", "right_ee_link"],
    viz_drawn_links=[
        "left_link1",
        "left_link2",
        "left_link3",
        "left_link4",
        "left_link5",
        "left_link6",
        "left_ee_link",
        "right_link1",
        "right_link2",
        "right_link3",
        "right_link4",
        "right_link5",
        "right_link6",
        "right_ee_link",
    ],
    viz_gripper_scale=2.0,
    viz_state_table_rows=[("L", 0, 7), ("R", 7, 7)],
)

# ---------------------------------------------------------------------------
# franka_panda
# (preset wired up; end-to-end not yet validated: no healthy mcap)
# ---------------------------------------------------------------------------
_FRANKA_CAMERAS = {
    "ext1_camera": CameraTopics(
        color_image="/observation/cameras/ext1_camera/color_image/image_raw",
        depth_image="/observation/cameras/ext1_camera/depth_image/image_raw",
        color_info="/observation/cameras/ext1_camera/color_image/camera_info",
        depth_info="/observation/cameras/ext1_camera/depth_image/camera_info",
        tf="/observation/cameras/ext1_camera/color_image/tf",
    ),
    "ext2_camera": CameraTopics(
        color_image="/observation/cameras/ext2_camera/color_image/image_raw",
        depth_image="/observation/cameras/ext2_camera/depth_image/image_raw",
        color_info="/observation/cameras/ext2_camera/color_image/camera_info",
        depth_info="/observation/cameras/ext2_camera/depth_image/camera_info",
        tf="/observation/cameras/ext2_camera/color_image/tf",
    ),
    "wrist_camera": CameraTopics(
        color_image="/observation/cameras/wrist_camera/color_image/image_raw",
        depth_image="/observation/cameras/wrist_camera/depth_image/image_raw",
        color_info="/observation/cameras/wrist_camera/color_image/camera_info",
        depth_info="/observation/cameras/wrist_camera/depth_image/camera_info",
        tf="/observation/cameras/wrist_camera/color_image/tf",
    ),
}

FRANKA_PANDA_CONFIG = EmbodimentConfig(
    name="franka_panda",
    robot_type="franka_panda",
    fps=30,
    arms=[
        JointGroupSpec(
            obs_topic="/robot/joint_states",
            action_arm_topic="/action/robot_state/joint/joint_states",
            action_gripper_topic="/action/robot_state/gripper/joint_states",
            arm_dim=7,
            gripper_joint_name="joint1",
            gripper_scale=2.0,
            obs_joint_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_gripper",
            ],
            action_arm_joint_names=[
                "panda_arm_joint1",
                "panda_arm_joint2",
                "panda_arm_joint3",
                "panda_arm_joint4",
                "panda_arm_joint5",
                "panda_arm_joint6",
                "panda_arm_joint7",
                "panda_arm_gripper",
            ],
        ),
    ],
    cameras=_FRANKA_CAMERAS,
    master_clock_topic="/action/robot_state/joint/joint_states",
    static_threshold=5e-3,
    head_time_to_filter_s=None,
    tail_time_to_filter_s=None,
    urdf_path=Path(
        "/horizon-bucket/robot_lab/assets/ROBOTS/FRANKA/franka_panda.urdf"
    ),
    # ---- viz overlay ----
    # state layout: [arm×7, gripper] = 8 dims; gripper fans to two prismatic
    # finger joints (panda_finger_joint{1,2}).
    viz_state_to_joints=[
        [("panda_joint1", 1.0)],
        [("panda_joint2", 1.0)],
        [("panda_joint3", 1.0)],
        [("panda_joint4", 1.0)],
        [("panda_joint5", 1.0)],
        [("panda_joint6", 1.0)],
        [("panda_joint7", 1.0)],
        [("panda_finger_joint1", 0.5), ("panda_finger_joint2", 0.5)],
    ],
    viz_ee_links=["ee_link"],
    viz_drawn_links=[
        "panda_link1",
        "panda_link2",
        "panda_link3",
        "panda_link4",
        "panda_link5",
        "panda_link6",
        "panda_link7",
        "panda_link8",
        "panda_hand",
        "ee_link",
    ],
    viz_gripper_scale=2.0,
    viz_state_table_rows=[("arm", 0, 8)],
)

# Registry for CLI --embodiment lookup
EMBODIMENT_REGISTRY: dict[str, EmbodimentConfig] = {
    "dualarm_piperx": DUALARM_PIPERX_CONFIG,
    "franka_panda": FRANKA_PANDA_CONFIG,
}
