# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Configuration dataclasses for the mcap → LeRobot converter.

Hierarchy
---------
PackConfig
└── EmbodimentConfig
    ├── arms: list[JointGroupSpec]
    │   (one per arm: left/right for dual, single for franka)
    └── cameras: dict[str, CameraTopics]

The two built-in presets are in ``config_presets.py``.
"""

from __future__ import annotations
from pathlib import Path

from pydantic import BaseModel, Field


class CameraTopics(BaseModel):
    """All mcap topics for a single camera."""

    color_image: str
    """Topic for compressed color image (jpg bytes)."""
    depth_image: str
    """Topic for compressed depth image (png uint16 bytes)."""
    color_info: str
    """Topic for color camera_info (intrinsics)."""
    depth_info: str
    """Topic for depth camera_info (intrinsics)."""
    tf: str
    """Topic for camera extrinsic FrameTransform."""


class JointGroupSpec(BaseModel):
    """Joint configuration for one arm (or the single arm on franka)."""

    obs_topic: str
    """Observation joint_states topic, e.g. /robot/left/joint_states."""
    action_arm_topic: str
    """Action arm joint_states topic.

    Example: /action/robot_state/left_joint/joint_states.
    """
    action_gripper_topic: str
    """Action gripper joint_states topic.

    Example: /action/robot_state/left_gripper/joint_states.
    """
    arm_dim: int
    """Number of arm joints (excluding gripper). piper=6, franka=7."""
    gripper_joint_name: str = "joint1"
    """Name of the gripper joint used as the scalar gripper value."""
    gripper_scale: float = 2.0
    """Multiplier applied to the scalar gripper position.

    Default 2.0, matching the arrow packer.
    """
    obs_joint_names: list[str] = Field(default_factory=list)
    """Ordered joint names for observation.state arm joints."""
    action_arm_joint_names: list[str] = Field(default_factory=list)
    """Ordered joint names for action arm joints. Used as feature names."""


class EmbodimentConfig(BaseModel):
    """Full description of one robot embodiment for the converter."""

    name: str
    """Short identifier, e.g. 'dualarm_piperx' or 'franka_panda'."""
    robot_type: str
    """LeRobot robot_type string stored in info.json."""
    fps: int = 30
    arms: list[JointGroupSpec]
    """One element per arm. Dual-arm: [left, right]. Single-arm: [arm]."""
    cameras: dict[str, CameraTopics]
    """Map from human-readable camera name to its topic bundle."""
    meta_topic: str = "/meta_data"
    """mcap topic carrying the episode Struct metadata."""
    master_clock_topic: str | None = None
    """Topic used as master clock for time_sync.

    None means ``arms[0].action_arm_topic``.
    """

    # ---- data-cleaning parameters ----
    static_threshold: float = Field(
        default=1e-3,
        description=(
            "Frames where max(|action_pos_diff|) < threshold are considered "
            "static and filtered out. Set to 0 to disable."
        ),
    )
    head_time_to_filter_s: float | None = Field(
        default=None,
        description=(
            "Seconds to strip from the start of each episode. "
            "None = no trimming."
        ),
    )
    tail_time_to_filter_s: float | None = Field(
        default=None,
        description=(
            "Seconds to strip from the end of each episode. "
            "None = no trimming."
        ),
    )

    # ---- robot description ----
    urdf_path: Path | None = None
    """Source URDF file on the packing machine.

    If provided, the URDF text is copied verbatim into ``meta/robot.urdf``
    at convert time and ``info.json.extras.urdf_relative_path`` records
    the relative path so downstream tools can find it without needing the
    original packing-machine path.  Mesh references inside the URDF are
    kept untouched but no mesh files are copied (FK does not need them).
    """

    # ---- visualization parameters (rich video overlay) ----
    viz_state_to_joints: list[list[tuple[str, float]]] = Field(
        default_factory=list,
        description=(
            "Per-state-dimension mapping for FK overlay. Each entry maps "
            "observation.state[i] to a list of (urdf_joint_name, factor); "
            "a single-entry list passes the value through, multi-entry "
            "lists fan a scalar (e.g. gripper) to multiple URDF joints. "
            "Empty list disables overlay-specific joint mapping in viz."
        ),
    )
    viz_ee_links: list[str] = Field(
        default_factory=list,
        description=(
            "URDF link names treated as end-effectors during viz overlay "
            "(longer axis arrows + gripper value label)."
        ),
    )
    viz_drawn_links: list[str] = Field(
        default_factory=list,
        description=(
            "URDF link names whose local XYZ axes are projected onto each "
            "camera image during viz overlay."
        ),
    )
    viz_gripper_scale: float = Field(
        default=2.0,
        description=(
            "Factor used to un-scale the gripper scalar before fan-out to "
            "finger joints in viz. Default 2.0 matches arrow packer."
        ),
    )
    viz_state_table_rows: list[tuple[str, int, int]] = Field(
        default_factory=list,
        description=(
            "Joint-state table layout for viz overlay. Each entry is "
            "(row_label, state_offset, state_length). Cells are labelled "
            "j1, j2, ... and the last column is 'g' (gripper). Empty list "
            "disables the joint table."
        ),
    )


class PackConfig(BaseModel):
    """Top-level packing configuration handed to convert.py."""

    embodiment: EmbodimentConfig
    use_videos: bool = True
    """Store color images as mp4 videos (True) or per-frame PNGs (False)."""
    image_scale: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description=(
            "Image downscale factor. Reserved for future use; "
            "currently ignored."
        ),
    )
