# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""Shared data containers for reader and packer."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CameraCalib:
    """Calibration snapshot for one camera."""

    cam_name: str
    intrinsic_K: list[float]
    """Flat 9-element K matrix (row-major 3x3) from camera_info.K."""
    intrinsic_P: list[float]
    """Flat 12-element projection matrix P from camera_info.P."""
    distortion_model: str
    distortion_D: list[float]
    width: int
    height: int
    # extrinsic
    parent_frame_id: str
    child_frame_id: str
    translation: dict[str, float]  # {x, y, z}
    rotation: dict[str, float]  # {x, y, z, w}

    def to_dict(self) -> dict:
        return {
            "intrinsic_K": self.intrinsic_K,
            "intrinsic_P": self.intrinsic_P,
            "distortion_model": self.distortion_model,
            "distortion_D": self.distortion_D,
            "width": self.width,
            "height": self.height,
            "extrinsic": {
                "parent_frame_id": self.parent_frame_id,
                "child_frame_id": self.child_frame_id,
                "translation": self.translation,
                "rotation": self.rotation,
            },
        }


@dataclass
class RawEpisode:
    """One decoded and aligned mcap episode."""

    frames: list[dict[str, Any]]
    """Per-frame dicts. Keys must match the LeRobot features schema exactly."""

    task: str
    """Episode-level task/instruction string."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Raw decoded /meta_data Struct (kept for debugging)."""

    calib: dict[str, CameraCalib] = field(default_factory=dict)
    """Camera calibration keyed by camera name."""

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:  # noqa: D105
        cams = list(self.calib)
        return (
            f"RawEpisode(frames={self.num_frames}, "
            f"task={self.task!r}, cameras={cams})"
        )
