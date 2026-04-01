## Copyright (c) 2024 Horizon Robotics. All Rights Reserved.

import pytest

from robo_orchard_sim.utils.env_utils import sample_poses


class TestEnvUtils:
    def test_scattered_mode_poses_scattered(self):
        pose_range = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "z": (0.0, 1.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        mode = "scattered"
        length = 10
        poses = sample_poses(pose_range, mode, length)
        assert len(poses) == length
        for pose in poses:
            assert pose_range["x"][0] <= pose[0] <= pose_range["x"][1]
            assert pose_range["y"][0] <= pose[1] <= pose_range["y"][1]
            assert pose_range["z"][0] <= pose[2] <= pose_range["z"][1]

    def test_orderly_mode_poses(self):
        pose_range = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "z": (0.0, 1.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        mode = "orderly"
        length = 9
        poses = sample_poses(pose_range, mode, length)
        assert len(poses) == length
        for pose in poses:
            assert pose_range["x"][0] <= pose[0] <= pose_range["x"][1]
            assert pose_range["y"][0] <= pose[1] <= pose_range["y"][1]
            assert pose_range["z"][0] <= pose[2] <= pose_range["z"][1]

    def test_stacked_mode_poses_stacked(self):
        pose_range = {
            "x": (0.0, 1.0),
            "y": (0.0, 1.0),
            "z": (0.0, 1.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        mode = "stacked"
        length = 5
        poses = sample_poses(pose_range, mode, length)
        assert len(poses) == length
        x_mid = (
            pose_range["x"][0]
            + (pose_range["x"][1] - pose_range["x"][0]) / 2.0
        )
        y_mid = (
            pose_range["y"][0]
            + (pose_range["y"][1] - pose_range["y"][0]) / 2.0
        )
        for i, pose in enumerate(poses):
            assert x_mid - 0.1 <= pose[0] <= x_mid + 0.1
            assert y_mid - 0.1 <= pose[1] <= y_mid + 0.1
            assert (
                pose_range["z"][0]
                + i * (pose_range["z"][1] - pose_range["z"][0]) / (length - 1)
                <= pose[2]
                <= pose_range["z"][1]
            )


if __name__ == "__main__":
    pytest.main(["-s", "test_env_utils.py"])
