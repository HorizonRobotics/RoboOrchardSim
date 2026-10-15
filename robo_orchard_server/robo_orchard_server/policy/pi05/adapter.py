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

"""Map canonical observations and physical joint actions for OpenPI."""

import numpy as np

from robo_orchard_server.policy.embodiments import get_embodiment_profile
from robo_orchard_server.server.types import JointAction, Observation


def parse_image(value):
    """Accept LeRobot float CHW or simulator uint8 HWC RGB."""
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError("Expected one RGB image")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError("Expected three RGB channels")
    if np.issubdtype(image.dtype, np.floating):
        if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("Float images must be finite and in [0, 1]")
        image = np.rint(image * 255).astype(np.uint8)
    if image.dtype != np.uint8:
        raise ValueError("Images must be uint8 or float [0, 1]")
    return np.ascontiguousarray(image)


class Pi05Adapter:
    """Use the runtime action layout and configured model input paths."""

    def __init__(self, camera_paths, state_path, prompt_path=("prompt",)):
        if not camera_paths:
            raise ValueError("At least one camera mapping is required")
        self.camera_paths = camera_paths
        self.state_path = state_path
        self.prompt_path = prompt_path
        paths = [state_path, prompt_path, *camera_paths.values()]
        for path in paths:
            if (
                not isinstance(path, (list, tuple))
                or not path
                or not all(isinstance(p, str) and p for p in path)
            ):
                raise ValueError("Input paths must be nonempty string lists")
        for i, path in enumerate(paths):
            for other in paths[i + 1 :]:
                n = min(len(path), len(other))
                if list(path[:n]) == list(other[:n]):
                    raise ValueError("Model input paths overlap")

    @staticmethod
    def layout(obs: Observation) -> list[dict]:
        layout = obs["action_layout"]
        order = layout["manipulator_order"]
        if (
            not order
            or len(order) != len(set(order))
            or set(order) != set(layout["manipulators"])
            or set(order) != set(obs["manipulators"])
        ):
            raise ValueError("Invalid manipulator layout")
        specs = [layout["manipulators"][slot] for slot in order]
        names = []
        for spec in specs:
            names.extend(spec["arm_joint_names"] + spec["gripper_joint_names"])
            count = len(spec["gripper_joint_names"])
            if not count:
                continue
            if (
                not np.isfinite(spec["gripper_policy_scale"])
                or not spec["gripper_policy_scale"]
            ):
                raise ValueError("Gripper scale must be finite and nonzero")
            rep, coupling = (
                spec["gripper_policy_representation"],
                spec["gripper_decode_coupling"],
            )
            if rep == "all_joints":
                valid = coupling == "identity"
            elif rep == "first_joint":
                valid = coupling in ("symmetric", "mirrored") or (
                    coupling == "identity" and count == 1
                )
            else:
                valid = False
            if not valid:
                raise ValueError("Unsupported gripper representation/coupling")
        if not names or len(names) != len(set(names)):
            raise ValueError("Layout must provide unique joint names")
        return specs

    @staticmethod
    def _set_path(result, path, value):
        for key in path[:-1]:
            result = result.setdefault(key, {})
        result[path[-1]] = value

    def build_model_input(self, obs: Observation) -> dict:
        specs = self.layout(obs)
        profile = get_embodiment_profile(
            obs["action_layout"]["embodiment_type"]
        )
        if not set(self.camera_paths).issubset(profile.camera_slots):
            raise ValueError("Model cameras do not match the embodiment")
        states = []
        for slot, spec in zip(
            obs["action_layout"]["manipulator_order"], specs, strict=True
        ):
            manip = obs["manipulators"][slot]
            arm_dim = len(spec["arm_joint_names"])
            grip_dim = len(spec["gripper_joint_names"])
            joints = np.asarray(manip["joint_position"], dtype=np.float32)
            if (
                joints.ndim != 2
                or joints.shape[0] != 1
                or joints.shape[1] not in (arm_dim, arm_dim + grip_dim)
            ):
                raise ValueError(
                    "Expected single-environment joint state matching layout"
                )
            states.append(joints[0, :arm_dim])
            if grip_dim:
                grip = np.asarray(
                    manip.get("gripper_position", joints[:, arm_dim:]),
                    dtype=np.float32,
                )
                if grip.shape != (1, grip_dim):
                    raise ValueError(
                        "Measured gripper state does not match layout"
                    )
                if spec["gripper_policy_representation"] == "first_joint":
                    grip = grip[:, :1]
                states.append(grip[0] * spec["gripper_policy_scale"])
        state = np.concatenate(states)
        if not np.isfinite(state).all():
            raise ValueError("Nonfinite measured state")
        prompt = obs["instruction"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A nonempty episode instruction is required")
        result = {}
        self._set_path(result, self.state_path, state)
        self._set_path(result, self.prompt_path, prompt)
        for camera, path in self.camera_paths.items():
            image = np.asarray(obs["cameras"][camera]["rgb"])
            if image.ndim != 4 or image.shape[0] != 1:
                raise ValueError("Expected RGB [1, H, W, 3]")
            self._set_path(result, path, parse_image(image[0]))
        return result

    def build_action_sequence(
        self, actions, obs, steps=None
    ) -> list[JointAction]:
        specs = self.layout(obs)
        values = np.asarray(actions, dtype=np.float32)
        dims = [
            len(s["arm_joint_names"])
            + (
                len(s["gripper_joint_names"])
                if s["gripper_policy_representation"] == "all_joints"
                else int(bool(s["gripper_joint_names"]))
            )
            for s in specs
        ]
        if (
            values.ndim != 2
            or not len(values)
            or values.shape[1] < sum(dims)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "Action chunk must be finite and cover the layout"
            )
        if steps is not None and not 1 <= steps <= len(values):
            raise ValueError("Execution steps exceed returned chunk")
        values = values[:steps]
        names, physical = [], []
        cursor = 0
        for spec, dim in zip(specs, dims, strict=True):
            arm_dim = len(spec["arm_joint_names"])
            names.extend(spec["arm_joint_names"] + spec["gripper_joint_names"])
            physical.append(values[:, cursor : cursor + arm_dim])
            count = len(spec["gripper_joint_names"])
            if count:
                grip = (
                    values[:, cursor + arm_dim : cursor + dim]
                    / spec["gripper_policy_scale"]
                )
                if spec["gripper_policy_representation"] == "first_joint":
                    grip = np.repeat(grip, count, axis=1)
                    if spec["gripper_decode_coupling"] == "mirrored":
                        grip[:, 1:] *= -1
                physical.append(grip)
            cursor += dim
        targets = np.concatenate(physical, axis=1)
        if not np.isfinite(targets).all():
            raise ValueError("Decoded joint targets must be finite")
        return [
            {"joint_names": list(names), "values": row[None].copy()}
            for row in targets
        ]
