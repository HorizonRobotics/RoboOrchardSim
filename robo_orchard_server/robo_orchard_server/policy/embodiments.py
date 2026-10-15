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

"""Robot descriptions shared by model-side adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbodimentProfile:
    """Canonical camera slots and simulator-to-robot-base translation."""

    camera_slots: tuple[str, ...]
    world_to_base_translation: tuple[float, float, float]


_PANDA = EmbodimentProfile(
    ("wrist_camera", "ext1_camera", "ext2_camera"), (0.0, 0.0, 0.0)
)
_PIPER = EmbodimentProfile(
    ("left_wrist", "right_wrist", "base"), (0.3, 0.0, 0.0)
)
_PROFILES = {
    "franka_panda": _PANDA,
    "panda_droid": _PANDA,
    "dualarm_piper": _PIPER,
    "dualarm_piperx": _PIPER,
}


def get_embodiment_profile(embodiment_type: str) -> EmbodimentProfile:
    """Resolve a supported robot without selecting a model checkpoint."""
    try:
        return _PROFILES[embodiment_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported embodiment {embodiment_type!r}; "
            f"expected one of {tuple(_PROFILES)}"
        ) from exc
