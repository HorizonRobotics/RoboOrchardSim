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

"""Visual color palette for procedurally colored stack-cubes assets."""

CUBE_COLOR_SRGB: dict[str, tuple[float, float, float]] = {
    "red": (0.90, 0.10, 0.10),
    "orange": (1.00, 0.45, 0.05),
    "yellow": (0.95, 0.85, 0.05),
    "lime": (0.55, 0.85, 0.10),
    "green": (0.05, 0.60, 0.20),
    "cyan": (0.05, 0.80, 0.80),
    "blue": (0.05, 0.30, 0.90),
    "purple": (0.45, 0.15, 0.80),
    "magenta": (0.85, 0.10, 0.65),
    "brown": (0.45, 0.22, 0.08),
}


def _srgb_to_linear(channel: float) -> float:
    """Convert one normalized sRGB channel for USD PreviewSurface."""
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


CUBE_COLOR_PALETTE = {
    name: tuple(_srgb_to_linear(channel) for channel in color)
    for name, color in CUBE_COLOR_SRGB.items()
}
CUBE_COLOR_NAMES = tuple(CUBE_COLOR_PALETTE)

__all__ = [
    "CUBE_COLOR_NAMES",
    "CUBE_COLOR_PALETTE",
    "CUBE_COLOR_SRGB",
]
