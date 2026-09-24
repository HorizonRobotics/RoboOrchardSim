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

"""Placing objects at stated distances from a reference.

Kept free of simulator imports so the arrangement can be studied on its
own: whether a target really does turn up in every direction, and how
often two objects crowd each other, are statistical questions that need
thousands of draws to answer and would be impractical to ask through a
running simulation.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass

__all__ = [
    "PolarLayout",
    "PolarLayoutError",
    "PolarPlacement",
]

_AXES_DEGREES: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0)
"""Directions objects are spread over, as offsets from the reference.

Four axes a quarter-turn apart keep the objects from crowding: even at
full jitter two of them stay 10 degrees apart, and typically far more.
"""


class PolarLayoutError(RuntimeError):
    """No arrangement satisfied the distances, workspace and spacing."""


@dataclass(frozen=True)
class PolarPlacement:
    """Where one object goes relative to the reference.

    ``radius`` is the band it is drawn from. ``axis`` pins the direction
    when the direction is itself the point -- an object described as
    being to the left has to actually be to the left, every episode.
    Leaving it unset hands the object whichever axis is free, which is
    what keeps a direction from becoming a giveaway when the
    instruction talks about distance instead.
    """

    radius: tuple[float, float]
    axis: float | None = None

    def __post_init__(self) -> None:
        low, high = self.radius
        if low <= 0.0:
            raise ValueError(f"radius must be positive, got {low}.")
        if high < low:
            raise ValueError(
                f"radius must be ordered (min, max), got {self.radius}."
            )
        if self.axis is not None and self.axis not in _AXES_DEGREES:
            raise ValueError(
                f"axis must be one of {list(_AXES_DEGREES)}, got {self.axis}."
            )


class PolarLayout:
    """Arranges objects around a reference within a fixed workspace.

    The workspace and spacing hold for a whole task, so they are settled
    once here and checked once, while the reference moves and the
    objects are redealt on every episode through :meth:`solve`.
    """

    def __init__(
        self,
        *,
        workspace_x: tuple[float, float],
        workspace_y: tuple[float, float],
        axis_jitter_deg: float,
        min_separation: float,
        axis_retries: int = 32,
        shuffle_retries: int = 8,
    ) -> None:
        """Settle the bounds every episode will be arranged inside.

        Args:
            workspace_x: Reachable ``x`` span; placements stay inside it.
            workspace_y: Reachable ``y`` span.
            axis_jitter_deg: Half-width of the arc an object may occupy
                around its axis. Must stay under half the axis spacing,
                or neighbouring arcs would overlap and two objects could
                swap which side of the reference they are on.
            min_separation: Least distance between any two objects, and
                between an object and the reference.
            axis_retries: Redraws allowed on one axis before the whole
                assignment is redealt.
            shuffle_retries: Redeals allowed before an arrangement is
                declared impossible. Redrawing within an axis can only
                shift an object along one direction, so an object whose
                axis points out of the workspace stays stuck no matter
                how many times it tries. Redealing moves it to another
                axis, which is what turns a ~75% success rate into a
                certainty.

        Raises:
            ValueError: The jitter is wide enough to let neighbouring
                arcs overlap.
        """
        axis_spacing = 360.0 / len(_AXES_DEGREES)
        if axis_jitter_deg >= axis_spacing / 2.0:
            raise ValueError(
                f"axis_jitter of {axis_jitter_deg} degrees reaches into the "
                f"neighbouring axis {axis_spacing} degrees away; keep it "
                f"under {axis_spacing / 2.0} so each object stays on its "
                "own side."
            )
        self._workspace_x = workspace_x
        self._workspace_y = workspace_y
        self._jitter = math.radians(axis_jitter_deg)
        self._min_separation = min_separation
        self._axis_retries = axis_retries
        self._shuffle_retries = shuffle_retries

    def solve(
        self,
        anchor: tuple[float, float],
        placements: dict[str, PolarPlacement],
        rng: random.Random | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Place every object at its stated distance from ``anchor``.

        Each object gets one axis and is drawn at a distance from its own
        band. An object that pins its ``axis`` keeps that direction,
        because for it the direction is the thing being described. The
        rest are dealt the remaining axes afresh on every call: were
        their directions fixed too, the nearest object would always lie
        the same way, and always the same angle from the farthest, and a
        policy could read either instead of the distance the instruction
        actually names.

        Args:
            anchor: Where the reference object ended up, ``(x, y)``.
            placements: Distance band, and optionally a pinned axis, per
                object keyed by scene name.
            rng: Source of randomness. Defaults to the module-level
                ``random``, which the simulator seeds per episode, so a
                replayed episode arranges itself the same way.

        Returns:
            Object scene name to ``(x, y)``.

        Raises:
            PolarLayoutError: Every attempt failed. The message reports
                which object could not be placed, so the cause -- bands
                too wide, workspace too tight, spacing too large -- is
                readable without instrumenting the sampler.
            ValueError: ``placements`` is empty, holds more objects than
                there are axes, or two objects pin the same axis.
        """
        if not placements:
            raise ValueError("A layout needs at least one placement.")
        if len(placements) > len(_AXES_DEGREES):
            raise ValueError(
                f"{len(placements)} objects will not fit on "
                f"{len(_AXES_DEGREES)} axes; add axes or drop an object."
            )

        draw = rng if rng is not None else random
        keys = sorted(placements)
        pinned = _pinned_axes(placements, keys)
        free_keys = [key for key in keys if key not in pinned]
        spare_axes = [
            axis for axis in _AXES_DEGREES if axis not in pinned.values()
        ]
        stuck_on: str | None = None

        for _ in range(self._shuffle_retries):
            draw.shuffle(spare_axes)
            assigned = dict(pinned)
            assigned.update(zip(free_keys, spare_axes, strict=False))
            # The reference is already down, and a near object sits close
            # to it, so it has to be avoided like any other.
            taken: list[tuple[float, float]] = [anchor]
            placed: dict[str, tuple[float, float]] = {}
            for key in keys:
                spot = self._draw_on_axis(
                    anchor=anchor,
                    axis_deg=assigned[key],
                    radius=placements[key].radius,
                    taken=taken,
                    draw=draw,
                )
                if spot is None:
                    stuck_on = key
                    break
                placed[key] = spot
                taken.append(spot)
            else:
                return placed

        assert stuck_on is not None
        raise PolarLayoutError(self._failure(stuck_on, anchor, placements))

    def _draw_on_axis(
        self,
        *,
        anchor: tuple[float, float],
        axis_deg: float,
        radius: tuple[float, float],
        taken: list[tuple[float, float]],
        draw,
    ) -> tuple[float, float] | None:
        """Find a spot on one axis, or report that this axis will not do."""
        axis = math.radians(axis_deg)
        low_x, high_x = self._workspace_x
        low_y, high_y = self._workspace_y
        for _ in range(self._axis_retries):
            theta = axis + draw.uniform(-self._jitter, self._jitter)
            distance = draw.uniform(*radius)
            x = anchor[0] + distance * math.cos(theta)
            y = anchor[1] + distance * math.sin(theta)
            if not (low_x <= x <= high_x and low_y <= y <= high_y):
                continue
            if any(
                math.dist((x, y), spot) < self._min_separation
                for spot in taken
            ):
                continue
            return (x, y)
        return None

    def _failure(
        self,
        stuck_on: str,
        anchor: tuple[float, float],
        placements: dict[str, PolarPlacement],
    ) -> str:
        """Say what ran out, and whether redealing had any say in it."""
        axis = placements[stuck_on].axis
        if axis is None:
            effort = (
                f"after {self._shuffle_retries} deals across "
                f"{len(_AXES_DEGREES)} axes"
            )
        else:
            # Redealing never moves a pinned object, so the budget it
            # exhausted was the per-axis one, however many deals ran.
            effort = (
                f"after {self._axis_retries} draws on the {axis} degree "
                "axis it is pinned to, which redealing cannot move it off"
            )
        return (
            f"Could not place {stuck_on!r} around a reference at "
            f"{anchor[0]:.3f}, {anchor[1]:.3f} {effort}. Its distance band "
            f"is {placements[stuck_on].radius}, the workspace spans "
            f"x={self._workspace_x} y={self._workspace_y}, and objects must "
            f"stay {self._min_separation}m apart -- widen the workspace, "
            "narrow the band, or reduce the spacing."
        )


def _pinned_axes(
    placements: dict[str, PolarPlacement],
    keys: list[str],
) -> dict[str, float]:
    """Objects whose direction is fixed, keyed by scene name."""
    pinned: dict[str, float] = {}
    for key in keys:
        axis = placements[key].axis
        if axis is None:
            continue
        clash = next(
            (other for other, taken in pinned.items() if taken == axis),
            None,
        )
        if clash is not None:
            raise ValueError(
                f"{key!r} and {clash!r} both ask for the {axis} degree "
                "axis; two objects cannot share a direction."
            )
        pinned[key] = axis
    return pinned
