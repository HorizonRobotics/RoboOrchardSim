# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
"""How objects get arranged around a reference at stated distances.

The properties that matter here are statistical -- whether a target
turns up in every direction, whether two objects crowd each other --
and answering them takes thousands of draws. That is why the
arrangement is solved apart from the simulator: these run in a second.
"""

from __future__ import annotations
import inspect
import math
import random

import pytest

from robo_orchard_sim.utils.polar_layout import (
    PolarLayout,
    PolarLayoutError,
    PolarPlacement,
)

_NEAR = (0.11, 0.16)
_FAR = (0.30, 0.38)
_CLUTTER = (0.20, 0.26)

_PLACEMENTS = {
    "pick_near": PolarPlacement(radius=_NEAR),
    "pick_far": PolarPlacement(radius=_FAR),
    "clutter_0": PolarPlacement(radius=_CLUTTER),
    "clutter_1": PolarPlacement(radius=_CLUTTER),
}

_WORKSPACE = {
    "workspace_x": (0.25, 0.80),
    "workspace_y": (-0.40, 0.40),
}
_ANCHOR_X = (0.48, 0.62)
_ANCHOR_Y = (-0.08, 0.08)

_SETTINGS = dict(
    axis_jitter_deg=40.0,
    min_separation=0.03,
    **_WORKSPACE,
)

_LAYOUT = PolarLayout(**_SETTINGS)


def _bearing(spot: tuple[float, float], anchor: tuple[float, float]) -> float:
    """Compass direction from the anchor to a spot, in degrees."""
    return (
        math.degrees(math.atan2(spot[1] - anchor[1], spot[0] - anchor[0]))
        % 360
    )


def _separation(a: float, b: float) -> float:
    """Smaller of the two angles between two bearings."""
    gap = abs(a - b) % 360
    return min(gap, 360 - gap)


def _sample(count: int, seed: int = 0):
    """Yield (anchor, layout) for a run of episodes."""
    rng = random.Random(seed)
    for _ in range(count):
        anchor = (
            rng.uniform(*_ANCHOR_X),
            rng.uniform(*_ANCHOR_Y),
        )
        yield (
            anchor,
            _LAYOUT.solve(anchor, _PLACEMENTS, rng=rng),
        )


def test_settings_in_use_always_find_an_arrangement():
    """The shipped numbers leave enough room to always succeed."""
    for _ in _sample(2000):
        pass


def test_the_far_object_is_always_further_than_the_near_one():
    """The words the instruction uses have to match the scene.

    Distance is the only thing separating the two candidates, so a
    single episode where they cross would be an episode whose
    instruction names the wrong object.
    """
    for anchor, layout in _sample(2000):
        near = math.dist(layout["pick_near"], anchor)
        far = math.dist(layout["pick_far"], anchor)
        assert far > near, f"near={near:.3f} far={far:.3f} at {anchor}"


def test_clutter_never_out_ranges_a_candidate():
    """Clutter stays in the gap, so neither word could describe it."""
    for anchor, layout in _sample(2000):
        near = math.dist(layout["pick_near"], anchor)
        far = math.dist(layout["pick_far"], anchor)
        for key in ("clutter_0", "clutter_1"):
            distance = math.dist(layout[key], anchor)
            assert near < distance < far, (
                f"{key} at {distance:.3f} escapes the "
                f"({near:.3f}, {far:.3f}) gap"
            )


def test_objects_keep_their_distance_from_each_other():
    """Placements respect the spacing, reference included."""
    for anchor, layout in _sample(2000):
        spots = [anchor, *layout.values()]
        for i, first in enumerate(spots):
            for second in spots[i + 1 :]:
                assert math.dist(first, second) >= 0.03


def test_spacing_pushes_objects_apart_that_would_otherwise_collide():
    """Objects sharing a distance band are held apart by the spacing.

    The shipped bands are far enough apart that objects rarely come
    near each other anyway, so this puts all four on one ring where the
    spacing is the only thing keeping them off each other.
    """
    crowded = {
        f"obj_{i}": PolarPlacement(radius=(0.18, 0.20)) for i in range(4)
    }
    solver = PolarLayout(
        workspace_x=(0.0, 1.2),
        workspace_y=(-0.6, 0.6),
        axis_jitter_deg=40.0,
        min_separation=0.12,
    )
    rng = random.Random(1)
    closest = 1e9
    for _ in range(400):
        layout = solver.solve((0.55, 0.0), crowded, rng=rng)
        spots = list(layout.values())
        for i, first in enumerate(spots):
            for second in spots[i + 1 :]:
                closest = min(closest, math.dist(first, second))
    assert closest >= 0.12, f"two objects came within {closest:.3f}m"


def test_spacing_keeps_objects_off_the_reference_itself():
    """The reference is an obstacle too, not just a origin to measure from.

    A near object can be placed close enough to overlap it, so the
    reference has to be counted among the things a placement avoids.
    """
    rng = random.Random(2)
    anchor = (0.55, 0.0)
    solver = PolarLayout(
        workspace_x=(0.0, 1.2),
        workspace_y=(-0.6, 0.6),
        axis_jitter_deg=40.0,
        min_separation=0.10,
    )
    closest = 1e9
    for _ in range(400):
        layout = solver.solve(
            anchor,
            {"pick_near": PolarPlacement(radius=(0.06, 0.14))},
            rng=rng,
        )
        closest = min(closest, math.dist(layout["pick_near"], anchor))
    assert closest >= 0.10, (
        f"an object came within {closest:.3f}m of the reference despite "
        "the 0.10m spacing"
    )


def test_every_placement_lands_inside_the_workspace():
    """Nothing is put where the arm cannot reach it."""
    for _, layout in _sample(2000):
        for key, (x, y) in layout.items():
            assert 0.25 <= x <= 0.80, f"{key} x={x:.3f}"
            assert -0.40 <= y <= 0.40, f"{key} y={y:.3f}"


def test_the_target_turns_up_in_every_direction():
    """No direction is a reliable guess for where the target is.

    Were the near object always to the east, a policy could score by
    reaching east and never read the word "near" at all.
    """
    buckets = [0] * 8
    total = 0
    for anchor, layout in _sample(4000):
        buckets[int(_bearing(layout["pick_near"], anchor) // 45)] += 1
        total += 1

    share = [count / total for count in buckets]
    assert min(share) > 0.08, f"a direction is starved: {share}"
    assert max(share) < 0.18, f"a direction is favoured: {share}"


def test_the_two_candidates_are_not_always_the_same_angle_apart():
    """Their relative angle is no more of a giveaway than their bearing.

    Dealing the axes afresh each episode is what buys this. Keeping the
    assignment fixed and merely rotating the whole arrangement would
    leave the two objects a half-turn apart every time, and a policy
    could find the far one by looking opposite the near one.
    """
    opposite = 0
    total = 0
    for anchor, layout in _sample(4000):
        separation = _separation(
            _bearing(layout["pick_near"], anchor),
            _bearing(layout["pick_far"], anchor),
        )
        opposite += separation >= 150.0
        total += 1

    assert opposite / total < 0.40, (
        f"{opposite / total:.0%} of episodes place the candidates "
        "roughly opposite each other; the axis assignment is not being "
        "redealt"
    )


def test_the_same_seed_arranges_the_scene_the_same_way():
    """A replayed episode has to reproduce its arrangement."""
    first = list(_sample(20, seed=7))
    second = list(_sample(20, seed=7))
    assert first == second


def test_an_impossible_workspace_is_reported_not_ignored():
    """Nothing fits, and the caller hears about it.

    Leaving the objects where the last episode put them would pair a
    fresh instruction with a stale scene, which no downstream check
    would catch.
    """
    anchor = (0.55, 0.0)
    pinhole = PolarLayout(
        workspace_x=(0.54, 0.56),
        workspace_y=(-0.01, 0.01),
        axis_jitter_deg=40.0,
        min_separation=0.03,
    )
    with pytest.raises(PolarLayoutError, match="Could not place"):
        pinhole.solve(anchor, _PLACEMENTS, rng=random.Random(0))


def test_the_failure_says_which_object_could_not_be_placed():
    """The message points at the cause without extra instrumentation."""
    with pytest.raises(PolarLayoutError) as excinfo:
        _LAYOUT.solve(
            (0.55, 0.0),
            {"pick_far": PolarPlacement(radius=(2.0, 2.5))},
            rng=random.Random(0),
        )
    message = str(excinfo.value)
    assert "pick_far" in message
    assert "(2.0, 2.5)" in message
    assert "workspace" in message


def test_redealing_the_axes_rescues_an_arrangement_one_axis_cannot():
    """The outer retry earns its keep.

    A workspace open on one side only leaves some axes unusable. Trying
    harder on the same axis cannot help -- the direction is what is
    wrong -- so the solver has to be able to deal the object elsewhere.
    Comparing one redeal against the full budget isolates that effect
    from how generous the budget happens to be.
    """
    anchor = (0.55, 0.0)
    one_sided = dict(
        workspace_x=(0.30, 0.80),
        workspace_y=(-0.05, 0.40),  # room to the left, none to the right
        axis_jitter_deg=40.0,
        min_separation=0.03,
    )
    placements = {
        "pick_near": PolarPlacement(radius=_NEAR),
        "pick_far": PolarPlacement(radius=_FAR),
    }

    def success_rate(solver: PolarLayout) -> float:
        solved = 0
        for seed in range(200):
            try:
                solver.solve(anchor, placements, rng=random.Random(seed))
            except PolarLayoutError:
                continue
            solved += 1
        return solved / 200

    single = success_rate(PolarLayout(shuffle_retries=1, **one_sided))
    full = success_rate(PolarLayout(**one_sided))
    assert single < 0.5, (
        f"one deal already succeeds {single:.0%} of the time; this "
        "workspace is not constraining enough to show the redeal working"
    )
    assert full > 0.85, (
        f"redealing lifts the rate only to {full:.0%}; an object whose "
        "axis points out of the workspace is not being moved elsewhere"
    )


def test_a_pinned_axis_holds_its_direction_every_episode():
    """A direction that is the point of the description cannot wander.

    An object called "to the left" has to be to the left every time,
    which is the opposite of what the redeal does for the objects whose
    description talks about distance instead.
    """
    directions = {
        "pick_front": (0, lambda dx, dy: dx > 0.05),
        "pick_left": (90, lambda dx, dy: dy > 0.05),
        "pick_behind": (180, lambda dx, dy: dx < -0.05),
        "pick_right": (270, lambda dx, dy: dy < -0.05),
    }
    placements = {
        key: PolarPlacement(radius=(0.15, 0.30), axis=axis)
        for key, (axis, _) in directions.items()
    }

    rng = random.Random(0)
    for _ in range(1500):
        anchor = (rng.uniform(0.50, 0.60), rng.uniform(-0.04, 0.04))
        layout = _LAYOUT.solve(anchor, placements, rng=rng)
        for key, (_, holds) in directions.items():
            dx = layout[key][0] - anchor[0]
            dy = layout[key][1] - anchor[1]
            assert holds(dx, dy), (
                f"{key} came out at dx={dx:+.3f} dy={dy:+.3f}, which does "
                "not read as the direction it was pinned to"
            )


def test_pinning_one_object_still_leaves_the_others_roaming():
    """Pinning is per object, not a mode the whole scene switches into.

    A scene can name one object by direction and another by distance;
    the second must still move around, or its direction becomes the
    easier thing to read.
    """
    placements = {
        "pick_left": PolarPlacement(radius=(0.15, 0.30), axis=90),
        "pick_near": PolarPlacement(radius=(0.11, 0.16)),
        "pick_far": PolarPlacement(radius=(0.30, 0.38)),
    }

    rng = random.Random(1)
    bearings = set()
    for _ in range(1500):
        anchor = (rng.uniform(0.50, 0.60), rng.uniform(-0.04, 0.04))
        layout = _LAYOUT.solve(anchor, placements, rng=rng)
        assert layout["pick_left"][1] - anchor[1] > 0.05
        bearings.add(int(_bearing(layout["pick_near"], anchor) // 45))

    assert len(bearings) >= 5, (
        f"the free object only ever appeared in {sorted(bearings)}; "
        "pinning one object should not pin the rest"
    )


def test_two_objects_cannot_claim_the_same_direction():
    """Both on one axis would put them on top of each other."""
    with pytest.raises(ValueError, match="cannot share a direction"):
        _LAYOUT.solve(
            (0.55, 0.0),
            {
                "pick_left": PolarPlacement(radius=(0.15, 0.3), axis=90),
                "other_left": PolarPlacement(radius=(0.2, 0.3), axis=90),
            },
        )


def test_an_axis_off_the_grid_is_rejected():
    """Axes are the four the arrangement is built from, not any angle."""
    with pytest.raises(ValueError, match="axis must be one of"):
        PolarPlacement(radius=(0.15, 0.3), axis=45)


def test_more_objects_than_axes_is_rejected():
    """Five objects cannot each have one of four axes."""
    with pytest.raises(ValueError, match="will not fit"):
        _LAYOUT.solve(
            (0.55, 0.0),
            {f"obj_{i}": PolarPlacement(radius=(0.15, 0.2)) for i in range(5)},
        )


def test_jitter_wide_enough_to_reach_the_next_axis_is_rejected():
    """Overlapping arcs would let two objects swap sides."""
    with pytest.raises(ValueError, match="neighbouring axis"):
        PolarLayout(
            workspace_x=(0.25, 0.80),
            workspace_y=(-0.40, 0.40),
            axis_jitter_deg=45.0,
            min_separation=0.03,
        )


def test_an_empty_arrangement_is_rejected():
    """There is nothing to solve without objects."""
    with pytest.raises(ValueError, match="at least one"):
        _LAYOUT.solve((0.55, 0.0), {})


@pytest.mark.parametrize("radius", [(0.0, 0.2), (-0.1, 0.2), (0.3, 0.2)])
def test_a_nonsensical_distance_band_is_rejected(radius):
    """Bands have to be positive and ordered."""
    with pytest.raises(ValueError):
        PolarPlacement(radius=radius)


def test_retry_budgets_are_settings_not_scattered():
    """Both retry layers are dials with sane defaults, not buried numbers."""
    defaults = inspect.signature(PolarLayout).parameters
    assert defaults["axis_retries"].default > 1
    assert defaults["shuffle_retries"].default > 1
