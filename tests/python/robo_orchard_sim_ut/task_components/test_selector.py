## Copyright (c) 2026 Horizon Robotics. All Rights Reserved.

"""Candidate selection contracts for round-robin and random sampling."""

import random

import pytest

from robo_orchard_sim.task_components.selector import (
    RandomSelector,
    RoundRobinSelector,
)

POOL = ["pick_0", "pick_1", "pick_2", "pick_3", "pick_4"]


def test_each_episode_advances_to_the_next_candidate():
    selector = RoundRobinSelector()

    drawn = [selector.select(POOL, k=1, episode_idx=ep)[0] for ep in range(5)]

    assert drawn == POOL


def test_the_cycle_wraps_around():
    selector = RoundRobinSelector()

    assert selector.select(POOL, k=1, episode_idx=5) == ["pick_0"]
    assert selector.select(POOL, k=1, episode_idx=12) == ["pick_2"]


def test_taking_several_wraps_past_the_end():
    selector = RoundRobinSelector()

    assert selector.select(POOL, k=2, episode_idx=0) == ["pick_0", "pick_1"]
    assert selector.select(POOL, k=2, episode_idx=4) == ["pick_4", "pick_0"]


def test_drawing_more_than_the_pool_holds_is_rejected():
    selector = RoundRobinSelector()

    with pytest.raises(ValueError, match="pool"):
        selector.select(POOL, k=6, episode_idx=0)


def test_random_selector_same_seed_reproduces_draw_sequence():
    selectors = [RandomSelector(seed=17), RandomSelector(seed=17)]
    sequences = [
        [selector.select(POOL, k=2, episode_idx=i) for i in range(10)]
        for selector in selectors
    ]

    assert sequences[0] == sequences[1]


def test_random_selector_repeated_draws_varies_without_round_robin_cycle():
    selector = RandomSelector(seed=0)
    drawn = [selector.select(POOL, k=1, episode_idx=i)[0] for i in range(30)]

    assert 1 < len(set(drawn)) <= len(POOL)
    assert drawn != [POOL[index % len(POOL)] for index in range(len(drawn))]


@pytest.mark.parametrize("k", [1, 3, 5])
def test_random_selector_multiple_candidates_samples_without_replacement(k):
    pool = [object() for _ in range(5)]
    drawn = RandomSelector(seed=7).select(pool, k=k, episode_idx=0)

    assert len(drawn) == len(set(drawn)) == k
    assert set(drawn).issubset(pool)


@pytest.mark.parametrize(
    ("pool", "k", "expected"),
    [([], 0, []), (POOL, 0, []), (["only"], 1, ["only"])],
)
def test_random_selector_small_pool_returns_requested_candidates(
    pool, k, expected
):
    assert RandomSelector(seed=0).select(pool, k=k, episode_idx=0) == expected


@pytest.mark.parametrize(("pool", "k"), [(POOL, -1), (POOL, 6), ([], 1)])
def test_random_selector_invalid_sample_size_raises_value_error(pool, k):
    with pytest.raises(ValueError, match="pool"):
        RandomSelector(seed=0).select(pool, k=k, episode_idx=0)


def test_random_selector_local_generator_preserves_global_random_state():
    state = random.getstate()
    RandomSelector(seed=0).select(POOL, k=3, episode_idx=0)

    assert random.getstate() == state
