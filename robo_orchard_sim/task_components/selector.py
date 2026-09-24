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

"""Picking which candidate a role points at for one episode."""

from __future__ import annotations
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from random import Random
from typing import Literal, TypeVar

T = TypeVar("T")
SelectorName = Literal["round_robin", "random"]


def role_selector_seed(seed: int, role_id: str) -> int:
    """Derive a stable independent random seed for one scene role."""
    if type(seed) is not int:
        raise ValueError("selector seed must be an integer")
    payload = f"{seed}:{role_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


class Selector(ABC):
    """Select original candidate references by episode index."""

    def select(
        self, candidates: Sequence[T], k: int, episode_idx: int
    ) -> list[T]:
        """Take distinct entries cyclically within the indexed round."""
        n = len(candidates)
        if type(k) is not int or not 0 <= k <= n:
            raise ValueError(f"Cannot draw {k} candidates from a pool of {n}.")
        if type(episode_idx) is not int or episode_idx < 0:
            raise ValueError("episode_idx must be a non-negative integer")
        if k == 0 or n == 1:
            return list(candidates[:k])
        order = self._candidate_order(n, episode_idx)
        return [candidates[order[(episode_idx + i) % n]] for i in range(k)]

    @abstractmethod
    def _candidate_order(self, n: int, episode_idx: int) -> Sequence[int]:
        """Return a permutation of candidate indices for this round."""


class RoundRobinSelector(Selector):
    """Cycle through candidates in their original order by episode index."""

    def _candidate_order(self, n: int, episode_idx: int) -> Sequence[int]:
        return range(n)


class RandomSelector(Selector):
    """Index seeded shuffled rounds independently of query order.

    Every round visits all candidates. With multiple candidates, the first
    entry of a new round differs from the last entry of the previous round.
    Replaying the local generator makes retries and skipped indices stable.
    """

    def __init__(self, seed: int) -> None:
        if type(seed) is not int:
            raise ValueError("selector seed must be an integer")
        self._seed = seed

    def _candidate_order(self, n: int, episode_idx: int) -> Sequence[int]:
        rng = Random(self._seed)
        previous_last = None
        for _ in range(episode_idx // n + 1):
            order = list(range(n))
            rng.shuffle(order)
            if order[0] == previous_last:
                other = rng.randrange(1, n)
                order[0], order[other] = order[other], order[0]
            previous_last = order[-1]
        return order


def create_selector(name: str, *, seed: int, role_id: str) -> Selector:
    """Build a configured selector with independent randomness per role."""
    if name == "random":
        return RandomSelector(role_selector_seed(seed, role_id))
    if name == "round_robin":
        return RoundRobinSelector()
    raise ValueError(f"Unknown selector: {name!r}")
