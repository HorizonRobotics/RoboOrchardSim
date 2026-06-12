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

import torch

from robo_orchard_sim.utils.env_utils import (
    SettleTracker,
    scene_is_stationary,
)


class _Data:
    def __init__(self, rs):
        self.root_state_w = rs


class _Asset:
    def __init__(self, rs, usd="<unknown>"):
        self.data = _Data(rs)
        self.cfg = type(
            "C", (), {"spawn": type("S", (), {"usd_path": usd})()}
        )()


class FakeScene(dict):
    pass


def _state(lin=0.0, ang=0.0, num_envs=1):
    # root_state_w layout: pos(3) quat(4) lin(3) ang(3) = 13
    rs = torch.zeros(num_envs, 13)
    rs[:, 7] = lin  # linear velocity on x
    rs[:, 10] = ang  # angular velocity on x
    return rs


def test_scene_is_stationary_all_below_threshold_returns_true():
    scene = FakeScene(obj=_Asset(_state(0.001, 0.01)))
    assert scene_is_stationary(scene) is True


def test_scene_is_stationary_one_above_ang_returns_false():
    scene = FakeScene(obj=_Asset(_state(0.001, 0.5)))
    assert scene_is_stationary(scene) is False


def test_scene_is_stationary_return_movers_includes_name_and_usd():
    scene = FakeScene(obj=_Asset(_state(0.001, 0.5), usd="/p/x.usd"))
    stationary, movers = scene_is_stationary(scene, return_movers=True)
    assert stationary is False
    assert movers[0][0] == "obj"
    assert movers[0][1] == "/p/x.usd"


def test_scene_is_stationary_empty_scene_returns_false():
    assert scene_is_stationary(FakeScene()) is False


def test_scene_is_stationary_no_root_state_assets_returns_false():
    scene = FakeScene(light=_Asset(torch.zeros(1, 7)))  # < 13 wide
    assert scene_is_stationary(scene) is False


def test_scene_is_stationary_multi_env_one_moving_returns_false():
    rs = torch.zeros(2, 13)
    rs[1, 10] = 0.5
    scene = FakeScene(obj=_Asset(rs))
    assert scene_is_stationary(scene) is False


def test_settle_tracker_consecutive_stationary_returns_settled():
    t = SettleTracker(streak=3)
    still = FakeScene(obj=_Asset(_state(0.0, 0.0)))
    assert t.update(still) is False  # 1
    assert t.update(still) is False  # 2
    assert t.update(still) is True  # 3 -> settled
    assert t.consecutive == 3
    assert t.settled is True


def test_settle_tracker_motion_resets_consecutive():
    t = SettleTracker(streak=3)
    still = FakeScene(obj=_Asset(_state(0.0, 0.0)))
    moving = FakeScene(obj=_Asset(_state(0.0, 0.5)))
    t.update(still)
    t.update(still)
    assert t.consecutive == 2
    t.update(moving)
    assert t.consecutive == 0
    assert t.settled is False


def test_settle_tracker_alternating_frames_never_returns_settled():
    # one still frame then a moving frame, repeatedly: the bug regression.
    t = SettleTracker(streak=3)
    still = FakeScene(obj=_Asset(_state(0.0, 0.0)))
    moving = FakeScene(obj=_Asset(_state(0.0, 0.5)))
    settled_ever = False
    for _ in range(20):
        settled_ever = settled_ever or t.update(still)
        settled_ever = settled_ever or t.update(moving)
    assert settled_ever is False


def test_settle_tracker_late_settle_returns_settled_at_n_plus_streak():
    t = SettleTracker(streak=3)
    moving = FakeScene(obj=_Asset(_state(0.0, 0.5)))
    still = FakeScene(obj=_Asset(_state(0.0, 0.0)))
    for _ in range(5):
        assert t.update(moving) is False
    assert t.update(still) is False  # 1
    assert t.update(still) is False  # 2
    assert t.update(still) is True  # 3 -> settled


def test_settle_tracker_reset_clears_consecutive():
    t = SettleTracker(streak=2)
    still = FakeScene(obj=_Asset(_state(0.0, 0.0)))
    t.update(still)
    t.reset()
    assert t.consecutive == 0
    assert t.update(still) is False  # only 1 after reset
