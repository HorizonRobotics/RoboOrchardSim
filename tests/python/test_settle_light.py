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

import math

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


def _pose_state(
    qw=1.0,
    qx=0.0,
    qy=0.0,
    qz=0.0,
    px=0.0,
    py=0.0,
    pz=0.0,
    lin=0.0,
    ang=0.0,
    num_envs=1,
):
    rs = torch.zeros(num_envs, 13)
    rs[:, 0], rs[:, 1], rs[:, 2] = px, py, pz
    rs[:, 3], rs[:, 4], rs[:, 5], rs[:, 6] = qw, qx, qy, qz
    rs[:, 7] = lin
    rs[:, 10] = ang
    return rs


def _z_rot_state(theta_deg, **kwargs):
    half = math.radians(theta_deg) / 2.0
    return _pose_state(qw=math.cos(half), qz=math.sin(half), **kwargs)


def test_settle_tracker_frozen_pose_with_ghost_velocity_returns_settled():
    t = SettleTracker(streak=3)
    buzzing = FakeScene(obj=_Asset(_pose_state(ang=5.0)))
    assert t.update(buzzing) is False
    assert t.update(buzzing) is False
    assert t.update(buzzing) is False
    assert t.update(buzzing) is True
    assert t.settled is True


def test_settle_tracker_rocking_quat_oscillation_never_returns_settled():
    t = SettleTracker(streak=3)
    settled_ever = False
    for _ in range(12):
        settled_ever = settled_ever or t.update(
            FakeScene(obj=_Asset(_z_rot_state(1.0)))
        )
        settled_ever = settled_ever or t.update(
            FakeScene(obj=_Asset(_z_rot_state(-1.0)))
        )
    assert settled_ever is False


def test_settle_tracker_slow_quat_creep_never_returns_settled():
    t = SettleTracker(streak=5, rot_eps_deg=0.2)
    settled_ever = False
    for i in range(30):
        scene = FakeScene(obj=_Asset(_z_rot_state(0.06 * i)))
        settled_ever = settled_ever or t.update(scene)
    assert settled_ever is False


def test_settle_tracker_zero_mean_quat_noise_returns_settled():
    t = SettleTracker(streak=3)
    settled_ever = False
    for i in range(10):
        theta = 0.08 if i % 2 == 0 else -0.08
        scene = FakeScene(obj=_Asset(_z_rot_state(theta)))
        settled_ever = settled_ever or t.update(scene)
    assert settled_ever is True


def test_settle_tracker_position_step_resets_then_returns_settled():
    t = SettleTracker(streak=3)
    still = FakeScene(obj=_Asset(_pose_state()))
    t.update(still)
    t.update(still)
    t.update(still)
    t.update(still)
    assert t.settled is True
    stepped = FakeScene(obj=_Asset(_pose_state(pz=0.002)))
    assert t.update(stepped) is False
    assert t.consecutive == 0
    assert t.update(stepped) is False
    assert t.update(stepped) is False
    assert t.update(stepped) is True


def test_settle_tracker_motion_then_frozen_returns_settled_at_n_plus_streak():
    n, streak = 5, 3
    t = SettleTracker(streak=streak)
    settled_at = None
    for i in range(1, 20):
        theta = 10.0 * min(i, n)
        scene = FakeScene(obj=_Asset(_z_rot_state(theta)))
        if t.update(scene) and settled_at is None:
            settled_at = i
    assert settled_at == n + streak


def test_settle_tracker_reset_clears_anchors_and_counter():
    t = SettleTracker(streak=2)
    still = FakeScene(obj=_Asset(_pose_state()))
    t.update(still)
    t.update(still)
    t.update(still)
    assert t.settled is True
    t.reset()
    assert t.consecutive == 0
    assert t.settled is False
    assert t.update(still) is False
    assert t.update(still) is False
    assert t.update(still) is True


def test_settle_tracker_empty_scene_never_returns_settled():
    t = SettleTracker(streak=2)
    for _ in range(10):
        assert t.update(FakeScene()) is False
    assert t.consecutive == 0
    assert t.settled is False


def test_settle_tracker_breach_reports_offending_asset_and_offsets():
    t = SettleTracker(streak=3)
    t.update(
        FakeScene(
            rock=_Asset(_z_rot_state(0.0)),
            still=_Asset(_pose_state()),
        )
    )
    t.update(
        FakeScene(
            rock=_Asset(_z_rot_state(1.0)),
            still=_Asset(_pose_state()),
        )
    )
    breaches = t.last_breaches
    assert [b[0] for b in breaches] == ["rock"]
    name, rot_deg, pos_mm = breaches[0]
    assert 0.8 <= rot_deg < 1.2
    assert abs(pos_mm) < 1e-6


def test_settle_tracker_clean_frame_reports_no_breaches():
    t = SettleTracker(streak=3)
    still = FakeScene(obj=_Asset(_pose_state()))
    t.update(still)
    t.update(still)
    assert t.last_breaches == []


def test_settle_tracker_reset_clears_breaches():
    t = SettleTracker(streak=3)
    t.update(FakeScene(obj=_Asset(_z_rot_state(0.0))))
    t.update(FakeScene(obj=_Asset(_z_rot_state(1.0))))
    assert t.last_breaches != []
    t.reset()
    assert t.last_breaches == []


def test_settle_tracker_multi_env_one_rocking_never_returns_settled():
    t = SettleTracker(streak=3)
    settled_ever = False
    for i in range(20):
        rs = _pose_state(num_envs=2)
        half = math.radians(1.0 if i % 2 == 0 else -1.0) / 2.0
        rs[1, 3] = math.cos(half)
        rs[1, 6] = math.sin(half)
        settled_ever = settled_ever or t.update(FakeScene(obj=_Asset(rs)))
    assert settled_ever is False
