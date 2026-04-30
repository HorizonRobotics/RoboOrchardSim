# Project RoboOrchard
#
# Copyright (c) 2024 Horizon Robotics. All Rights Reserved.
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

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from google.protobuf.timestamp_pb2 import Timestamp

from robo_orchard_sim.envs.manager_based_env import IsaacManagerBasedEnv
from robo_orchard_sim.envs.managers.record import (
    NoOpRecordControllerCfg,
    RecordControlDecision,
    RecordController,
    RecordControllerCfg,
    RecordManager,
    RecordManagerCfg,
    RecordTermBase,
    RecordTermBaseCfg,
    StationaryEpisodeRecordControllerCfg,
)
from robo_orchard_sim.utils.config import ClassType_co


class _FakeTensor:
    def __init__(self, value: list[int]):
        self._value = value

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> list[int]:
        return self._value


@dataclass
class _RecordedCall:
    data: dict[str, Any]
    timestamp: Timestamp


class _StubRecordTerm(
    RecordTermBase[Any, "_StubRecordTermCfg", dict[str, list]]
):
    instances: list["_StubRecordTerm"] = []

    def __init__(self, cfg: "_StubRecordTermCfg", env: Any):
        super().__init__(cfg, env)
        self.calls: list[_RecordedCall] = []
        type(self).instances.append(self)

    def __call__(
        self, data: dict[str, Any | dict[str, Any]], ts: Timestamp
    ) -> dict[str, list]:
        self.calls.append(
            _RecordedCall(
                data=self._parse_data_from_dict(data, self.cfg.key),
                timestamp=ts,
            )
        )
        return {}

    def reset(self, env_ids=None) -> None:
        return None


class _StubRecordTermCfg(RecordTermBaseCfg[_StubRecordTerm]):
    class_type: ClassType_co[_StubRecordTerm] = _StubRecordTerm

    key: str


class _StartOnPostResetController(RecordController):
    def on_post_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision(start=True)


class _StartOnPostResetControllerCfg(RecordControllerCfg):
    class_type: ClassType_co[_StartOnPostResetController] = (
        _StartOnPostResetController
    )


class _StartStopController(RecordController):
    def on_post_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision(start=True)

    def on_pre_reset(
        self, obs: dict[str, Any] | None
    ) -> RecordControlDecision:
        return RecordControlDecision(stop=True)


class _StartStopControllerCfg(RecordControllerCfg):
    class_type: ClassType_co[_StartStopController] = _StartStopController


@pytest.fixture(autouse=True)
def _clear_stub_terms() -> None:
    _StubRecordTerm.instances.clear()
    yield
    _StubRecordTerm.instances.clear()


@pytest.fixture(autouse=True)
def _disable_mcap_writer_io(monkeypatch) -> None:
    def _fake_start(self) -> None:
        self._running = True

    def _fake_end(self) -> None:
        self._writer = None
        self._file = None
        self._running = False

    monkeypatch.setattr(
        "robo_orchard_sim.envs.managers.record.record_manager."
        "McapRecorder.start",
        _fake_start,
    )
    monkeypatch.setattr(
        "robo_orchard_sim.envs.managers.record.record_manager."
        "McapRecorder.end",
        _fake_end,
    )


def _make_manager_cfg(
    tmp_path,
    *,
    terms: dict[str, _StubRecordTermCfg],
    controller: RecordControllerCfg | None = None,
) -> RecordManagerCfg[_StubRecordTermCfg]:
    return RecordManagerCfg(
        file_path=str(tmp_path),
        terms=terms,
        controller=controller or NoOpRecordControllerCfg(),
    )


def _make_env_stub(step_dt: float = 0.1):
    return type(
        "EnvStub",
        (),
        {"num_envs": 1, "step_dt": step_dt, "step_count": 0, "scene": {}},
    )()


class _StubAssetData:
    def __init__(self, root_state_w: torch.Tensor | None = None):
        if root_state_w is not None:
            self.root_state_w = root_state_w


class _StubAsset:
    def __init__(self, root_state_w: torch.Tensor | None = None):
        self.data = _StubAssetData(root_state_w=root_state_w)


def _make_root_state(
    *,
    lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    root_state_w = torch.zeros((1, 13), dtype=torch.float32)
    root_state_w[:, 3] = 1.0
    root_state_w[:, 7:10] = torch.tensor([lin_vel], dtype=torch.float32)
    root_state_w[:, 10:13] = torch.tensor([ang_vel], dtype=torch.float32)
    return root_state_w


def _make_pose_only_root_state() -> torch.Tensor:
    root_state_w = torch.zeros((1, 7), dtype=torch.float32)
    root_state_w[:, 3] = 1.0
    return root_state_w


def _make_manager_with_single_term(
    tmp_path,
    *,
    term_cfg: _StubRecordTermCfg,
    controller: RecordControllerCfg | None = None,
    step_dt: float = 0.1,
) -> tuple[RecordManager, Any, _StubRecordTerm]:
    env = _make_env_stub(step_dt=step_dt)
    manager = RecordManager(
        _make_manager_cfg(
            tmp_path,
            terms={"term": term_cfg},
            controller=controller,
        ),
        env,
    )
    [term] = _StubRecordTerm.instances
    return manager, env, term


class _StepRecordManagerSpy:
    def __init__(self):
        self.record_step_calls: list[dict[str, Any]] = []

    def record_step(self, obs: dict[str, Any]) -> None:
        self.record_step_calls.append(obs)


class _ResetRecordManagerSpy:
    def __init__(self):
        self.pre_reset_calls = 0
        self.post_reset_calls: list[tuple[dict[str, Any], dt.datetime]] = []

    def record_pre_reset(self) -> None:
        self.pre_reset_calls += 1

    def record_post_reset(
        self, obs: dict[str, Any], cur_time: dt.datetime
    ) -> None:
        self.post_reset_calls.append((obs, cur_time))


def _make_env_step_stub(
    observations: dict[str, Any],
    *,
    record_manager: _StepRecordManagerSpy,
) -> IsaacManagerBasedEnv:
    env = object.__new__(IsaacManagerBasedEnv)
    env.cfg = type(
        "Cfg",
        (),
        {
            "decimation": 0,
            "apply_action_when_no_action": False,
            "sim": type("SimCfg", (), {"dt": 0.1, "render_interval": 1})(),
        },
    )()
    env.sim = type(
        "Sim",
        (),
        {
            "has_gui": lambda self: False,
            "has_rtx_sensors": lambda self: False,
        },
    )()
    env.observation_manager = type(
        "ObservationManager",
        (),
        {"get_observations": lambda self: observations},
    )()
    env.record_manager = record_manager
    env.extras = {"ok": True}
    env._step_count = 0
    env._sim_step_counter = 0
    env._is_closed = True
    return env


def _make_env_reset_stub(
    observations: dict[str, Any],
    *,
    record_manager: _ResetRecordManagerSpy,
) -> IsaacManagerBasedEnv:
    env = object.__new__(IsaacManagerBasedEnv)
    env.record_manager = record_manager
    env.event_manager = type(
        "EventManager",
        (),
        {"notify": lambda self, event_name, payload: None},
    )()
    env.observation_manager = type(
        "ObservationManager",
        (),
        {"get_observations": lambda self: observations},
    )()
    env.extras = {}
    env._step_count = 3
    env._is_closed = True
    env.RESET = ("reset", lambda **kwargs: kwargs)
    return env


class TestRecordManagerLifecycle:
    @pytest.mark.parametrize(
        ("step_count", "expected_offset"),
        [
            pytest.param(1, 0.1, id="first-step"),
            pytest.param(2, 0.2, id="later-step"),
        ],
    )
    def test_record_step_after_post_reset_uses_episode_time_base_timestamp(
        self, tmp_path, step_count, expected_offset
    ):
        start_time = dt.datetime(2026, 1, 1, 12, 0, 0)
        manager, env, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
            controller=_StartOnPostResetControllerCfg(),
        )

        manager.record_post_reset({"obs": {"value": 0}}, start_time)
        env.step_count = step_count
        manager.record_step({"obs": {"value": step_count}})

        assert [call.timestamp.ToDatetime() for call in term.calls] == [
            start_time + dt.timedelta(seconds=expected_offset)
        ]

    def test_record_step_before_controller_start_does_not_record(
        self, tmp_path
    ):
        manager, _, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
        )

        manager.record_step({"obs": {"value": 1}})
        manager.close()

        assert term.calls == []
        assert manager.running is False
        assert not list(tmp_path.rglob("*.mcap"))

    def test_record_step_with_step_user_data_consumes_it_once(self, tmp_path):
        manager, _, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="step/custom",
            ),
            controller=_StartOnPostResetControllerCfg(),
        )

        manager.record_post_reset(
            {"obs": {"value": 3}},
            dt.datetime(2026, 1, 1, 12, 0, 0),
        )
        manager.set_step_user_data({"custom": "once"})
        manager.record_step({"obs": {"value": 3}})
        manager.record_step({"obs": {"value": 3}})

        assert [call.data for call in term.calls] == [
            {"step/custom": "once"},
        ]

    def test_record_pre_reset_with_episode_user_data_records_it(
        self, tmp_path
    ):
        manager, _, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/pre_reset",
                fps=1.0,
                key="episode/custom",
                record_mode="pre_reset",
            ),
            controller=_StartStopControllerCfg(),
        )

        manager.record_post_reset(
            {"obs": {"value": 5}},
            dt.datetime(2026, 1, 1, 12, 0, 0),
        )
        manager.update_episode_user_data({"custom": "episode"})
        manager.record_pre_reset()

        assert [call.data for call in term.calls] == [
            {"episode/custom": "episode"},
        ]

    def test_stationary_controller_before_assets_are_static_waits_to_record(
        self, tmp_path
    ):
        start_time = dt.datetime(2026, 1, 1, 12, 0, 0)
        manager, env, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                min_wait_step=0,
                max_wait_step=100,
            ),
        )
        env.scene = {
            "moving": _StubAsset(
                root_state_w=_make_root_state(
                    lin_vel=(0.03, 0.0, 0.0),
                    ang_vel=(0.11, 0.0, 0.0),
                )
            )
        }

        manager.record_post_reset({"obs": {"value": 0}}, start_time)

        env.step_count = 1
        manager.record_step({"obs": {"value": 1}})
        assert term.calls == []

        env.scene["moving"].data.root_state_w = _make_root_state(
            lin_vel=(0.01, 0.0, 0.0),
            ang_vel=(0.09, 0.0, 0.0),
        )
        env.step_count = 2
        manager.record_step({"obs": {"value": 2}})

        assert [call.data for call in term.calls] == [{"obs/value": 2}]
        assert manager.running is True

        manager.record_pre_reset()
        assert manager.running is False

    def test_stationary_controller_ignores_assets_without_motion_state(
        self,
        tmp_path,
    ):
        manager, env, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                min_wait_step=0,
                max_wait_step=100,
            ),
        )
        env.scene = {
            "camera": _StubAsset(),
            "object": _StubAsset(
                root_state_w=_make_root_state(
                    lin_vel=(0.01, 0.0, 0.0),
                    ang_vel=(0.09, 0.0, 0.0),
                )
            ),
        }

        manager.record_post_reset(
            {"obs": {"value": 0}},
            dt.datetime(2026, 1, 1, 12, 0, 0),
        )
        env.step_count = 1
        manager.record_step({"obs": {"value": 1}})

        assert [call.data for call in term.calls] == [{"obs/value": 1}]

    def test_stationary_controller_before_min_wait_step_waits_to_record(
        self, tmp_path
    ):
        manager, env, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                max_wait_step=100,
            ),
        )
        env.scene = {
            "object": _StubAsset(
                root_state_w=_make_root_state(
                    lin_vel=(0.0, 0.0, 0.0),
                    ang_vel=(0.0, 0.0, 0.0),
                )
            )
        }

        manager.record_post_reset(
            {"obs": {"value": 0}},
            dt.datetime(2026, 1, 1, 12, 0, 0),
        )
        for step in range(1, 10):
            env.step_count = step
            manager.record_step({"obs": {"value": step}})

        assert term.calls == []

        env.step_count = 10
        manager.record_step({"obs": {"value": 10}})

        assert [call.data for call in term.calls] == [{"obs/value": 10}]

    def test_stationary_controller_at_max_wait_step_starts_recording(
        self, tmp_path
    ):
        manager, env, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                min_wait_step=0,
                max_wait_step=3,
            ),
        )
        env.scene = {
            "moving": _StubAsset(
                root_state_w=_make_root_state(
                    lin_vel=(0.03, 0.0, 0.0),
                    ang_vel=(0.11, 0.0, 0.0),
                )
            )
        }

        manager.record_post_reset(
            {"obs": {"value": 0}},
            dt.datetime(2026, 1, 1, 12, 0, 0),
        )
        for step in (1, 2):
            env.step_count = step
            manager.record_step({"obs": {"value": step}})

        assert term.calls == []

        env.step_count = 3
        manager.record_step({"obs": {"value": 3}})

        assert [call.data for call in term.calls] == [{"obs/value": 3}]

    def test_stationary_controller_without_motion_assets_waits_until_max_step(
        self, tmp_path
    ):
        manager, env, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                min_wait_step=0,
                max_wait_step=2,
            ),
        )
        env.scene = {"camera": _StubAsset()}

        manager.record_post_reset(
            {"obs": {"value": 0}},
            dt.datetime(2026, 1, 1, 12, 0, 0),
        )
        env.step_count = 1
        manager.record_step({"obs": {"value": 1}})
        assert term.calls == []

        env.step_count = 2
        manager.record_step({"obs": {"value": 2}})

        assert [call.data for call in term.calls] == [{"obs/value": 2}]

    def test_stationary_controller_with_none_scene_entries_ignores_them(
        self, tmp_path
    ):
        manager, env, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                min_wait_step=0,
                max_wait_step=100,
            ),
        )
        env.scene = {
            "terrain": None,
            "object": _StubAsset(
                root_state_w=_make_root_state(
                    lin_vel=(0.01, 0.0, 0.0),
                    ang_vel=(0.09, 0.0, 0.0),
                )
            ),
        }

        manager.record_post_reset(
            {"obs": {"value": 0}},
            dt.datetime(2026, 1, 1, 12, 0, 0),
        )
        env.step_count = 1
        manager.record_step({"obs": {"value": 1}})

        assert [call.data for call in term.calls] == [{"obs/value": 1}]

    def test_stationary_controller_with_pose_only_scene_entries_ignores_them(
        self, tmp_path
    ):
        manager, env, term = _make_manager_with_single_term(
            tmp_path,
            term_cfg=_StubRecordTermCfg(
                topic="/step",
                fps=10.0,
                key="obs/value",
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                min_wait_step=0,
                max_wait_step=100,
            ),
        )
        env.scene = {
            "pose_only": _StubAsset(root_state_w=_make_pose_only_root_state()),
            "object": _StubAsset(
                root_state_w=_make_root_state(
                    lin_vel=(0.01, 0.0, 0.0),
                    ang_vel=(0.09, 0.0, 0.0),
                )
            ),
        }

        manager.record_post_reset(
            {"obs": {"value": 0}},
            dt.datetime(2026, 1, 1, 12, 0, 0),
        )
        env.step_count = 1
        manager.record_step({"obs": {"value": 1}})

        assert [call.data for call in term.calls] == [{"obs/value": 1}]


class TestEnvRecordHooks:
    def test_env_step_without_action_increments_step_count(self):
        observations = {"policy": {"obs": _FakeTensor([1])}}
        env = _make_env_step_stub(
            observations,
            record_manager=_StepRecordManagerSpy(),
        )

        IsaacManagerBasedEnv.step(env)

        assert env.step_count == 1

    def test_env_step_without_action_returns_current_observations(self):
        observations = {"policy": {"obs": _FakeTensor([1])}}
        env = _make_env_step_stub(
            observations,
            record_manager=_StepRecordManagerSpy(),
        )

        ret = IsaacManagerBasedEnv.step(env)

        assert ret.observations == observations

    def test_env_step_without_action_forwards_observations_to_record_manager(
        self,
    ):
        observations = {"policy": {"obs": _FakeTensor([1])}}
        record_manager = _StepRecordManagerSpy()
        env = _make_env_step_stub(
            observations,
            record_manager=record_manager,
        )

        IsaacManagerBasedEnv.step(env)

        assert record_manager.record_step_calls == [observations]

    def test_env_reset_with_seed_and_env_ids_resets_step_count(
        self, monkeypatch
    ):
        observations = {"policy": {"obs": 1}}
        env = _make_env_reset_stub(
            observations,
            record_manager=_ResetRecordManagerSpy(),
        )

        def _fake_reset(self, seed=None, env_ids=None):
            return None

        monkeypatch.setattr(
            "robo_orchard_sim.envs.manager_based_env.IsaacEnv.reset",
            _fake_reset,
        )

        IsaacManagerBasedEnv.reset(env, seed=1, env_ids=[0])

        assert env.step_count == 0

    def test_env_reset_with_seed_and_env_ids_returns_current_observations(
        self, monkeypatch
    ):
        observations = {"policy": {"obs": 1}}
        env = _make_env_reset_stub(
            observations,
            record_manager=_ResetRecordManagerSpy(),
        )

        def _fake_reset(self, seed=None, env_ids=None):
            return None

        monkeypatch.setattr(
            "robo_orchard_sim.envs.manager_based_env.IsaacEnv.reset",
            _fake_reset,
        )

        ret = IsaacManagerBasedEnv.reset(env, seed=1, env_ids=[0])

        assert ret.observations == observations

    def test_env_reset_with_seed_and_env_ids_calls_record_pre_reset(
        self, monkeypatch
    ):
        observations = {"policy": {"obs": 1}}
        record_manager = _ResetRecordManagerSpy()
        env = _make_env_reset_stub(
            observations,
            record_manager=record_manager,
        )

        def _fake_reset(self, seed=None, env_ids=None):
            return None

        monkeypatch.setattr(
            "robo_orchard_sim.envs.manager_based_env.IsaacEnv.reset",
            _fake_reset,
        )

        IsaacManagerBasedEnv.reset(env, seed=1, env_ids=[0])

        assert record_manager.pre_reset_calls == 1

    def test_env_reset_with_seed_and_env_ids_calls_record_post_reset(
        self, monkeypatch
    ):
        observations = {"policy": {"obs": 1}}
        record_manager = _ResetRecordManagerSpy()
        env = _make_env_reset_stub(
            observations,
            record_manager=record_manager,
        )

        def _fake_reset(self, seed=None, env_ids=None):
            return None

        monkeypatch.setattr(
            "robo_orchard_sim.envs.manager_based_env.IsaacEnv.reset",
            _fake_reset,
        )

        IsaacManagerBasedEnv.reset(env, seed=1, env_ids=[0])

        assert [obs for obs, _ in record_manager.post_reset_calls] == [
            observations
        ]
