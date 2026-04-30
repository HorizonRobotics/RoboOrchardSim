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

"""Unit tests for ``JointStateResetTerm`` using mocked articulations.

These tests bypass the heavy ``__init__`` path and inject fake
articulation/env objects via ``SimpleNamespace`` so behavior can be
verified without launching Isaac.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from robo_orchard_core.envs.manager_based_env import ResetEvent

from robo_orchard_sim.envs.managers.events.joint_state_reset import (
    JointStateResetTerm,
    JointStateResetTermCfg,
)


class _FakeArticulationData:
    def __init__(
        self,
        default_joint_pos: torch.Tensor,
        default_joint_vel: torch.Tensor,
        soft_joint_pos_limits: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel
        self.soft_joint_pos_limits = soft_joint_pos_limits
        self.joint_pos = joint_pos
        self.joint_vel = joint_vel


class _FakeArticulation:
    def __init__(
        self,
        name: str,
        joint_names: list[str],
        default_joint_pos: torch.Tensor,
        soft_joint_pos_limits: torch.Tensor,
    ) -> None:
        self.name = name
        self.joint_names = list(joint_names)
        num_envs, num_joints = default_joint_pos.shape
        self.num_instances = num_envs
        self.data = _FakeArticulationData(
            default_joint_pos=default_joint_pos,
            default_joint_vel=torch.zeros_like(default_joint_pos),
            soft_joint_pos_limits=soft_joint_pos_limits,
            joint_pos=torch.zeros_like(default_joint_pos),
            joint_vel=torch.zeros_like(default_joint_pos),
        )
        self.write_joint_state_calls: list[dict[str, Any]] = []
        self.set_joint_position_target_calls: list[dict[str, Any]] = []
        self.set_joint_velocity_target_calls: list[dict[str, Any]] = []
        self.write_data_to_sim_calls = 0

    def write_joint_state_to_sim(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        self.data.joint_pos[env_ids] = joint_pos.clone()
        self.data.joint_vel[env_ids] = joint_vel.clone()
        self.write_joint_state_calls.append(
            {
                "joint_pos": joint_pos.clone(),
                "joint_vel": joint_vel.clone(),
                "env_ids": env_ids.clone(),
            }
        )

    def set_joint_position_target(
        self,
        joint_pos: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        self.set_joint_position_target_calls.append(
            {
                "joint_pos": joint_pos.clone(),
                "env_ids": env_ids.clone(),
            }
        )

    def write_data_to_sim(self) -> None:
        self.write_data_to_sim_calls += 1


def _make_articulation(
    *,
    num_envs: int = 2,
    joint_names: list[str] | None = None,
    default_pos_value: float = 0.5,
    limit_low: float = -2.0,
    limit_high: float = 2.0,
    name: str = "robot",
) -> _FakeArticulation:
    joint_names = joint_names or [f"joint{i}" for i in range(4)]
    num_joints = len(joint_names)
    default_pos = torch.full(
        (num_envs, num_joints), default_pos_value, dtype=torch.float32
    )
    limits = torch.zeros(num_envs, num_joints, 2, dtype=torch.float32)
    limits[..., 0] = limit_low
    limits[..., 1] = limit_high
    return _FakeArticulation(
        name=name,
        joint_names=joint_names,
        default_joint_pos=default_pos,
        soft_joint_pos_limits=limits,
    )


def _make_term(
    articulations: list[_FakeArticulation],
    *,
    noise_std: float = 0.0,
    per_joint_noise_std: dict[str, float] | None = None,
    noise_excluded_joint_names: list[str] | None = None,
    init_joint_pos: dict[str, float] | None = None,
    clamp_to_joint_limits: bool = True,
    write_joint_state: bool = True,
    write_joint_position_target: bool = True,
    reset_joint_velocity_to_default: bool = True,
) -> JointStateResetTerm:
    """Build a term instance with mocked env/articulations.

    Bypasses the heavyweight ``EventTermBase`` initialization and wires
    in fakes directly so the term can be exercised without launching the
    simulator.
    """

    term = object.__new__(JointStateResetTerm)
    term._cfg = SimpleNamespace(  # noqa: SLF001
        asset_cfgs=None,
        noise_std=noise_std,
        per_joint_noise_std=per_joint_noise_std,
        noise_excluded_joint_names=noise_excluded_joint_names,
        init_joint_pos=init_joint_pos,
        clamp_to_joint_limits=clamp_to_joint_limits,
        write_joint_state=write_joint_state,
        write_joint_position_target=write_joint_position_target,
        reset_joint_velocity_to_default=reset_joint_velocity_to_default,
    )
    term._env = SimpleNamespace(  # noqa: SLF001
        device=torch.device("cpu"),
        num_envs=articulations[0].num_instances,
        scene=SimpleNamespace(
            articulations={a.name: a for a in articulations},
        ),
    )
    term._articulations = list(articulations)  # noqa: SLF001
    term._asset_display_names = [a.name for a in articulations]  # noqa: SLF001
    term._std_vectors = [  # noqa: SLF001
        term._build_std_vector(a, a.name) for a in articulations
    ]
    term._center_overrides = [  # noqa: SLF001
        term._build_init_pos_override(a, a.name) for a in articulations
    ]
    return term


class TestJointStateResetTermCfg:
    def test_default_cfg_has_zero_noise_and_writes_state_and_target(self):
        cfg = JointStateResetTermCfg(trigger_topic="reset")

        assert cfg.class_type is JointStateResetTerm
        assert cfg.noise_std == 0.0
        assert cfg.write_joint_state is True
        assert cfg.write_joint_position_target is True
        assert cfg.clamp_to_joint_limits is True
        assert cfg.reset_joint_velocity_to_default is True

    def test_default_cfg_has_no_init_joint_pos_override(self):
        cfg = JointStateResetTermCfg(trigger_topic="reset")

        assert cfg.init_joint_pos is None


class TestJointStateResetTermZeroNoise:
    def test_zero_noise_writes_default_joint_pos_unchanged(self):
        articulation = _make_articulation()
        term = _make_term([articulation], noise_std=0.0)

        term(ResetEvent(seed=None, env_ids=None))

        assert len(articulation.write_joint_state_calls) == 1
        call = articulation.write_joint_state_calls[0]
        torch.testing.assert_close(
            call["joint_pos"], articulation.data.default_joint_pos
        )
        torch.testing.assert_close(
            call["joint_vel"], articulation.data.default_joint_vel
        )

    def test_zero_noise_writes_default_pos_to_position_target(self):
        articulation = _make_articulation()
        term = _make_term([articulation], noise_std=0.0)

        term(ResetEvent(seed=None, env_ids=None))

        assert len(articulation.set_joint_position_target_calls) == 1
        target_call = articulation.set_joint_position_target_calls[0]
        torch.testing.assert_close(
            target_call["joint_pos"], articulation.data.default_joint_pos
        )
        assert articulation.write_data_to_sim_calls == 1

    def test_zero_noise_prints_joint_pos_before_and_after_reset(self, capsys):
        articulation = _make_articulation()
        term = _make_term([articulation], noise_std=0.0)

        term(ResetEvent(seed=None, env_ids=None))

        captured = capsys.readouterr()
        assert "joint_pos before reset" in captured.out
        assert "joint_pos after reset" in captured.out


class TestJointStateResetTermNoiseMask:
    def test_excluded_joints_match_default_pos_exactly(self):
        joint_names = [f"joint{i}" for i in range(6)]
        articulation = _make_articulation(joint_names=joint_names)
        excluded = ["joint2", "joint5"]

        term = _make_term(
            [articulation],
            noise_std=1.0,
            noise_excluded_joint_names=excluded,
        )
        torch.manual_seed(0)
        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        excluded_indices = [joint_names.index(n) for n in excluded]
        torch.testing.assert_close(
            call["joint_pos"][:, excluded_indices],
            articulation.data.default_joint_pos[:, excluded_indices],
        )

    def test_per_joint_noise_std_zero_overrides_global_noise(self):
        joint_names = ["a", "b", "c"]
        articulation = _make_articulation(joint_names=joint_names)

        term = _make_term(
            [articulation],
            noise_std=1.0,
            per_joint_noise_std={"b": 0.0},
        )
        torch.manual_seed(0)
        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        torch.testing.assert_close(
            call["joint_pos"][:, 1], articulation.data.default_joint_pos[:, 1]
        )


class TestJointStateResetTermClamping:
    def test_clamp_to_limits_keeps_joint_pos_within_soft_limits(self):
        articulation = _make_articulation(
            limit_low=-0.1, limit_high=0.1, default_pos_value=0.0
        )

        term = _make_term(
            [articulation],
            noise_std=10.0,
            clamp_to_joint_limits=True,
        )
        torch.manual_seed(42)
        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        assert torch.all(call["joint_pos"] >= -0.1)
        assert torch.all(call["joint_pos"] <= 0.1)

    def test_no_clamp_allows_joint_pos_outside_soft_limits(self):
        articulation = _make_articulation(
            limit_low=-0.1, limit_high=0.1, default_pos_value=0.0
        )

        term = _make_term(
            [articulation],
            noise_std=10.0,
            clamp_to_joint_limits=False,
        )
        torch.manual_seed(42)
        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        assert torch.any(call["joint_pos"].abs() > 0.1)


class TestJointStateResetTermWriteSwitches:
    def test_write_target_disabled_skips_set_joint_position_target(self):
        articulation = _make_articulation()
        term = _make_term(
            [articulation],
            noise_std=0.0,
            write_joint_position_target=False,
        )

        term(ResetEvent(seed=None, env_ids=None))

        assert articulation.set_joint_position_target_calls == []
        assert articulation.write_data_to_sim_calls == 0

    def test_write_state_disabled_skips_write_joint_state(self):
        articulation = _make_articulation()
        term = _make_term(
            [articulation],
            noise_std=0.0,
            write_joint_state=False,
        )

        term(ResetEvent(seed=None, env_ids=None))

        assert articulation.write_joint_state_calls == []


class TestJointStateResetTermPartialEnvIds:
    def test_partial_env_ids_only_updates_those_envs(self):
        articulation = _make_articulation(num_envs=4)
        term = _make_term([articulation], noise_std=0.0)

        term(ResetEvent(seed=None, env_ids=[1, 3]))

        call = articulation.write_joint_state_calls[0]
        torch.testing.assert_close(
            call["env_ids"], torch.tensor([1, 3], dtype=torch.long)
        )
        assert call["joint_pos"].shape == (2, len(articulation.joint_names))


class TestJointStateResetTermStdVectorValidation:
    def test_unknown_excluded_joint_name_raises_value_error(self):
        articulation = _make_articulation(joint_names=["a", "b", "c"])

        with pytest.raises(ValueError, match="unknown"):
            _make_term(
                [articulation],
                noise_std=0.05,
                noise_excluded_joint_names=["nonexistent"],
            )

    def test_unknown_per_joint_name_raises_value_error(self):
        articulation = _make_articulation(joint_names=["a", "b", "c"])

        with pytest.raises(ValueError, match="unknown"):
            _make_term(
                [articulation],
                noise_std=0.05,
                per_joint_noise_std={"nonexistent": 0.1},
            )

    def test_negative_global_noise_std_raises_value_error(self):
        articulation = _make_articulation()

        with pytest.raises(ValueError, match="non-negative"):
            _make_term([articulation], noise_std=-0.1)

    def test_negative_per_joint_noise_std_raises_value_error(self):
        articulation = _make_articulation(joint_names=["a", "b"])

        with pytest.raises(ValueError, match="non-negative"):
            _make_term(
                [articulation],
                noise_std=0.05,
                per_joint_noise_std={"a": -0.1},
            )


class TestJointStateResetTermInitJointPos:
    def test_dict_override_zero_noise_writes_override_for_listed_joints(self):
        joint_names = ["a", "b", "c"]
        articulation = _make_articulation(
            joint_names=joint_names, default_pos_value=0.5
        )
        term = _make_term(
            [articulation],
            noise_std=0.0,
            init_joint_pos={"b": 0.7},
        )

        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        torch.testing.assert_close(
            call["joint_pos"][:, joint_names.index("b")],
            torch.full(
                (articulation.num_instances,), 0.7, dtype=torch.float32
            ),
        )

    def test_dict_override_unlisted_joints_match_articulation_default(self):
        joint_names = ["a", "b", "c"]
        articulation = _make_articulation(
            joint_names=joint_names, default_pos_value=0.5
        )
        term = _make_term(
            [articulation],
            noise_std=0.0,
            init_joint_pos={"b": 0.7},
        )

        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        unlisted_indices = [
            joint_names.index("a"),
            joint_names.index("c"),
        ]
        torch.testing.assert_close(
            call["joint_pos"][:, unlisted_indices],
            articulation.data.default_joint_pos[:, unlisted_indices],
        )

    def test_dict_override_unknown_joint_raises_value_error(self):
        articulation = _make_articulation(joint_names=["a", "b", "c"])

        with pytest.raises(ValueError, match="unknown"):
            _make_term(
                [articulation],
                noise_std=0.0,
                init_joint_pos={"nonexistent": 0.0},
            )

    def test_excluded_joint_with_override_locks_to_override_value(self):
        joint_names = ["a", "b", "c"]
        articulation = _make_articulation(
            joint_names=joint_names,
            default_pos_value=0.0,
            limit_low=-10.0,
            limit_high=10.0,
        )
        term = _make_term(
            [articulation],
            noise_std=1.0,
            init_joint_pos={"b": 0.5},
            noise_excluded_joint_names=["b"],
        )
        torch.manual_seed(0)

        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        torch.testing.assert_close(
            call["joint_pos"][:, joint_names.index("b")],
            torch.full(
                (articulation.num_instances,), 0.5, dtype=torch.float32
            ),
        )

    def test_dict_override_with_noise_centers_distribution_around_override(
        self,
    ):
        joint_names = ["a", "b"]
        articulation = _make_articulation(
            num_envs=512,
            joint_names=joint_names,
            default_pos_value=0.0,
            limit_low=-100.0,
            limit_high=100.0,
        )
        term = _make_term(
            [articulation],
            noise_std=1.0,
            init_joint_pos={"a": 5.0},
            clamp_to_joint_limits=False,
        )
        torch.manual_seed(123)

        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        sample_mean_a = call["joint_pos"][:, joint_names.index("a")].mean()
        sample_mean_b = call["joint_pos"][:, joint_names.index("b")].mean()
        assert abs(sample_mean_a.item() - 5.0) < 0.3
        assert abs(sample_mean_b.item() - 0.0) < 0.3

    def test_override_clamped_when_outside_soft_joint_limits(self):
        joint_names = ["a"]
        articulation = _make_articulation(
            joint_names=joint_names,
            default_pos_value=0.0,
            limit_low=-0.1,
            limit_high=0.1,
        )
        term = _make_term(
            [articulation],
            noise_std=0.0,
            init_joint_pos={"a": 5.0},
            clamp_to_joint_limits=True,
        )

        term(ResetEvent(seed=None, env_ids=None))

        call = articulation.write_joint_state_calls[0]
        torch.testing.assert_close(
            call["joint_pos"],
            torch.full_like(call["joint_pos"], 0.1),
        )
