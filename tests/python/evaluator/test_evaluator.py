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

from __future__ import annotations
import types
import warnings
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from robo_orchard_core.envs.env_base import EnvStepReturn
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin

from robo_orchard_sim.evaluator import Evaluator, EvaluatorCfg, LaunchConfig
from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv
from robo_orchard_sim.tasks.instructions.base import InstructionActor
from robo_orchard_sim.tasks.validators.base import (
    ValidatorActor,
    ValidatorOutput,
)


@dataclass
class _StepState:
    terminated: bool = False
    truncated: bool = False
    observations: dict[str, int] | None = None


class _StubPolicy(PolicyMixin):
    def __init__(self) -> None:
        self.reset_calls = 0
        self.act_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, observations: dict[str, Any]) -> dict[str, Any]:
        del observations
        self.act_calls += 1
        return {"joint_action": self.act_calls}


class _ConfiguredStubPolicy(_StubPolicy):
    def __init__(
        self,
        cfg: Any = None,
        observation_space: Any = None,
        action_space: Any = None,
    ) -> None:
        del cfg, observation_space, action_space
        super().__init__()


class _ObservationCapturingPolicy(_StubPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.observations_seen: list[dict[str, Any]] = []

    def act(self, observations: dict[str, Any]) -> dict[str, Any]:
        self.observations_seen.append(dict(observations))
        return super().act(observations)


class _StubPolicyCfg(PolicyConfig[_ConfiguredStubPolicy]):
    class_type: type[_ConfiguredStubPolicy] = _ConfiguredStubPolicy
    factory_calls: int = 0

    def __call__(
        self,
        observation_space: Any = None,
        action_space: Any = None,
    ) -> _ConfiguredStubPolicy:
        self.factory_calls += 1
        return super().__call__(
            observation_space=observation_space,
            action_space=action_space,
        )


class _StubEnvCfg:
    def __init__(
        self,
        env: "_StubStepEnv",
        *,
        record_dir: str | None = None,
    ) -> None:
        self._env = env
        self.record_dir = record_dir

    def __call__(self) -> "_StubStepEnv":
        return self._env


class _StubEnvContextManager:
    created: list["_StubEnvContextManager"] = []

    def __init__(
        self,
        cfg: _StubEnvCfg,
        with_new_stage: bool = False,
        disable_exit_on_stop: bool = True,
    ) -> None:
        self.cfg = cfg
        self.with_new_stage = with_new_stage
        self.disable_exit_on_stop = disable_exit_on_stop
        self.enter_calls = 0
        self.exit_calls = 0
        self.env = cfg()
        self.created.append(self)

    def __enter__(self) -> "_StubStepEnv":
        self.enter_calls += 1
        if self.cfg.record_dir is not None:
            self.env.record_manager = _StubRecordManager(
                file_path=self.cfg.record_dir
            )
        return self.env

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_calls += 1


class _StubLauncher:
    created: list["_StubLauncher"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.app = object()
        self.close_calls = 0
        self.created.append(self)

    def close(self) -> None:
        self.close_calls += 1


class _StubStepEnv:
    def __init__(
        self,
        episodes: list[list[_StepState]],
        *,
        num_envs: int = 1,
    ) -> None:
        self._episodes = episodes
        self._episode_index = 0
        self._active_steps: list[_StepState] = []
        self.scene = _StubScene(
            {
                "actor_a": _StubAsset(
                    root_state_w=self._root_state(
                        num_envs=num_envs,
                        pos=self._positions_per_env(
                            num_envs=num_envs,
                            base_pos=(0.0, 0.1, 0.2),
                        ),
                    ),
                    category="apple",
                    actor_type="fruit",
                    uuid="uuid-apple",
                ),
                "actor_b": _StubAsset(
                    root_state_w=self._root_state(
                        num_envs=num_envs,
                        pos=self._positions_per_env(
                            num_envs=num_envs,
                            base_pos=(1.0, 1.1, 1.2),
                        ),
                    ),
                    category="basket",
                    actor_type="container",
                    uuid="uuid-basket",
                ),
            }
        )
        self.current_step = 0
        self.num_envs = num_envs
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[dict[str, Any]] = []
        self.record_manager = None

    def _positions_per_env(
        self,
        *,
        num_envs: int,
        base_pos: tuple[float, float, float],
    ) -> tuple[float, float, float] | list[tuple[float, float, float]]:
        if num_envs == 1:
            return base_pos
        return [
            tuple(coord + env_idx for coord in base_pos)
            for env_idx in range(num_envs)
        ]

    def _root_state(
        self,
        *,
        num_envs: int,
        pos: tuple[float, float, float] | list[tuple[float, float, float]],
    ) -> torch.Tensor:
        state = torch.zeros((num_envs, 13), dtype=torch.float64)
        if isinstance(pos, list):
            state[:, :3] = torch.tensor(pos, dtype=torch.float64)
        else:
            state[:, :3] = torch.tensor(pos, dtype=torch.float64)
        state[:, 3] = 1.0
        return state

    def _set_actor_pose(
        self,
        name: str,
        pos: tuple[float, float, float] | list[tuple[float, float, float]],
    ) -> None:
        self.scene[name].data = _StubAssetData(
            root_state_w=self._root_state(num_envs=self.num_envs, pos=pos)
        )

    def reset(self, **kwargs: Any) -> EnvStepReturn:
        self.reset_calls.append(dict(kwargs))
        self.current_step = 0
        self._active_steps = list(self._episodes[self._episode_index])
        self._episode_index += 1
        return EnvStepReturn(
            observations={"step": 0},
            rewards=None,
            terminated=None,
            truncated=None,
            info={"seed": kwargs.get("seed")},
        )

    def step(self, action: dict[str, Any] | None = None) -> EnvStepReturn:
        if action is None:
            return EnvStepReturn(
                observations={"step": self.current_step},
                rewards=None,
                terminated=False,
                truncated=False,
                info={"step": self.current_step},
            )
        self.step_calls.append(action)
        if not self._active_steps:
            raise RuntimeError("No step states configured for this episode.")

        step_state = self._active_steps.pop(0)
        self.current_step += 1
        self._set_actor_pose(
            "actor_a",
            self._positions_per_env(
                num_envs=self.num_envs,
                base_pos=(0.3, 0.4, 0.5),
            ),
        )
        self._set_actor_pose(
            "actor_b",
            self._positions_per_env(
                num_envs=self.num_envs,
                base_pos=(1.3, 1.4, 1.5),
            ),
        )
        observations = step_state.observations or {"step": self.current_step}
        return EnvStepReturn(
            observations=observations,
            rewards=None,
            terminated=step_state.terminated,
            truncated=step_state.truncated,
            info={"step": self.current_step},
        )

    def close(self) -> None:
        return None


class _StubAssetData:
    def __init__(self, root_state_w: torch.Tensor) -> None:
        self.root_state_w = root_state_w
        self.root_pos_w = root_state_w[:, :3]
        self.root_quat_w = root_state_w[:, 3:7]


class _StubAsset:
    def __init__(
        self,
        root_state_w: torch.Tensor,
        *,
        category: str = "unknown",
        actor_type: str = "unknown",
        uuid: str = "unknown",
    ) -> None:
        self.data = _StubAssetData(root_state_w=root_state_w)
        self.cfg = SimpleNamespace(
            category=category,
            actor_type=actor_type,
            uuid=uuid,
        )


class _StubScene:
    def __init__(self, assets: dict[str, _StubAsset]) -> None:
        self._assets = assets

    def keys(self) -> list[str]:
        return list(self._assets.keys())

    def __getitem__(self, key: str) -> _StubAsset:
        return self._assets[key]


class _StubSettlingEnv(_StubStepEnv):
    def __init__(
        self,
        episodes: list[list[_StepState]],
        settle_states: list[dict[str, torch.Tensor]],
    ) -> None:
        super().__init__(episodes=episodes)
        self._settle_states = settle_states
        self._settle_index = 0
        self.scene = _StubScene(
            {
                name: _StubAsset(root_state_w=state.clone())
                for name, state in settle_states[0].items()
            }
        )
        self.scene._assets["actor_a"] = _StubAsset(
            root_state_w=self._root_state(num_envs=1, pos=(0.0, 0.1, 0.2)),
            category="apple",
            actor_type="fruit",
            uuid="uuid-apple",
        )
        self.scene._assets["actor_b"] = _StubAsset(
            root_state_w=self._root_state(num_envs=1, pos=(1.0, 1.1, 1.2)),
            category="basket",
            actor_type="container",
            uuid="uuid-basket",
        )

    def step(self, action: dict[str, Any] | None = None) -> EnvStepReturn:
        if action is None:
            state = self._settle_states[
                min(self._settle_index, len(self._settle_states) - 1)
            ]
            for name, root_state_w in state.items():
                self.scene[name].data.root_state_w = root_state_w.clone()
            self._settle_index += 1
            return EnvStepReturn(
                observations={"settle_step": self._settle_index},
                rewards=None,
                terminated=False,
                truncated=False,
                info={"settle_step": self._settle_index},
            )

        return super().step(action)


class _StubValidator:
    def __init__(
        self,
        success_step: int,
        actors: list[ValidatorActor],
    ) -> None:
        self.success_step = success_step
        self.reset_calls = 0
        self.actors = list(actors)

    def reset(self) -> None:
        self.reset_calls += 1

    def evaluate(self, env: _StubStepEnv, env_idx: int = 0) -> ValidatorOutput:
        del env_idx
        success = env.current_step >= self.success_step
        progress = min(env.current_step / max(self.success_step, 1), 1.0)
        return ValidatorOutput(
            success=success,
            progress=progress,
            metrics={"current_step": env.current_step},
        )


@dataclass
class _StubInstruction:
    actor_description_mode: str = "seen"
    render_calls: list[dict[str, Any]] = field(default_factory=list)

    def render(
        self,
        actors: dict[str, Any] | None = None,
        template_seed: int | None = None,
        actor_description_seed: int | None = None,
    ) -> str:
        self.render_calls.append(
            {
                "actors": actors,
                "template_seed": template_seed,
                "actor_description_seed": actor_description_seed,
            }
        )
        return (
            "instruction-template-"
            f"{template_seed}-actor-{actor_description_seed}"
        )


@dataclass
class _StubInstructionActor:
    name: str


@dataclass
class _StubTask:
    success_steps: list[int]
    instruction: Any = None
    _validator_index: int = 0
    pick_object: Any = field(
        default_factory=lambda: SimpleNamespace(scene_name="actor_a")
    )
    place_object: Any = field(
        default_factory=lambda: SimpleNamespace(scene_name="actor_b")
    )

    def get_validator_actor_names(self) -> list[str]:
        return ["actor_a", "actor_b"]

    def build_validator(self, actors: list[ValidatorActor]) -> _StubValidator:
        success_step = self.success_steps[self._validator_index]
        self._validator_index += 1
        return _StubValidator(success_step=success_step, actors=actors)

    def build_instruction_context(
        self,
        env: Any,
        *,
        actor_description_seed: int,
    ) -> dict[str, Any]:
        return {
            "actor1": InstructionActor.from_rigid_object(
                env.scene[self.pick_object.scene_name],
                actor_description_mode=self.instruction.actor_description_mode,
                actor_description_seed=actor_description_seed,
            ),
            "actor2": InstructionActor.from_rigid_object(
                env.scene[self.place_object.scene_name],
                actor_description_mode=self.instruction.actor_description_mode,
                actor_description_seed=actor_description_seed,
            ),
        }


@dataclass
class _StubOrchardEnv(OrchardEnv):
    env: _StubStepEnv
    success_steps: list[int]
    task: _StubTask = field(init=False)
    record_dir: str | None = None

    def __post_init__(self) -> None:
        self.task = _StubTask(success_steps=self.success_steps)

    def configure_recording(
        self,
        file_path: str | None = None,
        controller: object | None = None,
    ) -> "_StubOrchardEnv":
        del controller
        self.record_dir = file_path
        return self

    def to_isaac_env_cfg(self) -> _StubEnvCfg:
        return _StubEnvCfg(self.env, record_dir=self.record_dir)


class _StubTaskRegistry:
    def __init__(self, tasks: list[_StubOrchardEnv]) -> None:
        self._tasks = list(tasks)
        self.build_calls: list[str] = []
        self.build_kwargs: list[dict[str, object | None]] = []

    def build_task(
        self,
        task_name: str,
        *,
        resolver: object | None = None,
        config_path: str | None = None,
    ) -> _StubOrchardEnv:
        self.build_calls.append(task_name)
        self.build_kwargs.append(
            {
                "resolver": resolver,
                "config_path": config_path,
            }
        )
        if not self._tasks:
            raise RuntimeError("No stub tasks left to build.")
        return self._tasks.pop(0)


class _StubRecordManager:
    def __init__(self, file_path: str | None = None) -> None:
        self.file_path = file_path
        self.episode_user_data_calls: list[dict[str, Any]] = []
        self.record_pre_reset_calls = 0

    def set_episode_user_data(self, data: dict[str, Any]) -> None:
        self.episode_user_data_calls.append(data)

    def record_pre_reset(self) -> None:
        self.record_pre_reset_calls += 1


class TestEvaluator:
    def patch_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        tasks: list[_StubOrchardEnv],
    ) -> _StubTaskRegistry:
        import robo_orchard_sim.evaluator.evaluator as evaluator_module

        _StubEnvContextManager.created.clear()
        _StubLauncher.created.clear()
        registry = _StubTaskRegistry(tasks=tasks)
        monkeypatch.setattr(
            evaluator_module,
            "_get_isaac_env_context_manager_cls",
            lambda: _StubEnvContextManager,
        )
        monkeypatch.setattr(
            evaluator_module,
            "_create_launcher",
            lambda **kwargs: _StubLauncher(**kwargs),
        )
        monkeypatch.setattr(
            evaluator_module,
            "_close_launcher",
            lambda launcher: launcher.close(),
        )
        monkeypatch.setattr(
            evaluator_module,
            "_create_asset_registry",
            lambda asset_root: f"registry:{asset_root}",
        )
        monkeypatch.setattr(
            evaluator_module,
            "_create_asset_resolver",
            lambda *, registry_obj, seed: types.SimpleNamespace(
                registry=registry_obj,
                splits=None,
                rng=f"rng:{seed}",
            ),
        )
        monkeypatch.setattr(
            evaluator_module,
            "_get_task_builder",
            lambda: registry.build_task,
        )
        return registry

    def build_evaluator(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        episodes: list[list[_StepState]],
        success_steps: list[int],
        episode_num: int,
        max_steps: int,
        seed: int = 0,
    ) -> tuple[Any, _StubStepEnv, _StubTaskRegistry]:
        env = _StubStepEnv(episodes=episodes)
        orchard_env = _StubOrchardEnv(
            env=env,
            success_steps=success_steps,
        )
        registry = self.patch_runtime(
            monkeypatch,
            tasks=[orchard_env],
        )
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            seed=seed,
            episode_num=episode_num,
            max_steps=max_steps,
        )()
        return evaluator, env, registry

    def test_cfg_instantiates_evaluator(self) -> None:
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
        )()

        assert isinstance(evaluator, Evaluator)
        assert evaluator.cfg.task_name == "place_a2b_easy"
        assert isinstance(evaluator.cfg.launch, LaunchConfig)
        assert evaluator.cfg.launch.headless is True
        assert evaluator.cfg.launch.enable_cameras is True
        assert evaluator.cfg.launch.virtual_display is False

    def test_task_runtime_is_built_lazily(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        orchard_env = _StubOrchardEnv(
            env=_StubStepEnv(episodes=[[_StepState()]]),
            success_steps=[1],
        )
        registry = self.patch_runtime(
            monkeypatch,
            tasks=[orchard_env],
        )

        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
        )()

        assert registry.build_calls == []
        assert _StubLauncher.created == []
        assert _StubEnvContextManager.created == []

        evaluator._ensure_env()

        assert registry.build_calls == ["place_a2b_easy"]
        assert _StubLauncher.created[0].kwargs == {
            "headless": True,
            "enable_cameras": True,
            "virtual_display": False,
        }
        assert len(_StubLauncher.created) == 1
        assert len(_StubEnvContextManager.created) == 1

    def test_task_suite_runtime_helpers_import_successfully(self) -> None:
        from robo_orchard_sim import task_suite

        assert callable(task_suite.build_task)
        assert not hasattr(task_suite, "resolve_launch_kwargs")

    def test_policy_cfg_is_instantiated_once_per_evaluate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator, _, _ = self.build_evaluator(
            monkeypatch,
            episodes=[
                [_StepState()],
                [_StepState()],
            ],
            success_steps=[1, 1],
            episode_num=2,
            max_steps=1,
        )
        policy_cfg = _StubPolicyCfg()

        evaluator.evaluate(policy_cfg)

        assert policy_cfg.factory_calls == 1

    def test_policy_is_reset_between_episodes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator, _, _ = self.build_evaluator(
            monkeypatch,
            episodes=[
                [_StepState()],
                [_StepState()],
                [_StepState()],
            ],
            success_steps=[1, 1, 1],
            episode_num=3,
            max_steps=1,
        )
        policy = _StubPolicy()

        evaluator.evaluate(policy)

        assert policy.reset_calls == 3

    def test_episode_stops_on_success_before_env_done(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator, env, _ = self.build_evaluator(
            monkeypatch,
            episodes=[
                [
                    _StepState(),
                    _StepState(),
                    _StepState(),
                ]
            ],
            success_steps=[1],
            episode_num=1,
            max_steps=3,
        )

        result = evaluator.evaluate(_StubPolicy())

        assert result.episode_results[0].stop_reason == "success"
        assert len(env.step_calls) == 1

    @pytest.mark.parametrize(
        ("stop_reason", "terminated", "truncated"),
        [
            ("terminated", True, False),
            ("truncated", False, True),
        ],
    )
    def test_episode_stops_on_terminated_or_truncated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stop_reason: str,
        terminated: bool,
        truncated: bool,
    ) -> None:
        evaluator, _, _ = self.build_evaluator(
            monkeypatch,
            episodes=[
                [_StepState(terminated=terminated, truncated=truncated)]
            ],
            success_steps=[99],
            episode_num=1,
            max_steps=3,
        )

        result = evaluator.evaluate(_StubPolicy())

        assert result.episode_results[0].stop_reason == stop_reason

    def test_evaluator_can_be_reused_for_multiple_evaluate_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator, env, registry = self.build_evaluator(
            monkeypatch,
            episodes=[
                [_StepState()],
                [_StepState()],
            ],
            success_steps=[1, 1],
            episode_num=1,
            max_steps=1,
        )
        policy_a = _StubPolicy()
        policy_b = _StubPolicy()

        first = evaluator.evaluate(policy_a)
        second = evaluator.evaluate(policy_b)

        assert first.episode_num == 1
        assert second.episode_num == 1
        assert policy_a.reset_calls == 1
        assert policy_b.reset_calls == 1
        assert len(env.reset_calls) == 2
        assert registry.build_calls == ["place_a2b_easy"]

    def test_evaluate_uses_evaluator_cfg_to_build_task_behavior(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        import robo_orchard_sim.evaluator.evaluator as evaluator_module

        success_task = _StubOrchardEnv(
            env=_StubStepEnv(episodes=[[_StepState()]]),
            success_steps=[1],
        )
        fallback_task = _StubOrchardEnv(
            env=_StubStepEnv(episodes=[[_StepState()]]),
            success_steps=[99],
        )
        config_path = tmp_path / "task.yaml"
        config_path.write_text("task: place_a2b_easy\n", encoding="utf-8")

        self.patch_runtime(monkeypatch, tasks=[fallback_task])

        def build_task(
            task_name: str,
            *,
            resolver: object | None = None,
            config_path: str | None = None,
        ) -> _StubOrchardEnv:
            if (
                task_name == "place_a2b_easy"
                and config_path == str(config_path_obj)
                and getattr(resolver, "registry", None)
                == "registry:/tmp/assets"
                and getattr(resolver, "rng", None) == "rng:0"
            ):
                return success_task
            return fallback_task

        config_path_obj = config_path.resolve()
        monkeypatch.setattr(
            evaluator_module,
            "_get_task_builder",
            lambda: build_task,
        )
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            task_config_path=str(config_path),
            episode_num=1,
            max_steps=1,
        )()

        result = evaluator.evaluate(_StubPolicy())

        assert result.episode_results[0].stop_reason == "success"

    def test_evaluate_configures_recording_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        explicit_env = _StubStepEnv(episodes=[[_StepState()]])
        explicit_task = _StubOrchardEnv(env=explicit_env, success_steps=[1])
        self.patch_runtime(monkeypatch, tasks=[explicit_task])
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            enable_recording=True,
            record_dir="logs/eval_records",
            episode_num=1,
            max_steps=1,
        )()

        evaluator.evaluate(_StubPolicy())

        assert explicit_env.record_manager is not None
        assert explicit_env.record_manager.file_path == "logs/eval_records"

    def test_episode_waits_for_scene_to_settle_before_evaluation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        moving = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        still = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        env = _StubSettlingEnv(
            episodes=[[_StepState()]],
            settle_states=[
                {"objects/cube": moving},
                {"objects/cube": still},
            ],
        )
        orchard_env = _StubOrchardEnv(env=env, success_steps=[1])
        self.patch_runtime(monkeypatch, tasks=[orchard_env])
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
            max_settle_steps=3,
        )()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = evaluator.evaluate(_StubPolicy())

        assert result.episode_results[0].stop_reason == "success"
        assert env.step_calls == [{"joint_action": 1}]
        assert len(record) == 0

    def test_episode_warns_when_scene_fails_to_settle_in_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        moving = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3, 0.0]]
        )
        env = _StubSettlingEnv(
            episodes=[[_StepState()]],
            settle_states=[
                {"objects/cube": moving, "robots/arm": moving},
                {"objects/cube": moving, "robots/arm": moving},
            ],
        )
        orchard_env = _StubOrchardEnv(env=env, success_steps=[1])
        self.patch_runtime(monkeypatch, tasks=[orchard_env])
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
            max_settle_steps=2,
        )()

        with pytest.warns(UserWarning, match="objects/cube, robots/arm"):
            evaluator.evaluate(_StubPolicy())

    def test_episode_passes_instruction_wrapped_with_observations_to_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        moving = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        still = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        )
        env = _StubSettlingEnv(
            episodes=[[_StepState()]],
            settle_states=[
                {"objects/cube": moving},
                {"objects/cube": still},
            ],
        )
        orchard_env = _StubOrchardEnv(env=env, success_steps=[1])
        orchard_env.task.instruction = _StubInstruction(
            actor_description_mode="seen"
        )
        self.patch_runtime(monkeypatch, tasks=[orchard_env])

        class _FakeInstructionActor:
            def __init__(self, name: str) -> None:
                self.name = name

        def _fake_from_rigid_object(
            rigid_object: Any,
            actor_description_mode: str = "raw",
            actor_description_seed: int | None = None,
        ) -> _FakeInstructionActor:
            del actor_description_mode, actor_description_seed
            return _FakeInstructionActor(rigid_object.cfg.uuid)

        monkeypatch.setattr(
            InstructionActor,
            "from_rigid_object",
            staticmethod(_fake_from_rigid_object),
        )
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
            max_settle_steps=3,
        )()
        policy = _ObservationCapturingPolicy()

        evaluator.evaluate(policy)

        assert policy.observations_seen[0] == {
            "settle_step": 2,
            "instruction": "instruction-template-0-actor-0",
        }

    def test_evaluate_with_instruction_prints_instruction_text(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        env = _StubStepEnv(episodes=[[_StepState()]])
        orchard_env = _StubOrchardEnv(env=env, success_steps=[1])
        orchard_env.task.instruction = _StubInstruction(
            actor_description_mode="seen"
        )
        self.patch_runtime(monkeypatch, tasks=[orchard_env])

        class _FakeInstructionActor:
            def __init__(self, name: str) -> None:
                self.name = name

        def _fake_from_rigid_object(
            rigid_object: Any,
            actor_description_mode: str = "raw",
            actor_description_seed: int | None = None,
        ) -> _FakeInstructionActor:
            del actor_description_mode, actor_description_seed
            return _FakeInstructionActor(rigid_object.cfg.uuid)

        monkeypatch.setattr(
            InstructionActor,
            "from_rigid_object",
            staticmethod(_fake_from_rigid_object),
        )

        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
        )()

        evaluator.evaluate(_StubPolicy())

        captured = capsys.readouterr()
        assert "instruction: instruction-template-0-actor-0" in captured.out

    def test_reload_env_reuses_launcher_and_rebuilds_env_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first_env = _StubStepEnv(episodes=[[_StepState()]])
        second_env = _StubStepEnv(episodes=[[_StepState()]])
        registry = self.patch_runtime(
            monkeypatch,
            tasks=[
                _StubOrchardEnv(env=first_env, success_steps=[1]),
                _StubOrchardEnv(env=second_env, success_steps=[1]),
            ],
        )
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
            launch=LaunchConfig(
                headless=False,
                enable_cameras=False,
                virtual_display=True,
            ),
        )()

        opened_env = evaluator._ensure_env()
        first_context = _StubEnvContextManager.created[-1]
        reloaded_env = evaluator._reload_env()
        second_context = _StubEnvContextManager.created[-1]
        launcher = _StubLauncher.created[0]

        assert opened_env is first_env
        assert reloaded_env is second_env
        assert first_context.exit_calls == 1
        assert second_context.enter_calls == 1
        assert launcher.kwargs == {
            "headless": False,
            "enable_cameras": False,
            "virtual_display": True,
        }
        assert registry.build_calls == ["place_a2b_easy", "place_a2b_easy"]
        assert len(_StubLauncher.created) == 1

        evaluator.close()

        assert second_context.exit_calls == 1
        assert launcher.close_calls == 1

    def test_episode_records_meta_dict_via_episode_user_data(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        evaluator, env, _ = self.build_evaluator(
            monkeypatch,
            episodes=[[_StepState()]],
            success_steps=[1],
            episode_num=1,
            max_steps=1,
            seed=7,
        )
        env.record_manager = _StubRecordManager()

        evaluator.evaluate(_StubPolicy())

        assert env.record_manager.record_pre_reset_calls == 1
        assert env.record_manager.episode_user_data_calls == [
            {
                "meta_dict": {
                    "init_position": {
                        "actor_a": [0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0],
                        "actor_b": [1.0, 1.1, 1.2, 1.0, 0.0, 0.0, 0.0],
                    },
                    "final_position": {
                        "actor_a": [0.3, 0.4, 0.5, 1.0, 0.0, 0.0, 0.0],
                        "actor_b": [1.3, 1.4, 1.5, 1.0, 0.0, 0.0, 0.0],
                    },
                    "actors": {
                        "actor_a": {
                            "actor_category": "apple",
                            "actor_type": "fruit",
                            "actor_uuid": "uuid-apple",
                        },
                        "actor_b": {
                            "actor_category": "basket",
                            "actor_type": "container",
                            "actor_uuid": "uuid-basket",
                        },
                    },
                    "task_success": 1.0,
                    "task_progress": 1.0,
                }
            }
        ]

    def test_episode_records_instruction_rendered_from_rigid_objects(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _StubStepEnv(episodes=[[_StepState()]])
        orchard_env = _StubOrchardEnv(env=env, success_steps=[1])
        orchard_env.task.instruction = _StubInstruction(
            actor_description_mode="seen"
        )
        self.patch_runtime(monkeypatch, tasks=[orchard_env])
        env.record_manager = _StubRecordManager()

        from_rigid_object_calls: list[dict[str, Any]] = []

        class _FakeInstructionActor:
            def __init__(self, name: str) -> None:
                self.name = name

        def _fake_from_rigid_object(
            rigid_object: Any,
            actor_description_mode: str = "raw",
            actor_description_seed: int | None = None,
        ) -> _FakeInstructionActor:
            from_rigid_object_calls.append(
                {
                    "rigid_object": rigid_object,
                    "actor_description_mode": actor_description_mode,
                    "actor_description_seed": actor_description_seed,
                }
            )
            return _FakeInstructionActor(rigid_object.cfg.uuid)

        monkeypatch.setattr(
            InstructionActor,
            "from_rigid_object",
            staticmethod(_fake_from_rigid_object),
        )

        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
            seed=7,
        )()

        evaluator.evaluate(_StubPolicy())

        assert [
            call["actor_description_seed"] for call in from_rigid_object_calls
        ] == [7, 7]
        assert [
            call["actor_description_mode"] for call in from_rigid_object_calls
        ] == [
            "seen",
            "seen",
        ]
        assert (
            env.record_manager.episode_user_data_calls[0]["meta_dict"][
                "instruction"
            ]
            == "instruction-template-7-actor-7"
        )

    def test_evaluate_uses_task_instruction_context_for_rendering(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _StubStepEnv(episodes=[[_StepState()]])
        orchard_env = _StubOrchardEnv(env=env, success_steps=[1])

        class _ContextTask:
            def __init__(self) -> None:
                self.instruction = _StubInstruction(
                    actor_description_mode="seen"
                )
                self.context_calls: list[dict[str, Any]] = []

            def get_validator_actor_names(self) -> list[str]:
                return ["actor_a", "actor_b"]

            def build_validator(
                self,
                actors: list[ValidatorActor],
            ) -> _StubValidator:
                return _StubValidator(success_step=1, actors=actors)

            def build_instruction_context(
                self,
                env: Any,
                *,
                actor_description_seed: int,
            ) -> dict[str, Any]:
                self.context_calls.append(
                    {
                        "env": env,
                        "actor_description_seed": actor_description_seed,
                    }
                )
                return {
                    "actor1": SimpleNamespace(name="pick-actor"),
                    "actor2": SimpleNamespace(name="place-actor"),
                }

        context_task = _ContextTask()
        orchard_env.task = context_task
        self.patch_runtime(monkeypatch, tasks=[orchard_env])
        env.record_manager = _StubRecordManager()

        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
            seed=7,
        )()
        policy = _ObservationCapturingPolicy()

        evaluator.evaluate(policy)

        assert policy.observations_seen[0]["instruction"] == (
            "instruction-template-7-actor-7"
        )
        assert context_task.context_calls == [
            {
                "env": env,
                "actor_description_seed": 7,
            }
        ]
        assert context_task.instruction.render_calls == [
            {
                "actors": {
                    "actor1": SimpleNamespace(name="pick-actor"),
                    "actor2": SimpleNamespace(name="place-actor"),
                },
                "template_seed": 7,
                "actor_description_seed": 7,
            }
        ]
        assert (
            env.record_manager.episode_user_data_calls[0]["meta_dict"][
                "instruction"
            ]
            == "instruction-template-7-actor-7"
        )

    def test_episode_records_distinct_meta_dicts_for_multi_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env = _StubStepEnv(episodes=[[_StepState()]], num_envs=2)
        orchard_env = _StubOrchardEnv(env=env, success_steps=[1])
        self.patch_runtime(monkeypatch, tasks=[orchard_env])
        evaluator = EvaluatorCfg(
            task_name="place_a2b_easy",
            asset_root="/tmp/assets",
            episode_num=1,
            max_steps=1,
            seed=7,
        )()
        env.record_manager = _StubRecordManager()

        evaluator.evaluate(_StubPolicy())

        assert env.record_manager.episode_user_data_calls == [
            {
                "meta_dict": [
                    {
                        "init_position": {
                            "actor_a": [0.0, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0],
                            "actor_b": [1.0, 1.1, 1.2, 1.0, 0.0, 0.0, 0.0],
                        },
                        "final_position": {
                            "actor_a": [0.3, 0.4, 0.5, 1.0, 0.0, 0.0, 0.0],
                            "actor_b": [1.3, 1.4, 1.5, 1.0, 0.0, 0.0, 0.0],
                        },
                        "actors": {
                            "actor_a": {
                                "actor_category": "apple",
                                "actor_type": "fruit",
                                "actor_uuid": "uuid-apple",
                            },
                            "actor_b": {
                                "actor_category": "basket",
                                "actor_type": "container",
                                "actor_uuid": "uuid-basket",
                            },
                        },
                        "task_success": 1.0,
                        "task_progress": 1.0,
                    },
                    {
                        "init_position": {
                            "actor_a": [1.0, 1.1, 1.2, 1.0, 0.0, 0.0, 0.0],
                            "actor_b": [2.0, 2.1, 2.2, 1.0, 0.0, 0.0, 0.0],
                        },
                        "final_position": {
                            "actor_a": [1.3, 1.4, 1.5, 1.0, 0.0, 0.0, 0.0],
                            "actor_b": [2.3, 2.4, 2.5, 1.0, 0.0, 0.0, 0.0],
                        },
                        "actors": {
                            "actor_a": {
                                "actor_category": "apple",
                                "actor_type": "fruit",
                                "actor_uuid": "uuid-apple",
                            },
                            "actor_b": {
                                "actor_category": "basket",
                                "actor_type": "container",
                                "actor_uuid": "uuid-basket",
                            },
                        },
                        "task_success": 1.0,
                        "task_progress": 1.0,
                    },
                ]
            }
        ]
