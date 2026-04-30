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

"""Evaluator implementation with explicit per-step episode loop."""

from __future__ import annotations
import os
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from robo_orchard_core.envs.env_base import EnvStepReturn
from robo_orchard_core.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType_co,
    Config,
)

from robo_orchard_sim.evaluator.base import EpisodeResult, EvaluationResult
from robo_orchard_sim.tasks.validators.base import (
    Validator,
    ValidatorActor,
    ValidatorOutput,
)

if TYPE_CHECKING:
    from robo_orchard_sim.envs.env_base import IsaacEnvContextManager
    from robo_orchard_sim.envs.manager_based_env import IsaacManagerBasedEnv
    from robo_orchard_sim.launcher import SimpleIsaacAppLauncher
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

__all__ = [
    "Evaluator",
    "EvaluatorCfg",
    "LaunchConfig",
]


def _get_task_builder():
    from robo_orchard_sim.task_suite import build_task

    return build_task


def _create_asset_registry(asset_root: str):
    from robo_orchard_sim.asset_manager.registry import AssetRegistry

    return AssetRegistry(asset_root)


def _create_asset_resolver(*, registry_obj: Any, seed: int):
    """Create the run-scoped resolver used for task assembly.

    The resolver RNG is seeded once per evaluator run. Asset identity
    selection therefore stays fixed for the run, while per-episode seeds
    passed to ``env.reset(...)`` only affect runtime reset randomness such as
    pose variation.
    """
    from robo_orchard_sim.asset_manager.resolver import AssetResolver

    return AssetResolver(
        registry=registry_obj,
        splits=None,  # TODO: support splits
        rng=np.random.default_rng(seed),
    )


def _get_isaac_env_context_manager_cls():
    from robo_orchard_sim.envs.env_base import IsaacEnvContextManager

    return IsaacEnvContextManager


def _create_launcher(**kwargs: Any):
    from robo_orchard_sim.launcher import SimpleIsaacAppLauncher

    return SimpleIsaacAppLauncher(**kwargs)


def _close_launcher(launcher: "SimpleIsaacAppLauncher") -> None:
    close = getattr(launcher, "close", None)
    if callable(close):
        close()
        return

    destructor = getattr(launcher, "__del__", None)
    if callable(destructor):
        destructor()


class LaunchConfig(Config):
    """Launcher options for evaluator-owned Isaac app startup."""

    headless: bool = True
    enable_cameras: bool = True
    virtual_display: bool = False


class Evaluator:
    """Evaluator that runs fixed-number episodes with explicit step loops."""

    InitFromConfig: bool = True

    cfg: "EvaluatorCfg"

    def __init__(self, cfg: "EvaluatorCfg") -> None:
        self.cfg = cfg
        self._launcher: SimpleIsaacAppLauncher | None = None
        self._env_cm: IsaacEnvContextManager | None = None
        self._env: IsaacManagerBasedEnv | None = None
        self._task: OrchardEnv | None = None

    def __enter__(self) -> "Evaluator":
        self._ensure_env()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_type, exc_val, exc_tb
        self.close()

    def close(self) -> None:
        self._close_env()
        if self._launcher is None:
            return

        _close_launcher(self._launcher)
        self._launcher = None

    def evaluate(
        self,
        policy_or_cfg: PolicyMixin | PolicyConfig,
    ) -> EvaluationResult:
        """Run evaluation episodes for one policy instance.

        Args:
            policy_or_cfg (PolicyMixin | PolicyConfig):
                A ready policy instance or a policy config that can build one.

        Returns:
            EvaluationResult: Aggregated episode evaluation statistics.
        """
        env = self._ensure_env()
        policy = self._normalize_policy(policy_or_cfg)

        episode_results = []
        for episode_idx in range(self.cfg.episode_num):
            seed = self.cfg.seed + episode_idx
            result = self._run_episode(env=env, policy=policy, seed=seed)
            episode_results.append(result)

        success_count = sum(1 for x in episode_results if x.success)
        average_progress = (
            sum(x.progress for x in episode_results) / len(episode_results)
            if episode_results
            else 0.0
        )
        success_rate = (
            success_count / len(episode_results) if episode_results else 0.0
        )
        return EvaluationResult(
            episode_num=self.cfg.episode_num,
            seed_start=self.cfg.seed,
            success_rate=success_rate,
            average_progress=average_progress,
            episode_results=episode_results,
        )

    def _ensure_launcher(self) -> SimpleIsaacAppLauncher:
        if self._launcher is not None:
            return self._launcher

        self._launcher = _create_launcher(
            headless=self.cfg.launch.headless,
            enable_cameras=self.cfg.launch.enable_cameras,
            virtual_display=self.cfg.launch.virtual_display,
        )
        return self._launcher

    def _ensure_env(self) -> IsaacManagerBasedEnv:
        if self._env is not None:
            return self._env
        return self._open_env()

    def _build_task_from_cfg(self) -> OrchardEnv:
        """Build the run-scoped orchard task from evaluator configuration.

        This task is assembled once per evaluator-owned environment. Asset
        selection happens through the resolver seeded by ``self.cfg.seed``,
        so asset identity is fixed within one evaluator run. Per-episode
        seeds only flow into ``env.reset(seed=...)`` and affect reset-time
        randomness after task assembly.
        """
        config_path = self.cfg.task_config_path
        if config_path is not None:
            config_path = os.path.abspath(config_path)
            if not os.path.isfile(config_path):
                raise FileNotFoundError(
                    f"task_config_path does not exist: {config_path}"
                )

        registry_obj = _create_asset_registry(self.cfg.asset_root)
        resolver = _create_asset_resolver(
            registry_obj=registry_obj,
            seed=self.cfg.seed,
        )
        task_builder = _get_task_builder()
        task = task_builder(
            self.cfg.task_name,
            resolver=resolver,
            config_path=config_path,
        )
        if self.cfg.enable_recording:
            task.configure_recording(file_path=self.cfg.record_dir)
        return task

    def _open_env(
        self,
        task: OrchardEnv | None = None,
    ) -> IsaacManagerBasedEnv:
        self._ensure_launcher()

        if task is None:
            task = self._build_task_from_cfg()
        self._task = task

        env_cfg = task.to_isaac_env_cfg()
        env_context_manager_cls = _get_isaac_env_context_manager_cls()
        self._env_cm = env_context_manager_cls(
            env_cfg,
            with_new_stage=True,
            disable_exit_on_stop=True,
        )
        self._env = self._env_cm.__enter__()
        return self._env

    def _close_env(self) -> None:
        if self._env_cm is None:
            self._task = None
            self._env = None
            return

        self._env_cm.__exit__(None, None, None)
        self._env_cm = None
        self._env = None
        self._task = None

    def _reload_env(
        self,
        task: OrchardEnv | None = None,
    ) -> IsaacManagerBasedEnv:
        self._close_env()
        return self._open_env(task=task)

    def _get_runtime_task(self) -> Any:
        """Return the task bound to the current evaluator environment."""
        if self._task is None:
            self._ensure_env()
        assert self._task is not None
        return self._task.task

    def _build_validator_actors(
        self,
        scene: Any,
    ) -> list[ValidatorActor]:
        """Build validator actor snapshots from the runtime scene."""
        actor_names = self._get_runtime_task().get_validator_actor_names()
        return [
            ValidatorActor.from_rigid_object(name, scene[name])
            for name in actor_names
        ]

    def _capture_init_state(
        self,
        scene: Any,
        actors: list[ValidatorActor],
    ) -> None:
        """Capture initial actor states from the runtime scene."""
        for actor in actors:
            actor.capture_init_state(scene[actor.name])

    def _capture_final_state(
        self,
        scene: Any,
        actors: list[ValidatorActor],
    ) -> None:
        """Capture final actor states from the runtime scene."""
        for actor in actors:
            actor.capture_final_state(scene[actor.name])

    def _build_validator(
        self,
        actors: list[ValidatorActor],
    ) -> Validator:
        return self._get_runtime_task().build_validator(actors=actors)

    def _normalize_policy(
        self,
        policy_or_cfg: PolicyMixin | PolicyConfig,
    ) -> PolicyMixin:
        if isinstance(policy_or_cfg, PolicyMixin):
            return policy_or_cfg
        return policy_or_cfg()

    def _extract_done_flag(self, done: bool | torch.Tensor | None) -> bool:
        if done is None:
            return False
        if isinstance(done, torch.Tensor):
            return bool(done.any().item())
        return done

    def _extract_terminated(self, step_return: EnvStepReturn) -> bool:
        return self._extract_done_flag(step_return.terminated)

    def _extract_truncated(self, step_return: EnvStepReturn) -> bool:
        return self._extract_done_flag(step_return.truncated)

    def _get_non_stationary_entities(
        self,
        env: IsaacManagerBasedEnv,
    ) -> list[str]:
        if not hasattr(env.scene, "keys"):
            return []

        non_stationary = []
        for name in env.scene.keys():
            asset = env.scene[name]
            data = getattr(asset, "data", None)
            root_state_w = getattr(data, "root_state_w", None)
            if not isinstance(root_state_w, torch.Tensor):
                continue
            if root_state_w.shape[-1] < 13:
                continue

            lin_vel = torch.linalg.vector_norm(root_state_w[..., 7:10], dim=-1)
            ang_vel = torch.linalg.vector_norm(
                root_state_w[..., 10:13], dim=-1
            )
            if bool((lin_vel > 0.01).any()) or bool((ang_vel > 0.01).any()):
                non_stationary.append(name)
        return non_stationary

    def _settle_scene(self, env: IsaacManagerBasedEnv) -> EnvStepReturn:
        latest_step_return = env.step()

        if self.cfg.max_settle_steps <= 0:
            return latest_step_return

        for _ in range(self.cfg.max_settle_steps):
            if not self._get_non_stationary_entities(env):
                return latest_step_return
            latest_step_return = env.step()

        non_stationary = self._get_non_stationary_entities(env)
        if non_stationary:
            warnings.warn(
                "Starting evaluation before scene fully settles. "
                "Non-stationary entities: "
                f"{', '.join(sorted(non_stationary))}.",
                UserWarning,
                stacklevel=2,
            )
        return latest_step_return

    def _build_episode_metadata(
        self,
        actors: list[ValidatorActor],
        validator_output: ValidatorOutput,
        instruction_text: str | None,
        *,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        # TODO：user can add meata data here

        if not actors:
            return {}

        meta_data = {
            "init_position": {
                actor.name: actor.init_state[env_idx].tolist()
                for actor in actors
                if actor.init_state is not None
            },
            "final_position": {
                actor.name: actor.final_state[env_idx].tolist()
                for actor in actors
                if actor.final_state is not None
            },
            "actors": {
                actor.name: {
                    "actor_category": actor.category,
                    "actor_type": actor.actor_type,
                    "actor_uuid": actor.uuid,
                }
                for actor in actors
            },
            "task_success": float(validator_output.success),
            "task_progress": float(validator_output.progress),
        }
        if instruction_text is not None:
            meta_data["instruction"] = instruction_text

        return meta_data

    def _build_policy_instruction(
        self,
        env: IsaacManagerBasedEnv,
        *,
        template_seed: int,
        actor_description_seed: int,
    ) -> str | None:
        task = self._get_runtime_task()
        if task.instruction is None:
            return None

        instruction = task.instruction
        actors = task.build_instruction_context(
            env,
            actor_description_seed=actor_description_seed,
        )
        return instruction.render(
            actors=actors,
            template_seed=template_seed,
            actor_description_seed=actor_description_seed,
        )

    def _record_episode_metadata(
        self,
        env: IsaacManagerBasedEnv,
        actors: list[ValidatorActor],
        validator_output: ValidatorOutput,
        instruction_text: str | None,
    ) -> None:
        record_manager = getattr(env, "record_manager", None)
        if record_manager is None:
            return

        num_envs = getattr(env, "num_envs", 1)
        meta_dict: dict[str, Any] | list[dict[str, Any]]
        if num_envs > 1:
            meta_dict = [
                self._build_episode_metadata(
                    actors,
                    validator_output,
                    instruction_text,
                    env_idx=env_idx,
                )
                for env_idx in range(num_envs)
            ]
        else:
            meta_dict = self._build_episode_metadata(
                actors,
                validator_output,
                instruction_text,
            )

        if not meta_dict:
            return

        record_manager.set_episode_user_data({"meta_dict": meta_dict})
        record_manager.record_pre_reset()

    def _prepare_episode(
        self,
        env: IsaacManagerBasedEnv,
        policy: PolicyMixin,
        seed: int,
        *,
        template_seed: int | None = None,
        actor_description_seed: int | None = None,
    ) -> tuple[dict[str, Any], list[ValidatorActor], Validator, str | None]:
        reset_return = env.reset(seed=seed)
        observations = reset_return.observations

        settle_return = self._settle_scene(env)
        observations = settle_return.observations

        if template_seed is None:
            template_seed = seed
        if actor_description_seed is None:
            actor_description_seed = seed
        instruction_text = self._build_policy_instruction(
            env=env,
            template_seed=template_seed,
            actor_description_seed=actor_description_seed,
        )
        if instruction_text is not None:
            print(f"instruction: {instruction_text}")

        actors = self._build_validator_actors(env.scene)
        validator = self._build_validator(actors)
        validator.reset()

        self._capture_init_state(env.scene, actors)
        policy.reset()
        return observations, actors, validator, instruction_text

    def _step_episode(
        self,
        env: IsaacManagerBasedEnv,
        policy: PolicyMixin,
        observations: dict[str, Any],
        validator: Validator,
        instruction_text: str | None,
    ) -> tuple[int, str, ValidatorOutput]:
        stop_reason = "max_steps"
        steps = 0
        validator_output = ValidatorOutput(
            success=False,
            progress=0.0,
            metrics={},
        )
        for step_idx in range(self.cfg.max_steps):
            policy_input = dict(observations)
            policy_input["instruction"] = instruction_text
            action = policy(policy_input)
            step_return = env.step(action)
            observations = step_return.observations
            validator_output = validator.evaluate(env, env_idx=0)
            steps = step_idx + 1

            if validator_output.success:
                stop_reason = "success"
                break
            if self._extract_terminated(step_return):
                stop_reason = "terminated"
                break
            if self._extract_truncated(step_return):
                stop_reason = "truncated"
                break

        return steps, stop_reason, validator_output

    def _run_episode(
        self,
        env: IsaacManagerBasedEnv,
        policy: PolicyMixin,
        seed: int,
        *,
        template_seed: int | None = None,
        actor_description_seed: int | None = None,
    ) -> EpisodeResult:
        observations, actors, validator, instruction_text = (
            self._prepare_episode(
                env=env,
                policy=policy,
                seed=seed,
                template_seed=template_seed,
                actor_description_seed=actor_description_seed,
            )
        )
        steps, stop_reason, validator_output = self._step_episode(
            env=env,
            policy=policy,
            observations=observations,
            validator=validator,
            instruction_text=instruction_text,
        )

        self._capture_final_state(env.scene, actors)
        self._record_episode_metadata(
            env,
            actors,
            validator_output,
            instruction_text,
        )

        return EpisodeResult(
            seed=seed,
            success=validator_output.success,
            progress=validator_output.progress,
            steps=steps,
            stop_reason=stop_reason,
            metrics=validator_output.metrics,
        )


class EvaluatorCfg(ClassConfig):
    class_type: ClassType_co[Evaluator] = Evaluator
    task_name: str
    asset_root: str
    task_config_path: str | None = None
    enable_recording: bool = False
    record_dir: str = "logs/records"
    launch: LaunchConfig = LaunchConfig()
    seed: int = 0
    episode_num: int = 1
    max_steps: int = 1000
    max_settle_steps: int = 50
