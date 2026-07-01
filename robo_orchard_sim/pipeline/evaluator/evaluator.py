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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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

from robo_orchard_sim.contracts.joint_command import UnifiedJointCommand
from robo_orchard_sim.contracts.policy_binding import CanonicalPolicyInput
from robo_orchard_sim.pipeline.evaluator.base import (
    EpisodeResult,
    EvaluationResult,
    SkippedEpisode,
)
from robo_orchard_sim.policy.canonicalizer import (
    canonicalize_observations,
    validate_policy_compatibility,
)
from robo_orchard_sim.task_components.validators.base import (
    Validator,
    ValidatorActor,
    ValidatorOutput,
)
from robo_orchard_sim.task_components.validators.context import (
    build_validator_context,
)

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.splits import AssetSplits
    from robo_orchard_sim.ext.envs.env_base import IsaacEnvContextManager
    from robo_orchard_sim.ext.envs.manager_based_env import (
        IsaacManagerBasedEnv,
    )
    from robo_orchard_sim.launcher import SimpleIsaacAppLauncher
    from robo_orchard_sim.orchard_env.orchard_env import OrchardEnv

__all__ = [
    "Evaluator",
    "EvaluatorCfg",
    "EvaluationRuntime",
    "LaunchConfig",
]


def _get_task_builder():
    from robo_orchard_sim.benchmark import build_task

    return build_task


def _get_asset_resolution_error_cls():
    from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
        AssetResolutionError,
    )

    return AssetResolutionError


def _create_asset_registry(asset_root: str):
    from robo_orchard_sim.asset_manager.registry import AssetRegistry

    return AssetRegistry(asset_root)


def _create_asset_resolver(
    *,
    registry_obj: Any,
    seed: int,
    active_snapshot: frozenset[str] | None = None,
    splits: "AssetSplits | None" = None,
):
    """Create the resolver used to assemble one episode-scoped task.

    The resolver is seeded from the episode seed supplied by the evaluator.
    That means asset identity selection can vary from episode to episode,
    while the same seed is also forwarded to ``env.reset(...)`` for
    reset-time randomness such as pose variation.
    """
    from robo_orchard_sim.asset_manager.resolver import AssetResolver

    return AssetResolver(
        registry=registry_obj,
        splits=splits,
        rng=np.random.default_rng(seed),
        active_snapshot=active_snapshot,
    )


def _get_isaac_env_context_manager_cls():
    from robo_orchard_sim.ext.envs.env_base import IsaacEnvContextManager

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


@dataclass(frozen=True)
class EvaluationRuntime:
    """Externally owned runtime objects for policy evaluation."""

    sim_app: Any


class Evaluator:
    """Evaluator that runs fixed-number episodes with explicit step loops."""

    InitFromConfig: bool = True

    cfg: "EvaluatorCfg"

    def __init__(self, cfg: "EvaluatorCfg") -> None:
        self.cfg = cfg
        self._launcher: SimpleIsaacAppLauncher | None = None
        self._runtime: EvaluationRuntime | None = None
        self._env_cm: IsaacEnvContextManager | None = None
        self._env: IsaacManagerBasedEnv | None = None
        self._task: OrchardEnv | None = None
        self._record_run_dir: str | None = None
        self._active_snapshot_uuids: frozenset[str] | None = None
        self._splits: AssetSplits | None = None
        if (
            self.cfg.snapshot_path is not None
            or self.cfg.splits_path is not None
        ):
            _reg = _create_asset_registry(self.cfg.asset_root)
            if self.cfg.snapshot_path is not None:
                from robo_orchard_sim.asset_manager.snapshot import (
                    SnapshotError,
                    load_snapshot,
                )

                try:
                    self._active_snapshot_uuids = load_snapshot(
                        self.cfg.snapshot_path, _reg
                    ).uuids
                except SnapshotError as exc:
                    raise SystemExit(
                        "\nERROR loading snapshot "
                        f"{self.cfg.snapshot_path}: {exc}"
                    ) from exc
            if self.cfg.splits_path is not None:
                from robo_orchard_sim.asset_manager.splits import (
                    AssetSplitsError,
                    load_asset_splits,
                )

                try:
                    self._splits = load_asset_splits(
                        self.cfg.splits_path, _reg
                    )
                except AssetSplitsError as exc:
                    raise SystemExit(
                        f"\nERROR loading splits {self.cfg.splits_path}: {exc}"
                    ) from exc
        if self.cfg.enable_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            self._record_run_dir = os.path.join(
                self.cfg.record_dir,
                f"{self.cfg.task_name}_{timestamp}",
            )

    def __enter__(self) -> "Evaluator":
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
        policy = self._normalize_policy(policy_or_cfg)
        self._ensure_launcher()
        resolution_error_cls = _get_asset_resolution_error_cls()

        max_attempts = (
            self.cfg.episode_num * 3
            if self.cfg.resample_on_skip
            else self.cfg.episode_num
        )
        episode_results: list[EpisodeResult] = []
        skipped_episodes: list[SkippedEpisode] = []
        attempt = 0
        while (
            len(episode_results) < self.cfg.episode_num
            and attempt < max_attempts
        ):
            seed = self.cfg.seed + attempt
            episode_idx = len(episode_results)
            attempt += 1
            try:
                if self.cfg.enable_recording:
                    env = self._prepare_episode_env(
                        episode_idx=episode_idx,
                        seed=seed,
                    )
                else:
                    env = self._reload_env(
                        task=self._build_task_from_cfg(seed=seed),
                    )
                result = self._run_episode(
                    env=env,
                    policy=policy,
                    seed=seed,
                )
            except resolution_error_cls as exc:
                print(f"[skip seed={seed}] asset resolution failed: {exc}")
                self._close_env()
                skipped_episodes.append(
                    SkippedEpisode(seed=seed, reason=str(exc))
                )
                continue
            except Exception as exc:
                print(
                    f"Episode {episode_idx + 1}/"
                    f"{self.cfg.episode_num} failed with "
                    f"{type(exc).__name__}: {exc}"
                )
                self._close_env()
                result = self._build_episode_error_result(
                    episode_idx=episode_idx,
                    seed=seed,
                    exc=exc,
                )
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
            skipped_episodes=skipped_episodes,
        )

    def _build_episode_error_result(
        self,
        *,
        episode_idx: int,
        seed: int,
        exc: Exception,
    ) -> EpisodeResult:
        """Build a complete failed episode result for per-seed errors."""
        error_type = type(exc).__name__
        metrics = {}
        if self.cfg.enable_recording:
            metrics["record_dir"] = self._episode_record_dir(
                episode_idx=episode_idx,
                seed=seed,
            )

        return EpisodeResult(
            seed=seed,
            success=False,
            progress=0.0,
            steps=0,
            stop_reason=f"episode_error:{error_type}",
            metrics=metrics,
        )

    def create_launcher(self) -> SimpleIsaacAppLauncher:
        """Create the Isaac application launcher for this evaluator."""
        return _create_launcher(
            headless=self.cfg.launch.headless,
            enable_cameras=self.cfg.launch.enable_cameras,
            virtual_display=self.cfg.launch.virtual_display,
        )

    def run_with_runtime(
        self,
        policy_or_cfg: PolicyMixin | PolicyConfig,
        runtime: EvaluationRuntime | None = None,
        *,
        sim_app: Any | None = None,
    ) -> EvaluationResult:
        """Evaluate using an externally owned Isaac runtime."""
        if runtime is None:
            if sim_app is None:
                raise ValueError(
                    "sim_app is required when runtime is not provided"
                )
            runtime = EvaluationRuntime(sim_app=sim_app)

        previous_runtime = self._runtime
        self._runtime = runtime
        try:
            return self.evaluate(policy_or_cfg)
        finally:
            self.close()
            self._runtime = previous_runtime

    def _ensure_launcher(self) -> SimpleIsaacAppLauncher | Any:
        if self._runtime is not None:
            return self._runtime.sim_app
        if self._launcher is not None:
            return self._launcher

        self._launcher = self.create_launcher()
        return self._launcher

    def _ensure_env(self) -> IsaacManagerBasedEnv:
        if self._env is not None:
            return self._env
        return self._open_env()

    def _build_task_from_cfg(self, *, seed: int) -> OrchardEnv:
        """Build one orchard task using the provided episode seed."""
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
            seed=seed,
            active_snapshot=self._active_snapshot_uuids,
            splits=self._splits,
        )
        task_builder = _get_task_builder()
        task = task_builder(
            self.cfg.task_name,
            resolver=resolver,
            config_path=config_path,
        )
        return task

    def _open_env(
        self,
        task: OrchardEnv | None = None,
        *,
        seed: int | None = None,
    ) -> IsaacManagerBasedEnv:
        self._ensure_launcher()

        if task is None:
            if seed is None:
                seed = self.cfg.seed
            task = self._build_task_from_cfg(seed=seed)
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
        *,
        seed: int | None = None,
    ) -> IsaacManagerBasedEnv:
        self._close_env()
        return self._open_env(task=task, seed=seed)

    def _episode_record_dir(self, *, episode_idx: int, seed: int) -> str:
        if self._record_run_dir is None:
            raise RuntimeError("Recording directory requested when disabled.")
        return os.path.join(
            self._record_run_dir,
            f"episode_{episode_idx:04d}_seed_{seed}",
        )

    def _prepare_episode_env(
        self,
        *,
        episode_idx: int,
        seed: int,
    ) -> IsaacManagerBasedEnv:
        from robo_orchard_sim.ext.envs.managers.record import (
            ManualRecordControllerCfg,
        )

        task = self._build_task_from_cfg(seed=seed).configure_recording(
            file_path=self._episode_record_dir(
                episode_idx=episode_idx,
                seed=seed,
            ),
            controller=ManualRecordControllerCfg(),
        )
        return self._reload_env(task=task)

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
        if self._task is None:
            self._ensure_env()
        assert self._task is not None
        return self._get_runtime_task().build_validator(
            actors=actors,
            context=build_validator_context(self._task.embodiment),
        )

    def _normalize_policy(
        self,
        policy_or_cfg: PolicyMixin | PolicyConfig,
    ) -> PolicyMixin:
        if isinstance(policy_or_cfg, PolicyMixin):
            return policy_or_cfg
        return policy_or_cfg()

    def _resolve_policy_tag(self, policy: PolicyMixin) -> str | None:
        """Return unified logging tag from policy or policy cfg."""
        logging_tag = getattr(policy, "logging_tag", None)
        if isinstance(logging_tag, str):
            return logging_tag
        cfg = getattr(policy, "cfg", None)
        cfg_logging_tag = getattr(cfg, "logging_tag", None)
        if isinstance(cfg_logging_tag, str):
            return cfg_logging_tag
        return None

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

    def _settle_scene(self, env: IsaacManagerBasedEnv) -> EnvStepReturn:
        from robo_orchard_sim.utils.env_utils import SettleTracker

        max_settle_steps = max(self.cfg.max_settle_steps, 1)
        tracker = SettleTracker(streak=self.cfg.settle_streak)
        latest_step_return = env.step()
        for _ in range(max_settle_steps):
            if tracker.update(env.scene):
                return latest_step_return
            latest_step_return = env.step()
        for name, rot_deg, pos_mm in tracker.last_breaches:
            print(
                f"[Scene asset not settled]: name={name}, "
                f"rot_offset={rot_deg:.3f}deg, pos_offset={pos_mm:.3f}mm",
            )
        return latest_step_return

    def _start_manual_recording(self, env: IsaacManagerBasedEnv) -> None:
        if not self.cfg.enable_recording:
            return

        record_manager = env.record_manager
        if record_manager is None:
            return

        record_manager.start_record()

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

        # start record env
        self._start_manual_recording(env)

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
        policy_tag = self._resolve_policy_tag(policy)
        validator_output = ValidatorOutput(
            success=False,
            progress=0.0,
            metrics={},
        )
        for step_idx in range(self.cfg.max_steps):
            policy_input = self._build_policy_input(
                policy=policy,
                observations=observations,
                instruction_text=instruction_text,
            )
            action = policy(policy_input)
            if isinstance(action, UnifiedJointCommand):
                if self._task is None:
                    raise RuntimeError(
                        "Task runtime is required to translate "
                        "UnifiedJointCommand actions."
                    )
                embodiment = self._task.embodiment
                action = embodiment.translate_joint_command_to_env_action(
                    action
                )
            step_return = env.step(action)
            observations = step_return.observations
            validator_output = validator.evaluate(env, env_idx=0)
            steps = step_idx + 1
            if step_idx % 50 == 0:
                if policy_tag is None:
                    print(
                        f"[step={step_idx}] "
                        f"validator_output={validator_output}"
                    )
                else:
                    print(
                        f"[{policy_tag}] [step={step_idx}] "
                        f"validator_output={validator_output}"
                    )

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

    def _build_policy_input(
        self,
        *,
        policy: PolicyMixin,
        observations: dict[str, Any],
        instruction_text: str | None,
    ) -> Any:
        assert self._task is not None
        canonical = canonicalize_observations(
            observations=observations,
            instruction=instruction_text,
            schema=self._task.embodiment.get_policy_binding_schema(),
        )
        self._validate_policy_input(policy=policy, canonical=canonical)
        return canonical

    @staticmethod
    def _validate_policy_input(
        *,
        policy: PolicyMixin,
        canonical: CanonicalPolicyInput,
    ) -> None:
        policy_requirement = getattr(policy, "policy_requirement", None)
        if not callable(policy_requirement):
            return

        requirement = policy_requirement()
        if requirement is None:
            return

        validate_policy_compatibility(
            canonical=canonical,
            requirement=requirement,
        )

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
    resample_on_skip: bool = True
    max_steps: int = 1000
    max_settle_steps: int = 250
    settle_streak: int = 50
    snapshot_path: Path | None = None
    splits_path: Path | None = None
