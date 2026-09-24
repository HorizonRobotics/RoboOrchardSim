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
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

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
    RESAMPLE_ATTEMPT_MULTIPLIER,
    EpisodeResult,
    EvaluationResult,
    SkippedEpisode,
)
from robo_orchard_sim.policy.canonicalizer import (
    canonicalize_observations,
    validate_policy_compatibility,
)
from robo_orchard_sim.task_components.selector import (
    SelectorName,
    create_selector,
)
from robo_orchard_sim.task_components.validators.base import (
    Validator,
    ValidatorOutput,
)
from robo_orchard_sim.task_components.validators.context import (
    ValidatorContext,
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
    "evaluation_runtime",
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


@contextmanager
def evaluation_runtime(
    launch: LaunchConfig | None = None,
) -> Generator[EvaluationRuntime, None, None]:
    """Own one Isaac app for the whole ``with`` block.

    The app is closed only on block exit. IsaacSim 5.1 shutdown may hard-exit
    the process (stage-transition deadlock plus the launcher watchdog), so any
    result that must survive has to be written inside the block.
    """
    launch = launch or LaunchConfig()
    launcher = _create_launcher(
        headless=launch.headless,
        enable_cameras=launch.enable_cameras,
        virtual_display=launch.virtual_display,
    )
    try:
        yield EvaluationRuntime(sim_app=launcher.app)
    finally:
        _close_launcher(launcher)


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
        # Built once for role-based tasks and rebound each episode, so
        # that anything holding a reference follows the current target.
        self._role_registry: Any = None
        self._active_snapshot_uuids: frozenset[str] | None = None
        self._splits: AssetSplits | None = None
        self._asset_registry = _create_asset_registry(self.cfg.asset_root)
        if (
            self.cfg.snapshot_path is not None
            or self.cfg.splits_path is not None
        ):
            _reg = self._asset_registry
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

        # Reusing a scene only serves swap; without it every episode
        # gets a scene of its own, as evaluation has always run.
        per_scene = (
            max(1, self.cfg.swap.swap_per_scene)
            if self.cfg.swap.enabled
            else 1
        )
        # A scene that fails to resolve costs an attempt either way; with
        # resampling off the attempt budget is just the episode count, so
        # a skipped scene means that many fewer episodes get run.
        scenes_needed = -(-self.cfg.episode_num // per_scene)
        max_attempts = scenes_needed * (
            RESAMPLE_ATTEMPT_MULTIPLIER if self.cfg.resample_on_skip else 1
        )
        if self.cfg.scene_seed_candidates is None:
            scene_seed_candidates = tuple(
                range(self.cfg.seed, self.cfg.seed + max_attempts)
            )
        else:
            scene_seed_candidates = tuple(self.cfg.scene_seed_candidates)
            if len(scene_seed_candidates) < max_attempts:
                raise ValueError(
                    "scene_seed_candidates must provide at least "
                    f"{max_attempts} seeds, got {len(scene_seed_candidates)}"
                )
            if len(set(scene_seed_candidates)) != len(scene_seed_candidates):
                raise ValueError("scene_seed_candidates must be unique")
            scene_seed_candidates = scene_seed_candidates[:max_attempts]
        episode_results: list[EpisodeResult] = []
        skipped_episodes: list[SkippedEpisode] = []
        attempted_scene_seeds: list[int] = []
        scene_idx = 0
        while (
            len(episode_results) < self.cfg.episode_num
            and scene_idx < max_attempts
        ):
            # Scene layer: objects are drawn once here and stay put for
            # every episode below.
            scene_seed = scene_seed_candidates[scene_idx]
            attempted_scene_seeds.append(scene_seed)
            scene_idx += 1
            try:
                env = self._open_scene(
                    scene_idx=scene_idx - 1,
                    scene_seed=scene_seed,
                )
            except resolution_error_cls as exc:
                print(
                    f"[skip scene seed={scene_seed}] "
                    f"asset resolution failed: {exc}"
                )
                self._close_env()
                skipped_episodes.append(
                    SkippedEpisode(seed=scene_seed, reason=str(exc))
                )
                continue
            except Exception as exc:
                # A usable seed whose scene cannot be built fails every
                # episode assigned to that scene, including swap episodes.
                print(
                    f"Scene seed={scene_seed} failed to build with "
                    f"{type(exc).__name__}: {exc}"
                )
                self._close_env()
                failed_count = min(
                    per_scene, self.cfg.episode_num - len(episode_results)
                )
                for _ in range(failed_count):
                    episode_results.append(
                        self._build_episode_error_result(
                            episode_idx=len(episode_results),
                            seed=scene_seed,
                            exc=exc,
                        )
                    )
                continue

            for episode_in_scene in range(per_scene):
                if len(episode_results) >= self.cfg.episode_num:
                    break
                global_idx = len(episode_results)
                # Under swap the whole point is that the instruction is
                # the only thing that changed, so the episodes of one
                # scene reset from the same seed and the objects land
                # where they landed before. Without it, each episode is
                # meant to be a fresh arrangement.
                episode_seed = (
                    scene_seed
                    if self.cfg.swap.enabled
                    else scene_seed + episode_in_scene
                )
                record_dir = None
                if self.cfg.enable_recording:
                    record_dir = os.path.join(
                        self._scene_record_dir(
                            scene_idx=scene_idx - 1, seed=scene_seed
                        ),
                        f"episode_{global_idx:04d}_seed_{episode_seed}",
                    )
                try:
                    result = self._run_episode(
                        env=env,
                        policy=policy,
                        seed=episode_seed,
                        episode_idx=episode_in_scene,
                        global_episode_idx=global_idx,
                    )
                except Exception as exc:
                    print(
                        f"Episode {global_idx + 1}/"
                        f"{self.cfg.episode_num} failed with "
                        f"{type(exc).__name__}: {exc}"
                    )
                    result = self._build_episode_error_result(
                        episode_idx=global_idx,
                        seed=episode_seed,
                        exc=exc,
                        record_dir=record_dir,
                    )
                if record_dir is not None:
                    result.metrics = {
                        **result.metrics,
                        "record_dir": record_dir,
                    }
                episode_results.append(result)

            self._close_env()

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
            attempted_scene_seeds=attempted_scene_seeds,
        )

    def _build_episode_error_result(
        self,
        *,
        episode_idx: int,
        seed: int,
        exc: Exception,
        record_dir: str | None = None,
    ) -> EpisodeResult:
        """Build a complete failed episode result for per-seed errors."""
        error_type = type(exc).__name__
        metrics = {}
        if self.cfg.enable_recording and record_dir is not None:
            metrics["record_dir"] = record_dir

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

        resolver = _create_asset_resolver(
            registry_obj=self._asset_registry,
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

    def _scene_record_dir(self, *, scene_idx: int, seed: int) -> str:
        if self._record_run_dir is None:
            raise RuntimeError("Recording directory requested when disabled.")
        return os.path.join(
            self._record_run_dir,
            f"scene_{scene_idx:04d}_seed_{seed}",
        )

    def _open_scene(
        self,
        *,
        scene_idx: int,
        scene_seed: int,
    ) -> IsaacManagerBasedEnv:
        """Build one scene, to be reused by all of its episodes.

        Objects are drawn here and stay put until the scene is closed;
        each episode below only rebinds roles and resets poses.
        """
        task = self._build_task_from_cfg(seed=scene_seed)
        if self.cfg.enable_recording:
            from robo_orchard_sim.ext.envs.managers.record import (
                ManualRecordControllerCfg,
            )

            task = task.configure_recording(
                file_path=self._scene_record_dir(
                    scene_idx=scene_idx,
                    seed=scene_seed,
                ),
                controller=ManualRecordControllerCfg(),
            )
        # A registry belongs to the scene it was built for: its bindings
        # name scene entities that disappear when the scene does.
        self._role_registry = None
        return self._reload_env(task=task)

    def _get_runtime_task(self) -> Any:
        """Return the task bound to the current evaluator environment."""
        if self._task is None:
            self._ensure_env()
        assert self._task is not None
        return self._task.task

    def _bind_roles_for_episode(self, episode_idx: int, seed: int) -> None:
        """Point each declared role at its target for this episode.

        The registry outlives the episode: only its bindings change, so
        whoever holds a reference to it follows along without being
        rebuilt.
        """
        from robo_orchard_sim.task_components.role_registry import RoleRegistry

        task = self._get_runtime_task()

        if self._role_registry is None:
            self._role_registry = RoleRegistry(num_envs=1)

        for role_id, role_spec in task.roles.items():
            candidates = task.get_role_candidates(
                role_id, swap=self.cfg.swap.enabled
            )
            if role_spec.cardinality == "one":
                if not candidates:
                    raise ValueError(f"Role {role_id!r} has no candidates")
                chosen = candidates[0]
                if self.cfg.swap.enabled:
                    selector = create_selector(
                        self.cfg.swap.selector, seed=seed, role_id=role_id
                    )
                    chosen = selector.select(candidates, 1, episode_idx)[0]
                self._role_registry.bind_one(0, role_id, chosen)
            else:
                self._role_registry.bind_many(0, role_id, list(candidates))

    def _build_validator_context(self) -> ValidatorContext:
        """Build the runtime context the validator and instruction share."""
        if self._task is None:
            self._ensure_env()
        assert self._task is not None
        from robo_orchard_sim.task_components.role_registry import RoleRegistry

        # Episodes always bind before this runs; the fallback registry
        # only serves callers that build a context outside that loop.
        role_registry = self._role_registry or RoleRegistry(num_envs=1)
        return ValidatorContext.from_embodiment(
            self._task.embodiment, role_registry
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

    def _start_manual_recording(
        self,
        env: IsaacManagerBasedEnv,
        *,
        prefix: str | None = None,
    ) -> None:
        if not self.cfg.enable_recording:
            return

        record_manager = env.record_manager
        if record_manager is None:
            return

        # One scene's episodes share a writer root, so each episode needs
        # its own prefix to land in a directory of its own.
        record_manager.start_record(prefix=prefix or "")

    def _build_episode_metadata(
        self,
        env: IsaacManagerBasedEnv,
        context: ValidatorContext,
        scene_names: list[str],
        validator_output: ValidatorOutput,
        instruction_text: str | None,
        *,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        # TODO：user can add meata data here

        if not scene_names:
            return {}

        from robo_orchard_sim.utils.env_utils import bbox_of

        def _pose(state: Any) -> list[float]:
            # Recorded poses stay 7-dim (pos + quat); the captured state
            # carries velocities beyond that.
            return state[:7].cpu().numpy().tolist()

        meta_data = {
            "init_position": {
                name: _pose(context.init_state_of(name, env_idx))
                for name in scene_names
            },
            "final_position": {
                name: _pose(context.final_state_of(name, env_idx))
                for name in scene_names
            },
            "actors": {
                name: {
                    "actor_category": env.scene[name].cfg.category
                    or "unknown",
                    "actor_type": env.scene[name].cfg.actor_type or "unknown",
                    "actor_uuid": env.scene[name].cfg.uuid or "unknown",
                    "bbox": bbox_of(env.scene[name].cfg),
                }
                for name in scene_names
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
        context: ValidatorContext | None = None,
    ) -> str | None:
        task = self._get_runtime_task()
        if task.instruction is None:
            return None

        instruction = task.instruction
        actors = task.build_instruction_context(
            env,
            actor_description_seed=actor_description_seed,
            context=context,
        )
        return instruction.render(
            actors=actors,
            template_seed=template_seed,
            actor_description_seed=actor_description_seed,
        )

    def _record_episode_metadata(
        self,
        env: IsaacManagerBasedEnv,
        context: ValidatorContext,
        scene_names: list[str],
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
                    env,
                    context,
                    scene_names,
                    validator_output,
                    instruction_text,
                    env_idx=env_idx,
                )
                for env_idx in range(num_envs)
            ]
        else:
            meta_dict = self._build_episode_metadata(
                env,
                context,
                scene_names,
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
        episode_idx: int = 0,
        global_episode_idx: int = 0,
        template_seed: int | None = None,
        actor_description_seed: int | None = None,
    ) -> tuple[
        dict[str, Any],
        list[str],
        Validator,
        str | None,
        ValidatorContext,
    ]:
        # Bind this episode's targets before the reset, so the terms
        # that place objects, the validator and the instruction all
        # describe the same ones.
        self._bind_roles_for_episode(episode_idx, seed)
        reset_return = env.reset(
            seed=seed,
            role_bindings=(
                self._role_registry.bindings(0)
                if self._role_registry is not None
                else None
            ),
        )
        observations = reset_return.observations

        settle_return = self._settle_scene(env)
        observations = settle_return.observations

        # start record env
        self._start_manual_recording(
            env,
            prefix=f"episode_{global_episode_idx:04d}_seed_{seed}",
        )

        context = self._build_validator_context()

        # Tasks own operable identities; the context owns settled state.
        operable_names = self._get_runtime_task().get_operable_scene_names()
        if operable_names:
            context.capture_init_states(env, operable_names)

        if template_seed is None:
            template_seed = seed
        if actor_description_seed is None:
            actor_description_seed = seed
        instruction_text = self._build_policy_instruction(
            env=env,
            template_seed=template_seed,
            actor_description_seed=actor_description_seed,
            context=context,
        )
        if instruction_text is not None:
            print(f"instruction: {instruction_text}")

        validator = self._get_runtime_task().build_validator(context=context)
        validator.reset()

        policy.reset()
        return (
            observations,
            operable_names,
            validator,
            instruction_text,
            context,
        )

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
        fixed_horizon = validator.fixed_horizon
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

            if not fixed_horizon:
                if validator_output.success:
                    stop_reason = "success"
                    break
                if self._extract_terminated(step_return):
                    stop_reason = "terminated"
                    break
                if self._extract_truncated(step_return):
                    stop_reason = "truncated"
                    break

        if fixed_horizon:
            validator_output = validator.finalize()

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
        episode_idx: int = 0,
        global_episode_idx: int = 0,
        template_seed: int | None = None,
        actor_description_seed: int | None = None,
    ) -> EpisodeResult:
        (
            observations,
            operable_names,
            validator,
            instruction_text,
            context,
        ) = self._prepare_episode(
            env=env,
            policy=policy,
            seed=seed,
            episode_idx=episode_idx,
            global_episode_idx=global_episode_idx,
            template_seed=template_seed,
            actor_description_seed=actor_description_seed,
        )
        steps, stop_reason, validator_output = self._step_episode(
            env=env,
            policy=policy,
            observations=observations,
            validator=validator,
            instruction_text=instruction_text,
        )

        if operable_names:
            context.capture_final_states(env, operable_names)
        self._record_episode_metadata(
            env,
            context,
            operable_names,
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
            instruction=instruction_text or "",
            stage_scores=dict(validator_output.stage_scores),
            checker_summary=(
                validator.metric_store.checker_summary()
                if validator.metric_store is not None
                else None
            ),
        )


class SwapConfig(Config):
    """Rotate a task's target between episodes of one scene.

    Off by default: a scene then serves a single episode and every role
    stays on the target it was built with, which is how evaluation has
    always behaved.

    Turning it on selects targets using ``selector``. By default each
    scene serves one episode, with a new seed for each scene. Setting
    ``swap_per_scene`` above one reuses a scene for multiple episodes.
    Within a scene, random visits each candidate once
    per shuffled round; round-robin follows declaration order. Those
    episodes also reset from the same seed, so the objects stay where
    they were and the instruction is the only thing that changed —
    which is what makes the score read as instruction-following rather
    than as luck with the arrangement. Tasks whose target has no
    alternatives — a single declared object, no clutter to draw from —
    are unaffected: their candidate pool holds one entry and every
    episode picks it.
    """

    enabled: bool = False
    swap_per_scene: int = 1
    """Episodes one scene serves. Only read when swap is enabled."""
    selector: SelectorName = "random"


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
    swap: SwapConfig = SwapConfig()
    resample_on_skip: bool = True
    scene_seed_candidates: tuple[int, ...] | None = None
    max_steps: int = 1000
    max_settle_steps: int = 250
    settle_streak: int = 50
    snapshot_path: Path | None = None
    splits_path: Path | None = None
