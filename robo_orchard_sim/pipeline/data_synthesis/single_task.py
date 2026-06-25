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

"""Single-task data synthesis runner."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from robo_orchard_core.utils.config import ClassConfig, ClassType_co

from robo_orchard_sim.contracts.joint_command import EnvActionState
from robo_orchard_sim.task_components.validators.base import ValidatorOutput

if TYPE_CHECKING:
    from robo_orchard_sim.asset_manager.splits import AssetSplits


@dataclass
class LaunchConfig:
    """Launcher options for single-task data synthesis."""

    headless: bool = True
    enable_cameras: bool = True
    virtual_display: bool = False


@dataclass
class EpisodeStopReason:
    """String constants for episode stop reasons."""

    MAX_STEPS: str = "max_steps"
    SIM_APP_STOPPED: str = "sim_app_stopped"
    ACTION_PLAN_COMPLETE: str = "action_plan_complete"
    SCENE_NOT_SETTLED: str = "scene_not_settled"
    SUCCESS: str = "success"
    EPISODE_ERROR: str = "episode_error"


STOP_REASON = EpisodeStopReason()


@dataclass
class EpisodeSummary:
    """Observable result for one synthesized episode."""

    episode_index: int
    seed: int
    steps: int
    stop_reason: str
    success: bool
    record_dir: str | None = None
    mcap_paths: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class DataSynthesisRuntime:
    """Externally owned runtime objects for data synthesis."""

    sim_app: Any


@dataclass
class TaskRunResult:
    """Observable result for one task-level data synthesis run."""

    task: str
    config_path: str | None
    task_save_root: str | None
    config_dir: str | None
    data_dir: str
    record_dir: str
    episodes: list[EpisodeSummary]
    total: int
    success_count: int
    success_rate: float
    error: str | None = None
    user_data: dict[str, Any] = field(default_factory=dict)


class TaskDataSynthesisRunner:
    """Run one task data synthesis request for configured episodes."""

    InitFromConfig = True

    cfg: "TaskDataSynthesisCfg"

    def __init__(self, cfg: "TaskDataSynthesisCfg") -> None:
        self.cfg = cfg
        self._apply_task_save_root()
        print(
            "Data synthesis recordings will be written to: "
            f"{self.cfg.record_dir}"
        )

        self._active_snapshot_uuids: frozenset[str] | None = None
        self._splits: AssetSplits | None = None
        if (
            self.cfg.snapshot_path is not None
            or self.cfg.splits_path is not None
        ):
            from robo_orchard_sim.asset_manager.registry import AssetRegistry

            _reg = AssetRegistry(self.cfg.asset_root)
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

    def iter_episode_seeds(self) -> range:
        """Return the deterministic seed sequence used by this run."""
        return range(self.cfg.seed, self.cfg.seed + self.cfg.episode_num)

    def run(self) -> TaskRunResult:
        """Launch Isaac and synthesize all configured episodes."""
        launcher = self.create_launcher()
        sim_app = launcher.app

        try:
            return self.run_with_runtime(sim_app=sim_app)
        finally:
            launcher.close()

    def run_task(self) -> TaskRunResult:
        """Launch Isaac and return a task-level aggregate run result."""
        return self.run()

    def run_with_runtime(
        self,
        *,
        sim_app: Any,
    ) -> TaskRunResult:
        """Synthesize episodes using an externally owned Isaac runtime."""
        action_manager = self.build_action_manager()
        summaries = []
        for episode_index, seed in enumerate(self.iter_episode_seeds()):
            record_dir = self.episode_record_dir(
                episode_index=episode_index,
                seed=seed,
            )
            try:
                summaries.append(
                    self.run_episode(
                        episode_index=episode_index,
                        seed=seed,
                        sim_app=sim_app,
                        action_manager=action_manager,
                    )
                )
            except Exception as exc:
                print(
                    f"Episode {episode_index + 1}/"
                    f"{self.cfg.episode_num} failed with "
                    f"{type(exc).__name__}: {exc}"
                )
                summaries.append(
                    EpisodeSummary(
                        episode_index=episode_index,
                        seed=seed,
                        steps=0,
                        stop_reason=(f"episode_error:{type(exc).__name__}"),
                        success=False,
                        record_dir=record_dir,
                        mcap_paths=self.find_mcap_paths(record_dir),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            finally:
                action_manager.clear(clear_planner_instances=False)

        self._write_successful_recording_paths(summaries)
        return self._build_task_run_result(summaries)

    def run_task_with_runtime(
        self,
        runtime: DataSynthesisRuntime | None = None,
        *,
        sim_app: Any | None = None,
    ) -> TaskRunResult:
        """Synthesize with external runtime and return task aggregate data."""
        if runtime is not None:
            sim_app = runtime.sim_app
        if sim_app is None:
            raise ValueError(
                "sim_app is required when runtime is not provided"
            )

        return self.run_with_runtime(sim_app=sim_app)

    def _apply_task_save_root(self) -> None:
        if self.cfg.task_save_root is None:
            return

        self.cfg.output_config_dir = os.path.join(
            self.cfg.task_save_root,
            "config",
        )
        self.cfg.record_dir = os.path.join(
            self.cfg.task_save_root,
            "data",
        )

    def _build_task_run_result(
        self,
        episodes: list[EpisodeSummary],
        *,
        error: str | None = None,
    ) -> TaskRunResult:
        total = len(episodes)
        success_count = sum(summary.success for summary in episodes)
        return TaskRunResult(
            task=self.cfg.task,
            config_path=self.cfg.config,
            task_save_root=self.cfg.task_save_root,
            config_dir=self.cfg.output_config_dir,
            data_dir=self.cfg.record_dir,
            record_dir=self.cfg.record_dir,
            episodes=episodes,
            total=total,
            success_count=success_count,
            success_rate=success_count / total if total else 0.0,
            error=error,
            user_data=dict(self.cfg.user_data),
        )

    def _pump_sim_app_updates(
        self,
        sim_app: Any,
        *,
        update_count: int = 3,
    ) -> None:
        for _ in range(update_count):
            sim_app.update()

    def create_launcher(self) -> Any:
        """Create the Isaac application launcher for this run."""
        from robo_orchard_sim.launcher import SimpleIsaacAppLauncher

        return SimpleIsaacAppLauncher(
            headless=self.cfg.launch.headless,
            enable_cameras=self.cfg.launch.enable_cameras,
            virtual_display=self.cfg.launch.virtual_display,
        )

    def build_action_manager(self) -> Any:
        """Create the run-scoped atomic action manager."""
        from robo_orchard_sim.task_components.trajs_gen import (
            atomic_action_manager as _atomic_action_manager,
        )

        return _atomic_action_manager.AtomicActionManagerCfg(
            debug_vis=self.cfg.debug_vis
        )()

    def run_episode(
        self,
        *,
        episode_index: int,
        seed: int,
        sim_app: Any,
        action_manager: Any,
    ) -> EpisodeSummary:
        """Sample assets, create an env, and run one atomic-action episode."""
        from robo_orchard_sim.benchmark.registry import (
            build_task_atomic_action_plan,
        )
        from robo_orchard_sim.ext.envs.env_base import IsaacEnvContextManager
        from robo_orchard_sim.task_components.trajs_gen import (
            atomic_action_manager as _atomic_action_manager,
        )

        ActionStatusLogger = _atomic_action_manager.ActionStatusLogger

        record_dir = self.episode_record_dir(
            episode_index=episode_index,
            seed=seed,
        )
        orchard_env = self.build_orchard_env(seed=seed)
        if self.cfg.enable_recording:
            self.prepare_recording(
                orchard_env=orchard_env,
                episode_index=episode_index,
                seed=seed,
            )

        env_cfg = orchard_env.to_isaac_env_cfg()
        self._write_env_cfg(
            env_cfg=env_cfg,
            episode_index=episode_index,
            seed=seed,
        )

        print(
            f"Episode {episode_index + 1}/{self.cfg.episode_num}: "
            f"task={self.cfg.task!r}, seed={seed}"
        )
        print(f"Scene type: {type(orchard_env.scene).__name__}")
        print(f"Embodiment: {orchard_env.embodiment.scene_name}")
        print(f"Task type: {type(orchard_env.task).__name__}")

        env_manager = IsaacEnvContextManager(
            env_cfg,
            with_new_stage=True,
            disable_exit_on_stop=False,
        )
        episode_success = False
        with env_manager as env:
            plan = build_task_atomic_action_plan(
                task_name=self.cfg.task,
                orchard_env=orchard_env,
            )
            action_manager.clear()
            action_manager.register(plan)

            _ = env.reset(seed=seed)

            # Initialize action-manager terms from current joint state.
            embodiment = orchard_env.embodiment
            _ = env.step(EnvActionState.build_hold_position(env))

            scene_settled = self.settle_until_recording_starts(env)
            self._update_episode_record_data(
                env,
                {
                    "task": self.cfg.task,
                    "episode_index": episode_index,
                    "seed": seed,
                },
            )

            actors = self.build_validator_actors(
                runtime_task=orchard_env.task,
                scene=env.scene,
            )
            validator = self.build_validator(
                runtime_task=orchard_env.task,
                actors=actors,
                embodiment=orchard_env.embodiment,
            )
            validator.reset()
            self.capture_init_states(scene=env.scene, actors=actors)

            status_logger = ActionStatusLogger()
            steps, stop_reason, validator_output = (
                self._run_atomic_action_loop(
                    env=env,
                    manager=action_manager,
                    status_logger=status_logger,
                    sim_app=sim_app,
                    validator=validator,
                    embodiment=embodiment,
                )
            )

            if not scene_settled:
                stop_reason = STOP_REASON.SCENE_NOT_SETTLED
                episode_success = False
            else:
                episode_success = bool(validator_output.success)

            self.capture_final_states(scene=env.scene, actors=actors)
            self.record_validator_metadata(
                env=env,
                actors=actors,
                validator_output=validator_output,
            )

            self._update_episode_record_data(
                env,
                {
                    "task": self.cfg.task,
                    "episode_index": episode_index,
                    "seed": seed,
                    "steps": steps,
                    "stop_reason": stop_reason,
                },
            )
            self._finalize_episode_recording(env)

        self._pump_sim_app_updates(sim_app)
        print(
            f"Episode {episode_index + 1} finished: "
            f"steps={steps}, stop_reason={stop_reason}, "
            f"success={episode_success}"
        )
        return EpisodeSummary(
            episode_index=episode_index,
            seed=seed,
            steps=steps,
            stop_reason=stop_reason,
            success=episode_success,
            record_dir=record_dir,
            mcap_paths=self.find_mcap_paths(record_dir),
        )

    def build_orchard_env(self, *, seed: int):
        """Build a fresh OrchardEnv with assets sampled by ``seed``."""
        from robo_orchard_sim.asset_manager.registry import AssetRegistry
        from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
            AssetResolver,
            AssetResolverError,
        )
        from robo_orchard_sim.benchmark.registry import build_task

        registry = AssetRegistry(self.cfg.asset_root)
        resolver = AssetResolver(
            registry=registry,
            splits=self._splits,
            rng=np.random.default_rng(seed),
            active_snapshot=self._active_snapshot_uuids,
        )
        config_path = self._resolve_config_path()

        try:
            return build_task(
                task_name=self.cfg.task,
                resolver=resolver,
                config_path=config_path,
            )
        except KeyError as exc:
            raise ValueError(f"\nERROR: {exc}") from exc
        except AssetResolverError as exc:
            raise RuntimeError(
                f"\nERROR resolving assets from registry: {exc}\n"
                "Check the asset_configs: block in the task YAML, or pass "
                "a different --config."
            ) from exc
        except ValueError as exc:
            raise ValueError(f"\nERROR: {exc}") from exc

    def prepare_recording(
        self,
        orchard_env: Any,
        episode_index: int,
        seed: int,
    ) -> None:
        """Configure recording to start after reset-time scene settling."""
        from robo_orchard_sim.ext.envs.managers.record import (
            StationaryEpisodeRecordControllerCfg,
        )

        orchard_env.configure_recording(
            file_path=self.episode_record_dir(
                episode_index=episode_index,
                seed=seed,
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                max_wait_step=self.cfg.settle_steps,
                min_wait_step=min(50, self.cfg.settle_steps),
                streak=self.cfg.settle_streak,
            ),
        )

    def settle_until_recording_starts(self, env: Any) -> bool:
        """Step the scene and report whether it settled within the window.

        Steps up to ``settle_steps`` times and returns whether the scene
        reached a stationary streak (``settle_streak`` consecutive still
        frames). The verdict comes from a local tracker, not from
        ``record_manager.running``: the recording controller's own tracker
        leads this one by the reset-time hold step, so gating on it would
        report unsettled one frame early. Assumes
        ``settle_streak <= settle_steps``.
        """
        from robo_orchard_sim.utils.env_utils import SettleTracker

        tracker = SettleTracker(streak=self.cfg.settle_streak)
        for _ in range(self.cfg.settle_steps):
            _ = env.step()
            if tracker.update(env.scene):
                return True
        return False

    def build_validator_actors(
        self,
        *,
        runtime_task: Any,
        scene: Any,
    ) -> list[Any]:
        """Build validator actor snapshots from the runtime scene."""
        from robo_orchard_sim.task_components.validators.base import (
            ValidatorActor,
        )

        actor_names = runtime_task.get_validator_actor_names()
        return [
            ValidatorActor.from_rigid_object(name, scene[name])
            for name in actor_names
        ]

    def build_validator(
        self,
        *,
        runtime_task: Any,
        actors: list[Any],
        embodiment: Any,
    ) -> Any:
        """Build the task validator bound to the current actor snapshots."""
        from robo_orchard_sim.task_components.validators.context import (
            build_validator_context,
        )

        return runtime_task.build_validator(
            actors=actors,
            context=build_validator_context(embodiment),
        )

    def capture_init_states(self, *, scene: Any, actors: list[Any]) -> None:
        """Capture initial validator actor states from the runtime scene."""
        for actor in actors:
            actor.capture_init_state(scene[actor.name])

    def capture_final_states(self, *, scene: Any, actors: list[Any]) -> None:
        """Capture final validator actor states from the runtime scene."""
        for actor in actors:
            actor.capture_final_state(scene[actor.name])

    def build_episode_metadata(
        self,
        *,
        actors: list[Any],
        validator_output: Any,
        env_idx: int = 0,
    ) -> dict[str, Any]:
        """Build evaluator-compatible validator metadata for recording."""
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

        return meta_data

    def record_validator_metadata(
        self,
        *,
        env: Any,
        actors: list[Any],
        validator_output: Any,
    ) -> None:
        """Merge validator metadata into episode user data."""
        record_manager = getattr(env, "record_manager", None)
        if record_manager is None:
            return

        num_envs = getattr(env, "num_envs", 1)
        meta_dict: dict[str, Any] | list[dict[str, Any]]
        if num_envs > 1:
            meta_dict = [
                self.build_episode_metadata(
                    actors=actors,
                    validator_output=validator_output,
                    env_idx=env_idx,
                )
                for env_idx in range(num_envs)
            ]
        else:
            meta_dict = self.build_episode_metadata(
                actors=actors,
                validator_output=validator_output,
            )

        if not meta_dict:
            return

        record_manager.update_episode_user_data({"meta_dict": meta_dict})

    def robot_is_stationary(self, env: Any, embodiment: Any) -> bool:
        """Return whether all configured robot bodies have low velocity."""
        robot = env.scene[embodiment.scene_name]
        body_link_vel_w = robot.data.body_link_vel_w

        lin_vel = torch.linalg.vector_norm(body_link_vel_w[..., :3], dim=-1)
        ang_vel = torch.linalg.vector_norm(body_link_vel_w[..., 3:6], dim=-1)
        return bool(torch.all(lin_vel < 0.05) and torch.all(ang_vel < 0.1))

    def wait_until_robot_stationary(self, env: Any, sim_app: Any) -> None:
        """Keep stepping after task completion until the robot settles."""
        for _ in range(20):
            if not sim_app.is_running():
                return
            _ = env.step()

    def _run_atomic_action_loop(
        self,
        *,
        env: Any,
        manager: Any,
        status_logger: Any,
        sim_app: Any,
        validator: Any,
        embodiment: Any,
    ) -> tuple[int, str, ValidatorOutput]:
        stop_reason = STOP_REASON.MAX_STEPS
        steps = 0
        validator_output = ValidatorOutput(
            success=False,
            progress=0.0,
            metrics={},
        )
        env_action_state = EnvActionState.from_env(env)

        while steps < self.cfg.max_steps:
            if not sim_app.is_running():
                stop_reason = STOP_REASON.SIM_APP_STOPPED
                break

            joint_commands, state = manager.get_action(env)
            for log_line in status_logger.collect(
                running_actions=state.running_actions,
                step_idx=steps,
            ):
                print(log_line)

            if joint_commands:
                for joint_command in joint_commands.values():
                    env_action_state.update(
                        embodiment.translate_joint_command_to_env_action(
                            joint_command
                        ),
                    )
            _ = env.step(env_action_state.action())
            validator_output = validator.evaluate(env, env_idx=0)
            steps += 1

            # if validator_output.success:
            #     stop_reason = STOP_REASON.SUCCESS
            #     self.wait_until_robot_stationary(env, sim_app)
            #     break
            if not state.env_busy:
                stop_reason = STOP_REASON.ACTION_PLAN_COMPLETE
                self.wait_until_robot_stationary(env, sim_app)
                break

        return steps, stop_reason, validator_output

    def _resolve_config_path(self) -> str | None:
        if self.cfg.config is None:
            return None

        config_path = os.path.abspath(self.cfg.config)
        if not os.path.isfile(config_path):
            raise SystemExit(
                f"\nERROR: --config path does not exist: {config_path}"
            )
        return config_path

    def episode_record_dir(self, *, episode_index: int, seed: int) -> str:
        """Return the MCAP output directory for one synthesized episode."""
        return os.path.join(
            self.cfg.record_dir,
            f"episode_{episode_index:04d}_seed_{seed}",
        )

    def find_mcap_paths(self, record_dir: str) -> list[str]:
        """Return sorted MCAP files below one episode recording directory."""
        if not os.path.isdir(record_dir):
            return []

        mcap_paths: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(record_dir):
            for filename in filenames:
                if filename.endswith(".mcap"):
                    mcap_paths.append(os.path.join(dirpath, filename))
        return sorted(mcap_paths)

    def successful_recording_paths_file(self) -> str | None:
        """Return the output file for successful episode recording paths."""
        if self.cfg.output_config_dir is None:
            return None
        return os.path.join(
            self.cfg.output_config_dir,
            f"successful_recording_paths_{self.cfg.task}.txt",
        )

    def _write_successful_recording_paths(
        self,
        summaries: list[EpisodeSummary],
    ) -> None:
        output_path = self.successful_recording_paths_file()
        if output_path is None:
            return

        os.makedirs(self.cfg.output_config_dir, exist_ok=True)
        successful_paths = [
            mcap_path
            for summary in summaries
            if summary.success
            for mcap_path in summary.mcap_paths
        ]
        with open(output_path, "w", encoding="utf-8") as fw:
            fw.write("\n".join(successful_paths))
            if successful_paths:
                fw.write("\n")

    def _write_env_cfg(
        self,
        *,
        env_cfg: Any,
        episode_index: int,
        seed: int,
    ) -> None:
        if self.cfg.output_config_dir is None:
            return

        os.makedirs(self.cfg.output_config_dir, exist_ok=True)
        output_path = os.path.join(
            self.cfg.output_config_dir,
            f"env_config_{self.cfg.task}_{episode_index:04d}_seed_{seed}.json",
        )
        with open(output_path, "w", encoding="utf-8") as fw:
            fw.write(env_cfg.to_str(format="json", indent=4))

    def _update_episode_record_data(
        self,
        env: Any,
        data: dict[str, Any],
    ) -> None:
        record_manager = env.record_manager
        record_manager.update_episode_user_data(data)

    def _finalize_episode_recording(self, env: Any) -> None:
        env.record_manager.record_pre_reset()


class TaskDataSynthesisCfg(ClassConfig[TaskDataSynthesisRunner]):
    """Configuration for the single-task data synthesis runner."""

    class_type: ClassType_co[TaskDataSynthesisRunner] = TaskDataSynthesisRunner
    task: str
    asset_root: str
    config: str | None = None
    seed: int = 0
    episode_num: int = 1
    max_steps: int = 1000
    settle_steps: int = 250
    settle_streak: int = 50
    enable_recording: bool = True
    record_dir: str = "logs/data_synthesis"
    output_config_dir: str | None = "configs/data_synthesis"
    task_save_root: str | None = None
    snapshot_path: Path | None = None
    splits_path: Path | None = None
    launch: LaunchConfig = LaunchConfig()
    debug_vis: bool = False
    user_data: dict[str, Any] = {}
