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

"""Example: synthesize task episodes by resampling assets per seed."""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch

from robo_orchard_sim.tasks.validators.base import ValidatorOutput

_ASSET_ROOT_ENV = "ORCHARD_ASSET_LIBRARY"
_ROBOT_SCENE_NAME = "robots/dualarm_piper"
_LEFT_ARM_KEY = f"{_ROBOT_SCENE_NAME}/left_arm"
_RIGHT_ARM_KEY = f"{_ROBOT_SCENE_NAME}/right_arm"


@dataclass
class LaunchConfig:
    """Launcher options for the data synthesis example."""

    headless: bool = True
    enable_cameras: bool = True
    virtual_display: bool = False


@dataclass
class DataSynthesisConfig:
    """Configuration for the data synthesis example runner."""

    task: str
    asset_root: str
    config: str | None = None
    seed: int = 0
    episode_num: int = 1
    max_steps: int = 1000
    settle_steps: int = 50
    enable_recording: bool = True
    record_dir: str = "logs/data_synthesis"
    output_config_dir: str | None = "configs/data_synthesis"
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    debug_vis: bool = False


@dataclass
class EpisodeSummary:
    """Observable result for one synthesized episode."""

    episode_index: int
    seed: int
    steps: int
    stop_reason: str
    success: bool


class DataSynthesisRunner:
    """Run data synthesis episodes with per-seed asset sampling."""

    def __init__(self, cfg: DataSynthesisConfig) -> None:
        self.cfg = cfg
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._record_run_dir = os.path.join(
            self.cfg.record_dir,
            f"{self.cfg.task}_{timestamp}",
        )

    def iter_episode_seeds(self) -> range:
        """Return the deterministic seed sequence used by this run."""
        return range(self.cfg.seed, self.cfg.seed + self.cfg.episode_num)

    def run(self) -> list[EpisodeSummary]:
        """Launch Isaac and synthesize all configured episodes."""
        launcher = self.create_launcher()
        sim_app = launcher.app
        action_manager = self.build_action_manager()

        try:
            summaries = []
            for episode_index, seed in enumerate(self.iter_episode_seeds()):
                summaries.append(
                    self.run_episode(
                        episode_index=episode_index,
                        seed=seed,
                        sim_app=sim_app,
                        action_manager=action_manager,
                    )
                )
            return summaries
        finally:
            launcher.close()

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
        from robo_orchard_sim.tasks.trajs_gen.atomic_action_manager import (
            AtomicActionManagerCfg,
        )

        return AtomicActionManagerCfg(debug_vis=self.cfg.debug_vis)()

    def run_episode(
        self,
        *,
        episode_index: int,
        seed: int,
        sim_app: Any,
        action_manager: Any,
    ) -> EpisodeSummary:
        """Sample assets, create an env, and run one atomic-action episode."""
        from robo_orchard_sim.envs.env_base import IsaacEnvContextManager
        from robo_orchard_sim.task_suite.registry import (
            build_task_atomic_action_plan,
        )
        from robo_orchard_sim.tasks.trajs_gen.atomic_action_manager import (
            ActionStatusLogger,
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
        with env_manager as env:
            plan = build_task_atomic_action_plan(
                task_name=self.cfg.task,
                orchard_env=orchard_env,
            )
            action_manager.clear()
            action_manager.register(plan)

            _ = env.reset(seed=seed)
            self.settle_until_recording_starts(env)
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
                )
            )

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

        print(
            f"Episode {episode_index + 1} finished: "
            f"steps={steps}, stop_reason={stop_reason}, "
            f"success={validator_output.success}"
        )
        return EpisodeSummary(
            episode_index=episode_index,
            seed=seed,
            steps=steps,
            stop_reason=stop_reason,
            success=bool(validator_output.success),
        )

    def build_orchard_env(self, *, seed: int):
        """Build a fresh OrchardEnv with assets sampled by ``seed``."""
        from robo_orchard_sim.asset_manager.registry import AssetRegistry
        from robo_orchard_sim.asset_manager.resolver.asset_resolver import (
            AssetResolver,
            AssetResolverError,
        )
        from robo_orchard_sim.task_suite.registry import build_task

        registry = AssetRegistry(self.cfg.asset_root)
        resolver = AssetResolver(
            registry=registry,
            splits=None,  # TODO: support splits when task configs expose them.
            rng=np.random.default_rng(seed),
        )
        config_path = self._resolve_config_path()

        try:
            return build_task(
                task_name=self.cfg.task,
                resolver=resolver,
                config_path=config_path,
            )
        except KeyError as exc:
            raise SystemExit(f"\nERROR: {exc}") from exc
        except AssetResolverError as exc:
            raise SystemExit(
                f"\nERROR resolving assets from registry: {exc}\n"
                "Check the asset_configs: block in the task YAML, or pass "
                "a different --config."
            ) from exc
        except ValueError as exc:
            raise SystemExit(f"\nERROR: {exc}") from exc

    def prepare_recording(
        self,
        orchard_env: Any,
        episode_index: int,
        seed: int,
    ) -> None:
        """Configure recording to start after reset-time scene settling."""
        from robo_orchard_sim.envs.managers.record import (
            StationaryEpisodeRecordControllerCfg,
        )

        orchard_env.configure_recording(
            file_path=self.episode_record_dir(
                episode_index=episode_index,
                seed=seed,
            ),
            controller=StationaryEpisodeRecordControllerCfg(
                max_wait_step=self.cfg.settle_steps,
            ),
        )

    def settle_until_recording_starts(self, env: Any) -> None:
        """Step after reset until stationary recording starts or times out."""
        record_manager = env.record_manager
        for _ in range(self.cfg.settle_steps):
            _ = env.step()
            if record_manager is not None and record_manager.running:
                return

    def build_validator_actors(
        self,
        *,
        runtime_task: Any,
        scene: Any,
    ) -> list[Any]:
        """Build validator actor snapshots from the runtime scene."""
        from robo_orchard_sim.tasks.validators.base import ValidatorActor

        actor_names = runtime_task.get_validator_actor_names()
        return [
            ValidatorActor.from_rigid_object(name, scene[name])
            for name in actor_names
        ]

    def build_validator(self, *, runtime_task: Any, actors: list[Any]) -> Any:
        """Build the task validator bound to the current actor snapshots."""
        return runtime_task.build_validator(actors=actors)

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

    def translate_actions(self, act: Any, env: Any) -> dict[str, Any]:
        """Translate atomic-action output into env action terms."""
        # TODO: remove hard code of robot seeting in future
        robot = env.scene[_ROBOT_SCENE_NAME]
        left_arm_ids, _ = robot.find_joints(["left_joint[1-6]"])
        left_gripper_ids, _ = robot.find_joints(["left_joint[7-8]"])
        right_arm_ids, _ = robot.find_joints(["right_joint[1-6]"])
        right_gripper_ids, _ = robot.find_joints(["right_joint[7-8]"])

        joint_pos = robot.data.joint_pos
        action = {
            "left_robot_joint_position": joint_pos[:, left_arm_ids].clone(),
            "left_robot_gripper_control": joint_pos[
                :,
                left_gripper_ids,
            ].clone(),
            "right_robot_joint_position": joint_pos[:, right_arm_ids].clone(),
            "right_robot_gripper_control": joint_pos[
                :,
                right_gripper_ids,
            ].clone(),
        }

        if _LEFT_ARM_KEY in act:
            action["left_robot_joint_position"] = act[_LEFT_ARM_KEY][:, :-2]
            action["left_robot_gripper_control"] = act[_LEFT_ARM_KEY][:, -2:]
        if _RIGHT_ARM_KEY in act:
            action["right_robot_joint_position"] = act[_RIGHT_ARM_KEY][:, :-2]
            action["right_robot_gripper_control"] = act[_RIGHT_ARM_KEY][:, -2:]
        return action

    def robot_is_stationary(self, env: Any) -> bool:
        """Return whether all configured robot bodies have low velocity."""
        robot = env.scene[_ROBOT_SCENE_NAME]
        body_link_vel_w = robot.data.body_link_vel_w

        lin_vel = torch.linalg.vector_norm(body_link_vel_w[..., :3], dim=-1)
        ang_vel = torch.linalg.vector_norm(body_link_vel_w[..., 3:6], dim=-1)
        return bool(torch.all(lin_vel < 0.06) and torch.all(ang_vel < 0.1))

    def wait_until_robot_stationary(self, env: Any, sim_app: Any) -> None:
        """Keep stepping after task completion until the robot settles."""
        for _ in range(self.cfg.settle_steps):
            if not sim_app.is_running():
                return
            if self.robot_is_stationary(env):
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
    ) -> tuple[int, str, ValidatorOutput]:
        stop_reason = "max_steps"
        steps = 0
        validator_output = ValidatorOutput(
            success=False,
            progress=0.0,
            metrics={},
        )
        while steps < self.cfg.max_steps:
            if not sim_app.is_running():
                stop_reason = "sim_app_stopped"
                break

            manager_actions, state = manager.get_action(env)
            for log_line in status_logger.collect(
                running_actions=state.running_actions,
                step_idx=steps,
            ):
                print(log_line)

            actions = self.translate_actions(manager_actions, env)
            _ = env.step(actions)
            validator_output = validator.evaluate(env, env_idx=0)
            steps += 1

            # if validator_output.success:
            #     stop_reason = "success"
            #     self.wait_until_robot_stationary(env, sim_app)
            #     break
            if not state.env_busy:
                stop_reason = "action_plan_complete"
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
            self._record_run_dir,
            f"episode_{episode_index:04d}_seed_{seed}",
        )

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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the data synthesis example."""
    env_default = os.environ.get(_ASSET_ROOT_ENV)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=(
            "Registered task namespace to build (e.g. place_a2b_easy, "
            "pick_category, pick_attribute, pick_disambiguation)."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML path overriding the selected task config_path.",
    )
    parser.add_argument(
        "--asset-root",
        type=str,
        default=env_default,
        required=env_default is None,
        help=(
            f"Asset library root. Defaults to the ${_ASSET_ROOT_ENV} env "
            "var and is required if that env var is not set."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="First RNG seed for reproducible asset sampling.",
    )
    parser.add_argument(
        "--episodes",
        dest="episode_num",
        type=int,
        default=1,
        help="How many episodes to synthesize.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum manager-driven env steps per episode.",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="logs/data_synthesis",
        help=(
            "Base output directory for synthesized MCAP recordings. "
            "Episodes are written below <record-dir>/<task>_<timestamp_ms>/."
        ),
    )
    parser.add_argument(
        "--disable-recording",
        dest="enable_recording",
        action="store_false",
        help="Run the synthesis loop without enabling MCAP recording.",
    )
    parser.add_argument(
        "--output-config-dir",
        type=str,
        default="configs/data_synthesis",
        help="Directory for per-episode serialized env configs.",
    )
    parser.set_defaults(
        enable_recording=True,
        enable_cameras=True,
        headless=True,
    )
    return parser


def main() -> None:
    """Run the data synthesis example from CLI arguments."""
    args = build_arg_parser().parse_args()
    cfg_kwargs = vars(args)
    launch = LaunchConfig(
        headless=cfg_kwargs.pop("headless"),
        enable_cameras=cfg_kwargs.pop("enable_cameras"),
    )
    cfg = DataSynthesisConfig(**cfg_kwargs, launch=launch)
    summaries = DataSynthesisRunner(cfg).run()

    print("Data synthesis finished successfully.")
    for summary in summaries:
        print(
            f"episode={summary.episode_index}, seed={summary.seed}, "
            f"steps={summary.steps}, stop_reason={summary.stop_reason}, "
            f"success={summary.success}"
        )


if __name__ == "__main__":
    main()
