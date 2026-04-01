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

"""Basic Env for Isaac."""

from __future__ import annotations
import builtins

import carb
import isaacsim.core.utils.torch as torch_utils
import torch
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.utils.timer import Timer
from robo_orchard_core.envs.env_base import EnvStepReturn
from typing_extensions import Any, Generic, Sequence, TypeVar

from robo_orchard_sim.cfg_wrappers.envs.env_cfg import (
    BaseEnvWindow,
    SimulationCfg,
    ViewerCfg,
)
from robo_orchard_sim.cfg_wrappers.scenes_cfg import InteractiveSceneCfg
from robo_orchard_sim.models.scenes.interactive_scene import InteractiveScene
from robo_orchard_sim.sim_ctx import (
    SimulationContext,
    SimulationContextManager,
)
from robo_orchard_sim.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType,
    ClassType_co,
)

EnvStepReturnTypeT = TypeVar("EnvStepReturnTypeT", bound=EnvStepReturn)
IsaacEnvCfgType_co = TypeVar(
    "IsaacEnvCfgType_co", bound="IsaacEnvCfg", covariant=True
)

#:
InteractiveSceneCfgType_co = TypeVar(
    "InteractiveSceneCfgType_co",
    bound=InteractiveSceneCfg,
    covariant=True,
)


class IsaacEnv(
    ClassInitFromConfigMixin,
    Generic[IsaacEnvCfgType_co],
):
    """The Env wrapper for All Envs from Isaac Lab.

    This class extends `EnvBase` to use base env implementation from Isaac
    Lab implementation.

    Difference from the original implementation:
    - The configuration class is `IsaacManagerBasedEnvCfg`.
    - The scene manager is wrapped with `InteractiveScene`.

    User should extend this class to implement `step` method.

    """

    def __init__(self, cfg: IsaacEnvCfgType_co):
        self.cfg = cfg
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.seed(self.cfg.seed)
        else:
            carb.log_warn(
                "Seed not set for the environment. The environment "
                "creation may not be deterministic."
            )

        # create a simulation context to control the simulator
        # The simulation context is a singleton and we use our modified
        # version of it
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)  # type: ignore
        else:
            # simulation context should only be created before the environment
            # when in extension mode
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:  # type: ignore
                if SimulationContext.instance().cfg != self.cfg.sim:  # type: ignore
                    raise RuntimeError(
                        "Simulation context already exists. Cannot create a new one."  # noqa: E501
                    )
            self.sim: SimulationContext = SimulationContext.instance()  # type: ignore

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(
            f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}"  # noqa
        )
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) "
                f"is smaller than the decimation  ({self.cfg.decimation}). "
                "Multiple multiple render calls will happen for each "
                "environment step.  If this is not intended, set the render "
                "interval to be equal to the decimation."
            )
            carb.log_warn(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # generate scene
        # We use our modified version of the InteractiveScene class
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)
        print("[INFO]: Scene assets: ")
        for asset in self.scene.keys():
            print(f"\t{asset}")

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function
        # will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI
        # functionalities even for  non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(
                self,  # type: ignore
                self.cfg.viewer,
            )
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes
        # TensorAPIs
        # note: when started in extension mode, first call sim.reset_async()
        # and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:  # type: ignore
            print(
                "[INFO]: Starting the simulation. This may take a few seconds. Please wait..."  # noqa: E501
            )
            with Timer(
                "[INFO]: Time taken for simulation start", "simulation_start"
            ):
                self.sim.reset()
            # add timeline event to load managers

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            self._window = self.cfg.ui_window_class_type(
                self, window_name="IsaacLab"
            )
        else:
            # if no window, then we don't need to store the window
            self._window = None

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation
        is happening.
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None
    ) -> None:
        if seed is not None:
            self.seed(seed)

        if env_ids is None:
            env_ids = list(range(self.num_envs))

        self._reset_idx(env_ids)
        # if sensors are added to the scene, make sure we render to reflect
        # changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)

        # iterate over all managers and reset them

        # self.extras["log"] = dict()

    def close(self):
        """Cleanup for the environment."""
        if not self._is_closed:
            # destructor is order-sensitive
            del self.viewport_camera_controller
            # del self.action_manager
            # del self.observation_manager

            # hotfix for the issue where some assets are not deleted
            # in the scene
            if hasattr(self.scene, "delete_all_assets"):
                self.scene.delete_all_assets()

            del self.scene
            # clear callbacks and instance
            self.sim.clear_all_callbacks()
            self.sim.clear_instance()
            # destroy the window
            if self._window is not None:
                self._window = None
            # update closing status
            self._is_closed = True

    def step(self):
        """Execute one step of the environment."""

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()

            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)


#:
IsaacEnvType_co = TypeVar("IsaacEnvType_co", bound=IsaacEnv, covariant=True)


class IsaacEnvCfg(
    ClassConfig[IsaacEnvType_co],
    Generic[IsaacEnvType_co, InteractiveSceneCfgType_co],
):
    """The configuration for IsaacManagerBasedEnv.

    This configuration contains the configuration for the simulation, the
    viewer, and the scene.

    Template Args:
        InteractiveSceneCfgType_co: The type of the scene configuration.
        IsaacEnvType_co: The type of the environment.
    """

    class_type: ClassType_co[IsaacEnvType_co]

    viewer: ViewerCfg = ViewerCfg()

    sim: SimulationCfg = SimulationCfg()
    """Physics simulation configuration. Default is SimulationCfg()."""

    ui_window_class_type: ClassType[Any] | None = BaseEnvWindow

    seed: int | None = None

    decimation: int
    """Number of simulation updates and control action @ sim dt per
    environment step.

    For instance, if the simulation dt is 0.01s and the env dt is 0.1s,
    then the decimation is 10. This means that the env is updated every
    10 simulation steps. If control actions are available, they are applied
    every simulation step.
    """

    rerender_on_reset: bool = False

    scene: InteractiveSceneCfgType_co
    """Scene settings.

    Can be a subclass of `InteractiveSceneCfg`.
    """


#:
IsaacEnvCfgType_co = TypeVar(
    "IsaacEnvCfgType_co", bound=IsaacEnvCfg, covariant=True
)


class IsaacEnvContextManager(
    Generic[IsaacEnvType_co, InteractiveSceneCfgType_co]
):
    """Context manager for creating an IsaacManagerBasedEnv object.

    Template Args:
        InteractiveSceneCfgType_co: The type of the scene configuration.
        IsaacEnvType_co: The type of the environment.

    Args:
        cfg (IsaacManagerBasedEnvCfg): The configuration for the environment.
        with_new_stage (bool): Whether to create a new stage. Default is False.
        disable_exit_on_stop (bool): Whether to disable the exit on stop.
            Default is True. Note that in isaac lab, the simulation will exit
            when the simulation stops. This flag disables that behavior.

    """

    def __init__(
        self,
        cfg: IsaacEnvCfg[IsaacEnvType_co, InteractiveSceneCfgType_co],
        with_new_stage: bool = False,
        disable_exit_on_stop: bool = True,
    ):
        self.cfg = cfg
        self._sim_ctx_manager = SimulationContextManager(
            cfg.sim,
            with_new_stage=with_new_stage,
            disable_exit_on_stop=disable_exit_on_stop,
        )

    @property
    def env(self) -> IsaacEnvType_co:
        if not hasattr(self, "_env"):
            raise ValueError(
                "The environment ('env') has not been initialized. "
                "Please ensure this property is accessed within a 'with' block "  # noqa: E501
                "of the IsaacEnvContextManager, as the environment is only "
                "initialized during the '__enter__' phase of the context manager."  # noqa: E501
            )
        return self._env

    def __enter__(self) -> IsaacEnvType_co:
        # enter to initialize simulation context to global singleton
        self._sim_ctx_manager.__enter__()
        # initialize environment. The simulation context is a singleton
        # and will be shared with the environment.
        self._env = self.cfg()
        return self.env

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_env"):
            self._env.close()
        self._sim_ctx_manager.__exit__(exc_type, exc_val, exc_tb)
