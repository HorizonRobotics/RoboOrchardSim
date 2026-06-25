# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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


import os
import tempfile
import traceback

import carb
import isaacsim.core.utils.stage as stage_utils
from isaaclab.sim import SimulationContext as LabSimulationContext
from robo_orchard_core.utils.timer import FPSCounter

from robo_orchard_sim.ext.cfg_wrappers.sim.simulation_cfg import SimulationCfg

__all__ = ["SimulationContext", "SimulationContextManager"]


class SimulationContext(LabSimulationContext):
    """The SimulationContext class extended from isaac lab SimulationContext.

    This class add `unsubscribe_exit_on_stop` to avoid closing the app
    when the simulation stops. This is useful when running multiple stages
    in a single app. For example, creating a new stage with some assets
    and then clearing the stage to create a new one.

    """

    def __init__(self, cfg: SimulationCfg | None = None):
        super().__init__(cfg)
        self._fps_counter = FPSCounter()

    @classmethod
    def instance(cls) -> LabSimulationContext | None:
        """Return the current instance of the SimulationContext.

        Returns:
            LabSimulationContext: The current instance of the
                SimulationContext.
        """
        return cls._instance

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    def _physics_timer_callback_fn(self, step_size: float):
        """Override the physics timer callback function to enable fps counter.

        It will update the fps counter every time the physics timer callback.
        """
        super()._physics_timer_callback_fn(step_size)  # type: ignore
        self._fps_counter.update()

    def _timeline_timer_callback_fn(self, event):
        """Override the timeline callback function to reset the fps counter.

        FPS counter will be reset every time when STOP event is triggered.
        """
        super()._timeline_timer_callback_fn(event)
        self._fps_counter.reset()

    def unsubscribe_exit_on_stop(self):
        """Unsubscribe the exit on stop handle."""
        if self._app_control_on_stop_handle is not None:
            self._app_control_on_stop_handle.unsubscribe()
            self._app_control_on_stop_handle = None

    async def render_async(
        self, mode: LabSimulationContext.RenderMode | None = None
    ):
        """Async version of render function.

        Almost the same as the original render function but with async
        update call.

        Warning:
            Currently not working as expected.

        TODO: Fix or remove this function.

        """

        # check if we need to change the render mode
        if mode is not None:
            self.set_render_mode(mode)
        # render based on the render mode
        if self.render_mode == self.RenderMode.NO_GUI_OR_RENDERING:
            # we never want to render anything here (this is for complete
            # headless mode)
            pass
        elif self.render_mode == self.RenderMode.NO_RENDERING:
            # throttle the rendering frequency to keep the UI responsive
            self._render_throttle_counter += 1
            if (
                self._render_throttle_counter % self._render_throttle_period
                == 0
            ):
                self._render_throttle_counter = 0
                # here we don't render viewport so don't need to flush fabric
                # data
                # note: we don't call super().render() anymore because they do
                # flush the fabric data
                self.set_setting("/app/player/playSimulations", False)
                await self._app.next_update_async()  # type: ignore #
                self.set_setting("/app/player/playSimulations", True)
        else:
            # manually flush the fabric data to update Hydra textures
            if self._fabric_iface is not None:
                self._fabric_iface.update(0.0, 0.0)
            # render the simulation
            # note: we don't call super().render() anymore because they do
            # above operation inside and we don't want to do it twice.
            # We may remove it once we drop support for Isaac Sim 2022.2.
            self.set_setting("/app/player/playSimulations", False)
            await self._app.next_update_async()  # type: ignore #
            self.set_setting("/app/player/playSimulations", True)


class SimulationContextManager:
    """Context manager for simulation to support with statement.

    When entering the context, it will create a new stage if `with_new_stage`

    Args:
        cfg (SimulationCfg): The configuration for the environment.
        with_new_stage (bool): Whether to create a new stage. Default is False.
        disable_exit_on_stop (bool): Whether to disable the exit on stop.
            Default is True. Note that in isaac lab, the simulation will exit
            when the simulation stops. This flag disables that behavior.

    """

    def __init__(
        self,
        cfg: SimulationCfg,
        with_new_stage: bool = False,
        disable_exit_on_stop: bool = True,
    ):
        self.cfg = cfg
        self.with_new_stage = with_new_stage
        self.disable_exit_on_stop = disable_exit_on_stop
        self._sim: SimulationContext | None = None
        self._has_new_stage = False
        self._has_old_stage = False
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    @property
    def sim(self) -> SimulationContext:
        """The simulation object."""

        if self._sim is None:
            raise RuntimeError("Simulation object is not created.")
        return self._sim

    def __enter__(self) -> SimulationContext:
        try:
            if self.with_new_stage:
                if stage_utils.get_current_stage() is not None:
                    # create a temp folder and write stage into it.
                    self._has_old_stage = True
                    temp_dir = tempfile.TemporaryDirectory()
                    self._temp_dir = temp_dir
                    carb.log_info(
                        "Current stage exists. "
                        f"Saving it to a temp folder: {temp_dir.name}"
                    )
                    stage_utils.save_stage(
                        os.path.join(temp_dir.name, "temp_stage.usd")
                    )

                    stage_utils.close_stage()

                self._has_new_stage = stage_utils.create_new_stage()
                if not self._has_new_stage:
                    raise RuntimeError("Failed to create a new stage.")
            else:
                self._has_new_stage = False

            # Clear the current instance if exists
            if SimulationContext.instance() is not None:
                sim: SimulationContext = SimulationContext.instance()
                sim.stop()
                sim.clear_all_callbacks()
                sim.clear()
                SimulationContext.clear_instance()

            self._sim = SimulationContext(self.cfg)  # type: ignore
            if self.disable_exit_on_stop:
                self._sim.unsubscribe_exit_on_stop()  # type: ignore

            return self.sim

        except Exception:
            carb.log_error(traceback.format_exc())
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._sim is not None:
            sim = self._sim
            sim.clear_all_callbacks()
            sim.clear_instance()

            # Clear the stage
        if self._has_new_stage:
            # stage_utils.clear_stage() # this seems to be unnecessary
            stage_utils.close_stage()
            if self._has_old_stage:
                assert self._temp_dir is not None
                assert stage_utils.open_stage(
                    os.path.join(self._temp_dir.name, "temp_stage.usd")
                )
                self._temp_dir.cleanup()
