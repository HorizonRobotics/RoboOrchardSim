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

import argparse
import atexit
import importlib
import os
import sys
import threading
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Generator

# Set environment variable OMNI_KIT_ACCEPT_EULA=YES and
# OMNI_KIT_ALLOW_ROOT=1
# os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
# os.environ["OMNI_KIT_ALLOW_ROOT"] = "1"


from isaaclab.app import AppLauncher  # isort:skip # noqa: E402
from robo_orchard_core.utils.misc import (  # isort:skip # noqa: E402
    SingletonMixin,
)

if TYPE_CHECKING:
    from isaacsim.simulation_app import SimulationApp
    from pyvirtualdisplay.smartdisplay import SmartDisplay as Display


# In IsaacSim 5.1 headless subprocess exit, the stage-transition step inside
# `SimulationApp.close()` (both `close_stage()` and `new_stage()` go through
# it) blocks on an async task that never fires when the render loop is not
# active. NVIDIA's own workaround for this upstream bug is a hard exit after a
# grace period. See launcher.py history / plan doc for details.
_CLOSE_APP_WATCHDOG_SECONDS = 5.0


def close_app(
    simulation_app: "SimulationApp",
    wait_for_replicator: bool = True,  # type: ignore
):
    """Close Simulation App via the official IsaacSim shutdown path.

    Delegates to `simulation_app.close()` (IsaacSim 5.1 fixed the historical
    crash-on-exit issues that the earlier manual shutdown workaround was
    written to avoid).

    A daemon watchdog timer force-exits the process if the close call does
    not return within :data:`_CLOSE_APP_WATCHDOG_SECONDS`, working around
    the IsaacSim 5.1 headless stage-transition deadlock.

    ``wait_for_replicator`` is kept for backward compatibility but no
    longer used - replicator cleanup is handled inside SimulationApp.close.
    """
    del wait_for_replicator

    def _force_exit() -> None:
        print(
            "[close_app] SimulationApp.close() exceeded "
            f"{_CLOSE_APP_WATCHDOG_SECONDS}s, forcing process exit.",
            flush=True,
        )
        os._exit(0)

    watchdog = threading.Timer(_CLOSE_APP_WATCHDOG_SECONDS, _force_exit)
    watchdog.daemon = True
    watchdog.start()
    try:
        simulation_app.close()
    finally:
        watchdog.cancel()


class LauncherCallback(metaclass=ABCMeta):
    """Launcher Callback class.

    This class is an abstract class that should be inherited by the user to
    create a callback function to be executed by the launcher.

    """

    @abstractmethod
    def __call__(self, simulation_app: "SimulationApp"):
        """Main function to be implemented by the user.

        Args:
            simulation_app (isaaclab.app.SimulationApp): Simulation app.

        """
        pass


class FileLauncherCallback(LauncherCallback):
    """File Launcher Callback.

    This class is a concrete implementation of the LauncherCallback class that
    is used to execute a file as a callback function.

    """

    def __init__(self, file: str):
        """Initialize File Launcher Callback.

        Args:
            file (str): File to execute.

        """
        # check if file exists
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found")
        self.file = file

    def __call__(self, simulation_app: "SimulationApp"):
        """Execute file as callback function.

        Args:
            simulation_app (isaaclab.app.SimulationApp): Simulation app.

        """
        # add file to sys.path and import it
        sys.path.insert(0, os.path.dirname(self.file))
        file_name = os.path.basename(self.file)
        module_name = os.path.splitext(file_name)[0]
        module = importlib.import_module(module_name)
        callback = getattr(module, "main", None)

        if callback is not None:
            callback(simulation_app)
        else:
            raise AttributeError(f"Function main not found in {self.file}")


@contextmanager
def isaac_lab_launcher_context(
    launcher_args: argparse.Namespace | dict | None = None, **kwargs
) -> Generator["SimulationApp", None, None]:
    """Isaac Lab Launcher Context."""

    simulation_app = None

    try:
        app_launcher = AppLauncher(launcher_args, **kwargs)
        simulation_app = app_launcher.app
        yield simulation_app
    except Exception as e:
        raise e
    finally:
        if simulation_app is not None:
            close_app(simulation_app)


def isaac_lab_launcher(
    callback: LauncherCallback | Callable[["SimulationApp"], None],
    launcher_args: argparse.Namespace | dict | None = None,
    **kwargs,
):
    """Isaac Lab Launcher.

    The launcher is a wrapper around the AppLauncher class, which is used to
    launch the Isaac Lab application and separate the application logic from
    the launcher logic.

    User should provide callback function
    `main` in the script file, and `main` function should accept
    `simulation_app` as an argument.

    Args:
        callback (LauncherCallback): Callback function.
        launcher_args (argparse.Namespace | dict | None): Launcher arguments.
        parser (argparse.ArgumentParser): Argument parser.

    """

    # launch omniverse app
    with isaac_lab_launcher_context(launcher_args, **kwargs) as simulation_app:
        try:
            callback(simulation_app)
        except Exception as e:
            import traceback

            error_msg = (
                f"Error in callback function: {e},"
                + f" {traceback.format_exc()}"
            )

            print(error_msg)
            raise e


def main():
    """Main function of Launcher.

    The launcher is a wrapper around the AppLauncher class, which is used to
    launch the Isaac Lab application and separate the application logic from
    the launcher logic.

    User should provide callback function
    `main` in the script file, and `main` function should accept
    `simulation_app` as an argument.

    """

    parser = argparse.ArgumentParser(description="Isaac Lab Launcher")
    # add file argument to be able to load a file
    file_arg_group = parser.add_argument_group("File to load")
    file_arg_group.add_argument(
        "file", type=str, help="File to load to execute."
    )

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    callback = FileLauncherCallback(args_cli.file)

    isaac_lab_launcher(callback, args_cli)


class SimpleIsaacAppLauncher(SingletonMixin):
    """Isaac Lab Launcher.

    The launcher is a wrapper around the AppLauncher class to launch the Isaac
    application with virtual display.

    The isaac sim should be launched first before importing any other modules
    from the isaac simulator.

    Args:
        launcher_args (argparse.Namespace | dict | None): Launcher arguments.
            For more information, see
            :py:class:`isaaclab.app.app_launcher.AppLauncher`.

        virtual_display (bool): Whether to use virtual display. If True, a
            virtual display will be created with xvfb backend and $DISPLAY
            environment variable will be set to the virtual display.
            Be careful that the virtual display will increate the
            resource usage of the simulator significantly.
            Default is False.

    Usage:

    .. code-block:: python

        # One line of code is enough to launch the Isaac application
        # in a container
        from robo_orchard_sim.launcher import SimpleIsaacAppLauncher

        launcher = SimpleIsaacAppLauncher(
            headless=True, enable_cameras=True, virtual_display=False
        )

        # Now you can import other modules from the isaac simulator
        from isaacsim import SimulationApp

        ...

    """

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            return cls._instance

        raise RuntimeError("IsaacAppLauncher is a singleton class")

    def __init__(
        self,
        launcher_args: argparse.Namespace | dict | None = None,
        virtual_display: bool = False,
        **kwargs,
    ):
        self._closed = False
        self._display = None

        if virtual_display:
            from pyvirtualdisplay.smartdisplay import SmartDisplay as Display

            self._display = Display(backend="xvfb", manage_global_env=True)
            self._display.start()

        self._app_gen = isaac_lab_launcher_context(
            launcher_args=launcher_args, **kwargs
        )
        self._app = self._app_gen.__enter__()
        atexit.register(self.close)

    @property
    def app(self) -> "SimulationApp":
        """The isaac simulation app."""
        return self._app

    @property
    def virtual_display(self) -> "Display|None":
        """The virtual display."""
        return self._display

    def close(self) -> None:
        """Close the launcher and release singleton ownership once."""
        if getattr(self, "_closed", False):
            return

        self._closed = True
        print("exiting isaac app launcher...")

        app_gen = getattr(self, "_app_gen", None)
        if app_gen is not None:
            app_gen.__exit__(None, None, None)
            self._app_gen = None

        display = getattr(self, "_display", None)
        if display is not None:
            print("stop virtual display...")
            display.stop()
            self._display = None

        launcher_cls = type(self)
        if getattr(launcher_cls, "_instance", None) is self:
            delattr(launcher_cls, "_instance")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    main()
