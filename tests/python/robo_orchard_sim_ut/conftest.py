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


import os
from typing import TYPE_CHECKING

import pytest
import xdist

from robo_orchard_sim.launcher import SimpleIsaacAppLauncher

# from robo_orchard_core.utils.logging import LoggerManager

# logger = LoggerManager().get_child(__name__)

# LoggerManager().set_level(logging.DEBUG)

if TYPE_CHECKING:
    from omni.isaac.kit import SimulationApp  # noqa: E402

launcher = SimpleIsaacAppLauncher(enable_cameras=True, virtual_display=True)

# launcher = SimpleIsaacAppLauncher(
#     enable_cameras=True, virtual_display=False, headless=True, livestream=1
# )


@pytest.fixture(scope="session", autouse=True)
def app() -> "SimulationApp":  # type: ignore
    """Fixture to create a SimulationApp instance for the entire test session.

    The simulation app is a singleton instance that is created once and used
    throughout the whole test session. Any test that requires a simulation app
    can use this fixture to get the instance.
    """
    # virtual_display=True is required to manager global env
    # to a virtual display for testing.
    global launcher
    yield launcher.app  # type: ignore
    del launcher


@pytest.fixture(scope="session")
def is_xdist_worker(request) -> bool:
    return xdist.is_xdist_worker(request)


def share_fixtures(file_ends: str = "fixtures.py"):
    """Share fixtures from all test modules.

    Args:
        file_ends (str): The file ending to search for.

    """
    pytest_plugins = []
    here = os.path.dirname(os.path.realpath(__file__))

    def _as_module(root: str, path: str) -> str:
        path = os.path.join(root, path)
        path = path.replace(here, "")
        path = path.replace(".py", "")
        path = path.replace(os.path.sep, ".")[1:]
        return path

    for root, _, files in os.walk(here, topdown=True):
        pytest_plugins += [
            _as_module(root, f) for f in files if f.endswith(file_ends)
        ]
    return pytest_plugins


# pytest_plugins = share_fixtures()


# print("global shared fixtures: ", pytest_plugins)
