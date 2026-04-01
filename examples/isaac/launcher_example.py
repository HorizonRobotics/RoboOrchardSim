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

"""This script demonstrates how to use RoboOrchard launcher.

User should provide callback function `main` in the script file, and `main`
function should accept `simulation_app` as an argument.

This example is modified version of
"IsaacLab/source/standalone/tutorials/00_sim/create_empty.py.",
which demonstrates how to create a simple stage in Isaac Sim.


Usage:
    `python3 -m robo_orchard_sim.launcher examples/isaac/launcher_example.py`
    or
    `RoboOrchard-SimLauncher examples/isaac/launcher_example.py`

"""

import time

from isaaclab.sim import SimulationCfg, SimulationContext
from isaacsim.simulation_app import SimulationApp


def main(simulation_app: SimulationApp):
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1, 1, 1], [0.0, 0.0, 0.0])

    sim.reset()
    print("Isaac sim Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()
        print("Isaac sim will exits in 2 seconds...")
        time.sleep(2)
        break
    print("Isaac sim exited...")
