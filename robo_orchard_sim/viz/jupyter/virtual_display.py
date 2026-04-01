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
from __future__ import annotations

from robo_orchard_core.viz.jupyter.virtual_display import IpyVirtualDisplay

from robo_orchard_sim.sim.context import SimulationContext


class IsaacIpyVirtualDisplay(IpyVirtualDisplay):
    def __init__(
        self,
        sim_ctx: SimulationContext,
        max_fps: int = 10,
        *args,
        **kwargs,
    ):
        self._sim_ctx: SimulationContext = sim_ctx
        super().__init__(*args, max_fps=max_fps, **kwargs)

    def _render(self):
        # seems that need to render multiple times to get the correct response
        # from the simulation app
        for _ in range(3):
            self._sim_ctx.render()
        super()._render()
