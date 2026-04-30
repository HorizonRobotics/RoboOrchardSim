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

"""Pin the invariant: importing asset_registry must not pull in isaaclab.

asset_registry is intended to be a lightweight metadata layer usable from
CLI tools and subprocess tests without a full Isaac Sim stack. Heavyweight
dependencies (isaaclab, isaacsim, orchard_env) must be lazily imported
only when actually needed (e.g., AssetRegistry.build_spec).
"""

import subprocess
import sys

_HEAVY_MODULES = ("isaaclab", "isaacsim")


def test_asset_registry_import_does_not_load_isaaclab():
    """Importing asset_registry in a fresh process must not load isaaclab."""
    code = (
        "import robo_orchard_sim.asset_manager.registry\n"
        "import sys\n"
        "leaked = [\n"
        "    m for m in sys.modules\n"
        "    if any(m == h or m.startswith(h + '.') for h in "
        f"{list(_HEAVY_MODULES)!r})\n"
        "]\n"
        "assert not leaked, f'heavy modules leaked: {leaked}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"asset_registry import leaked heavy modules.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
