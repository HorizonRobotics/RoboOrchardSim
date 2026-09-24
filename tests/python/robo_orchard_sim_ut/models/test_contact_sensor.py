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

"""Contact sensor behavior with metadata supplied at the PhysX boundary."""

import pytest
import torch

from robo_orchard_sim.ext.cfg_wrappers.assets_cfg import RigidObjectCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.schemas.schemas_cfg import (
    CollisionPropertiesCfg,
    RigidBodyPropertiesCfg,
)
from robo_orchard_sim.ext.cfg_wrappers.sim.simulation_cfg import SimulationCfg
from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.shapes_cfg import CuboidCfg
from robo_orchard_sim.ext.models.scenes.asset_scene import AssetSceneCfg
from robo_orchard_sim.ext.models.scenes.interactive_scene import (
    InteractiveScene,
)
from robo_orchard_sim.ext.models.sensors.contact_sensor import ContactSensorCfg
from robo_orchard_sim.sim_ctx import SimulationContextManager


@pytest.fixture
def contact_sensor(app):
    with SimulationContextManager(
        SimulationCfg(), with_new_stage=True, disable_exit_on_stop=True
    ) as simulation:
        scene = InteractiveScene(
            AssetSceneCfg(
                num_envs=1,
                env_spacing=1.0,
                assets={
                    "objects": {
                        "cube": RigidObjectCfg(
                            prim_path="{ENV_REGEX_NS}/Cube",
                            spawn=CuboidCfg(
                                size=(0.1, 0.1, 0.1),
                                rigid_props=RigidBodyPropertiesCfg(
                                    disable_gravity=True
                                ),
                                collision_props=CollisionPropertiesCfg(),
                                activate_contact_sensors=True,
                            ),
                        ),
                    },
                    "sensors": {
                        "contact": ContactSensorCfg(
                            prim_path="{ENV_REGEX_NS}/Cube",
                            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
                            track_contact_points=True,
                            track_friction_forces=True,
                        ),
                    },
                },
            )
        )
        simulation.reset()
        simulation.step(render=False)
        yield scene["sensors/contact"]


@pytest.fixture
def inject_contact_data(contact_sensor, monkeypatch):
    def inject(count, start, capacity, *, friction=False):
        data = torch.tensor(
            [1.0, 2.0, 3.0], device=contact_sensor.device
        ).repeat(capacity, 1)
        counts = torch.tensor(
            [[count]], dtype=torch.int32, device=contact_sensor.device
        )
        starts = torch.tensor(
            [[start]], dtype=torch.int32, device=contact_sensor.device
        )
        if friction:
            monkeypatch.setattr(
                contact_sensor.contact_physx_view,
                "get_friction_data",
                lambda **kwargs: (data, None, counts, starts),
            )
        else:
            monkeypatch.setattr(
                contact_sensor.contact_physx_view,
                "get_contact_data",
                lambda **kwargs: (None, data, None, None, counts, starts),
            )

    return inject


def test_contact_sensor_config_default_capacity_is_1024():
    config = ContactSensorCfg(prim_path="/World/Finger")
    assert config.max_contact_data_count_per_prim == 1024


@pytest.mark.parametrize(
    "count,start,capacity,friction",
    [
        (64, 0, 64, False),
        (186, 0, 1024, False),
        (0, -1, 64, False),
        (2, 62, 64, True),
    ],
)
def test_contact_sensor_valid_metadata_preserves_aggregation(
    contact_sensor, inject_contact_data, count, start, capacity, friction
):
    inject_contact_data(count, start, capacity, friction=friction)

    contact_sensor.update(1 / 60, force_recompute=True)
    actual = (
        contact_sensor.data.friction_forces_w
        if friction
        else contact_sensor.data.contact_pos_w
    )
    expected = torch.tensor([1.0, 2.0, 3.0], device=contact_sensor.device)
    if friction:
        expected *= count
    elif count == 0:
        expected.fill_(float("nan"))
    torch.testing.assert_close(actual[0, 0, 0], expected, equal_nan=True)


@pytest.mark.parametrize(
    "count,start,capacity,friction",
    [
        (20, 45, 64, False),
        (22, 47, 64, False),
        (2, 1023, 1024, False),
        (2, 2147483647, 64, False),
        (20, 45, 64, True),
    ],
)
def test_contact_sensor_overflow_reports_capacity_without_corrupting_cuda(
    contact_sensor, inject_contact_data, count, start, capacity, friction
):
    inject_contact_data(count, start, capacity, friction=friction)

    with pytest.raises(
        RuntimeError,
        match=(
            f"Contact buffer overflow.*capacity={capacity}, "
            f"required={start + count}"
        ),
    ):
        contact_sensor.update(1 / 60, force_recompute=True)

    assert torch.arange(4, device=contact_sensor.device).sum().item() == 6


@pytest.mark.parametrize("count,start", [(-1, 0), (1, -1)])
def test_contact_sensor_negative_metadata_raises_clear_error(
    contact_sensor, inject_contact_data, count, start
):
    inject_contact_data(count, start, 64)

    with pytest.raises(RuntimeError, match="Negative contact"):
        contact_sensor.update(1 / 60, force_recompute=True)
