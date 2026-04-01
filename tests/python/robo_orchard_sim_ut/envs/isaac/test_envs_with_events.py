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

import pytest
from robo_orchard_core.envs.managers.events.event_manager import (
    EventManagerCfg,
)

from robo_orchard_sim.envs import (
    IsaacEnvContextManager,
    IsaacManagerBasedEnv,
    IsaacManagerBasedEnvCfg,
)
from robo_orchard_sim.models.scenes.table_scene import TableSceneCfg
from robo_orchard_sim_ut.envs.isaac.event_helper import (
    ResetEventTermBase,
    ResetEventTermCfg,
)


class TestResetEventTerm:
    def test_reset_evnet_init(
        self,
        simple_table_scene_cfg_with_camera: TableSceneCfg,
    ):
        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=simple_table_scene_cfg_with_camera,
            events=EventManagerCfg(
                terms={
                    "reset_print": ResetEventTermCfg(
                        class_type=ResetEventTermBase,
                        trigger_topic=IsaacManagerBasedEnv.RESET[0],
                    )
                },
            ),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            _ = env.step()
            env.reset()
            _ = env.step()

    @pytest.mark.parametrize("expect_id", [[1, 0, 2], [0, 1, 2], [2, 10, 4]])
    def test_event_order(
        self,
        simple_table_scene_cfg_with_camera: TableSceneCfg,
        capfd,
        expect_id,
    ):
        terms = {
            f"term{chr(65 + i)}": ResetEventTermCfg(
                class_type=ResetEventTermBase,
                trigger_topic=IsaacManagerBasedEnv.RESET[0],
                id=id,
            )
            for i, id in enumerate(expect_id)
        }

        env_cfg = IsaacManagerBasedEnvCfg(
            decimation=1,
            scene=simple_table_scene_cfg_with_camera,
            events=EventManagerCfg(terms=terms),
        )
        with IsaacEnvContextManager(
            env_cfg, with_new_stage=True, disable_exit_on_stop=True
        ) as env:
            assert isinstance(env, IsaacManagerBasedEnv)
            _ = env.step()
            env.reset()
            _ = env.step()

        # Capture the output
        captured = capfd.readouterr()

        # Check the order of ids in the output
        expected_order = [f"ResetEventTermBase[{id}]" for id in expect_id]
        output = captured.out

        # Verify the order of ids
        previous_index = -1
        for term_id in expected_order:
            current_index = output.find(term_id)
            assert current_index != -1, f"{term_id} not found in output"
            assert current_index > previous_index, f"{term_id} is out of order"
            previous_index = current_index
