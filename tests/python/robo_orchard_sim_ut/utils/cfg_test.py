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

import cloudpickle as pickle
from robo_orchard_core.utils.config import Config, load_config_class


class CfgTestBase:
    """Base class for Config tests.

    User should inherit this class and implement the `simple_cfg` fixture.
    """

    def test_to_json(self, simple_cfg: Config):
        # print(simple_cfg.to_dict())
        json_str = simple_cfg.to_str(format="json")
        assert isinstance(json_str, str)

    def test_from_dict(self, simple_cfg: Config):
        config = simple_cfg
        json_str = config.to_dict(include_config_type=True)
        # print("old_cfg: ", config)
        new_config = type(simple_cfg).from_dict(json_str)
        # print("new_cfg: ", new_config)
        assert new_config.content_equal(config)
        assert new_config == config
        new_config2 = load_config_class(json_str, format="json")
        assert new_config2.content_equal(config)

    def test_from_json(self, simple_cfg: Config):
        config = simple_cfg
        json_str = config.to_str(format="json")
        # print("old_cfg: ", json_str)
        new_config = type(simple_cfg).from_str(json_str, format="json")
        # print("new_cfg: ", new_config.to_str(format="json"))
        assert new_config.content_equal(config)
        assert new_config == config

        new_config2 = load_config_class(json_str, format="json")
        assert new_config2.content_equal(config)

    def test_serialization_json(self, simple_cfg: Config):
        config = simple_cfg
        # print(config)
        # print(config.to_dict())
        json_str = config.to_str(format="json")
        new_config = type(simple_cfg).from_str(json_str, format="json")
        assert new_config.content_equal(config)

    def test_serialization_pickable(self, simple_cfg: Config):
        config = simple_cfg
        new_config = pickle.loads(pickle.dumps(config))
        assert config.content_equal(new_config)

    def test_dump_json_ignore_default(self, simple_cfg: Config):
        config = simple_cfg
        json_str = config.to_str(exclude_defaults=True, format="json")
        # print(json_str)
        new_config = type(simple_cfg).from_str(json_str, format="json")
        assert new_config.content_equal(config)

    def test_from_toml(self, simple_cfg: Config):
        config = simple_cfg
        json_str = config.to_str(format="toml")
        # print("toml old_cfg: ", json_str)
        new_config = type(simple_cfg).from_str(json_str, format="toml")
        # print("toml new_cfg: ", new_config.to_str(format="toml"))
        assert new_config.content_equal(config)
        assert new_config == config

        new_config2 = load_config_class(json_str, format="toml")
        assert new_config2.content_equal(config)

    def test_from_yaml(self, simple_cfg: Config):
        config = simple_cfg
        json_str = config.to_str(format="yaml")
        # print("yaml old_cfg: ", json_str)
        new_config = type(simple_cfg).from_str(json_str, format="yaml")
        # print("yaml new_cfg: ", new_config.to_str(format="yaml"))
        assert new_config.content_equal(config)
        assert new_config == config

        new_config2 = load_config_class(json_str, format="yaml")
        assert new_config2.content_equal(config)
