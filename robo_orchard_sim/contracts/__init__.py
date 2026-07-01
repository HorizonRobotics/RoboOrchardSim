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

"""Inter-package contract types shared across robo_orchard_sim modules.

This package holds the data shapes that multiple top-level packages must
agree on, but that don't naturally belong to any single one. Putting them
here keeps `orchard_env/` and `policy/` from importing each other and
turns their interaction into a clean producer/consumer relationship over
a shared protocol.
"""
