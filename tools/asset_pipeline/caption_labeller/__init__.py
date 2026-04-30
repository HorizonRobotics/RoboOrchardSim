# Project RoboOrchard
#
# Copyright (c) 2026 Horizon Robotics. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Caption candidates labeller skill.

Post-processes already-labelled URDFs to generate a pool of diverse
natural-language caption phrases per asset, written as
``caption_candidates.json`` next to the URDF and linked via
``<caption_candidates>`` in the URDF's ``<extra_info>`` block.
"""
