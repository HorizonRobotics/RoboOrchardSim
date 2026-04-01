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

import functools

import pytest


# borrow from: https://stackoverflow.com/questions/63517507/pytest-skip-for-a-specific-exception  # noqa
def skip_on(exception, reason: str = ""):
    # Func below is the real decorator and will receive the test function as param  # noqa
    def decorator_func(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                # Try to run the test
                return f(*args, **kwargs)
            except exception:
                # If exception of given type happens
                # just swallow it and raise pytest.Skip with given reason
                pytest.skip(reason)

        return wrapper

    return decorator_func
