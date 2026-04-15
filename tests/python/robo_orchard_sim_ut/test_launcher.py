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
import subprocess
import sys
import types
from typing import Any

import pytest


class _StubLauncherContext:
    def __init__(self) -> None:
        self.enter_calls = 0
        self.exit_calls = 0
        self.app = object()

    def __enter__(self) -> object:
        self.enter_calls += 1
        return self.app

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_calls += 1


class TestLauncher:
    def get_file_path_for_launcher(self):
        # find examples/launcher_example.py and return the path

        return os.path.join(
            os.path.dirname(__file__),
            "../../../examples/isaac/launcher_example.py",
        )

    def test_launcher_with_python_module(self):
        file = self.get_file_path_for_launcher()
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robo_orchard_sim.launcher",
                file,
                "--headless",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        while True:
            output = worker.stdout.readline()
            if output == b"" and worker.poll() is not None:
                assert worker.returncode == 0
                break
            if output:
                print(output.strip().decode("utf-8"))

    def test_launcher_with_script(self):
        file = self.get_file_path_for_launcher()
        worker = subprocess.Popen(
            [
                "RoboOrchard-SimLauncher",
                file,
                "--headless",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        while True:
            output = worker.stdout.readline()
            if output == b"" and worker.poll() is not None:
                assert worker.returncode == 0
                break
            if output:
                print(output.strip().decode("utf-8"))

    @pytest.mark.skip(reason="Removed")
    def test_launcher_with_script_deprecated(self):
        file = self.get_file_path_for_launcher()
        worker = subprocess.Popen(
            [
                "RoboOrchard-Launcher",
                file,
                "--headless",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        while True:
            output = worker.stdout.readline()
            if output == b"" and worker.poll() is not None:
                assert worker.returncode == 0
                break
            if output:
                print(output.strip().decode("utf-8"))

    def test_simple_launcher_close_is_idempotent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import atexit

        import robo_orchard_sim.launcher as launcher_module

        registered_callbacks: list[Any] = []
        stub_context = _StubLauncherContext()

        monkeypatch.setattr(
            launcher_module,
            "isaac_lab_launcher_context",
            lambda launcher_args=None, **kwargs: stub_context,
        )
        monkeypatch.setattr(
            atexit,
            "register",
            lambda callback: registered_callbacks.append(callback),
        )
        if hasattr(launcher_module.SimpleIsaacAppLauncher, "_instance"):
            delattr(launcher_module.SimpleIsaacAppLauncher, "_instance")

        launcher = launcher_module.SimpleIsaacAppLauncher(headless=True)

        assert stub_context.enter_calls == 1
        assert len(registered_callbacks) == 1

        launcher.close()
        registered_callbacks[0]()
        launcher.__del__()

        assert stub_context.exit_calls == 1
        assert not hasattr(launcher_module.SimpleIsaacAppLauncher, "_instance")

    def test_configure_torch_cuda_arch_list_uses_env_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import robo_orchard_sim.launcher as launcher_module

        monkeypatch.setenv("ROBO_ORCHARD_TORCH_CUDA_ARCH_LIST", "12.0+PTX")
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0;9.0a;9.0+PTX")

        launcher_module._configure_torch_cuda_arch_list()

        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "12.0+PTX"

    def test_configure_torch_cuda_arch_list_auto_detects_gpu_capability(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import robo_orchard_sim.launcher as launcher_module

        monkeypatch.delenv("ROBO_ORCHARD_TORCH_CUDA_ARCH_LIST", raising=False)
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0;9.0a;9.0+PTX")

        class _Result:
            stdout = "12.0\n12.0\n"

        monkeypatch.setattr(
            launcher_module.subprocess,
            "run",
            lambda *args, **kwargs: _Result(),
        )

        launcher_module._configure_torch_cuda_arch_list()

        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "12.0+PTX"

    def test_disable_torch_jit_gpu_fusion_configures_runtime_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On Blackwell (compute >= 12), all four JIT flags are disabled."""
        import robo_orchard_sim.launcher as launcher_module

        _Result = types.SimpleNamespace(stdout="12.0\n")
        monkeypatch.setattr(
            launcher_module.subprocess,
            "run",
            lambda *args, **kwargs: _Result,
        )

        calls: list[tuple[str, bool]] = []

        for name in (
            "_jit_override_can_fuse_on_gpu",
            "_jit_set_texpr_fuser_enabled",
            "_jit_set_profiling_executor",
            "_jit_set_profiling_mode",
        ):
            monkeypatch.setattr(
                launcher_module.torch._C,
                name,
                lambda value, _name=name: calls.append((_name, value)),
            )

        launcher_module._disable_torch_jit_gpu_fusion()

        assert calls == [
            ("_jit_override_can_fuse_on_gpu", False),
            ("_jit_set_texpr_fuser_enabled", False),
            ("_jit_set_profiling_executor", False),
            ("_jit_set_profiling_mode", False),
        ]

    def test_disable_torch_jit_gpu_fusion_skipped_on_pre_blackwell(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On pre-Blackwell GPUs (compute < 12), JIT flags are not touched."""
        import robo_orchard_sim.launcher as launcher_module

        _Result = types.SimpleNamespace(stdout="8.9\n")
        monkeypatch.setattr(
            launcher_module.subprocess,
            "run",
            lambda *args, **kwargs: _Result,
        )

        calls: list[tuple[str, bool]] = []

        for name in (
            "_jit_override_can_fuse_on_gpu",
            "_jit_set_texpr_fuser_enabled",
            "_jit_set_profiling_executor",
            "_jit_set_profiling_mode",
        ):
            monkeypatch.setattr(
                launcher_module.torch._C,
                name,
                lambda value, _name=name: calls.append((_name, value)),
            )

        launcher_module._disable_torch_jit_gpu_fusion()

        assert calls == [], (
            "JIT flags must not be modified on pre-Blackwell GPUs"
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
