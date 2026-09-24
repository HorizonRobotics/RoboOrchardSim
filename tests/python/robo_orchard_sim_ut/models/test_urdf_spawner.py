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

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def test_urdf_file_cfg_default_func_uses_process_locked_spawner(
    tmp_path: Path,
) -> None:
    from robo_orchard_sim.ext.cfg_wrappers.sim.spawners.from_files import (
        UrdfFileCfg,
        spawn_from_urdf_locked,
    )

    cfg = UrdfFileCfg(
        asset_path=str(tmp_path / "robot.urdf"),
        usd_dir=str(tmp_path / "usd"),
        fix_base=True,
    )

    assert cfg.func is spawn_from_urdf_locked


def test_convert_urdf_to_usd_locked_concurrent_requests_serialize_conversion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import from_files

    asset_path = tmp_path / "robot.urdf"
    asset_path.write_text("<robot name='test'/>", encoding="utf-8")
    usd_dir = tmp_path / "usd"
    cfg = from_files.UrdfFileCfg(
        asset_path=str(asset_path),
        usd_dir=str(usd_dir),
        fix_base=True,
    )
    state_lock = threading.Lock()
    conversion_active = False

    class FakeUrdfConverter:
        def __init__(self, converter_cfg: object) -> None:
            del converter_cfg
            nonlocal conversion_active
            with state_lock:
                if conversion_active:
                    raise RuntimeError("concurrent conversion detected")
                conversion_active = True
            try:
                time.sleep(0.05)
                usd_dir.mkdir(parents=True, exist_ok=True)
                self.usd_path = str(usd_dir / "robot.usd")
            finally:
                with state_lock:
                    conversion_active = False

    monkeypatch.setattr(
        from_files.converters,
        "UrdfConverter",
        FakeUrdfConverter,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                from_files.convert_urdf_to_usd_locked,
                (cfg, cfg),
            )
        )

    assert results == [str(usd_dir / "robot.usd")] * 2


def test_convert_urdf_to_usd_locked_stale_marker_rebuilds_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robo_orchard_sim.ext.cfg_wrappers.sim.spawners import from_files

    asset_path = tmp_path / "robot.urdf"
    asset_path.write_text("<robot name='test'/>", encoding="utf-8")
    usd_dir = tmp_path / "usd"
    usd_dir.mkdir()
    (usd_dir / ".conversion_in_progress").touch()
    stale_file = usd_dir / "partial.usd"
    stale_file.touch()
    cfg = from_files.UrdfFileCfg(
        asset_path=str(asset_path),
        usd_dir=str(usd_dir),
        fix_base=True,
    )

    class FakeUrdfConverter:
        def __init__(self, converter_cfg: object) -> None:
            del converter_cfg
            if stale_file.exists():
                raise RuntimeError("stale cache was not removed")
            self.usd_path = str(usd_dir / "robot.usd")

    monkeypatch.setattr(
        from_files.converters,
        "UrdfConverter",
        FakeUrdfConverter,
    )

    usd_path = from_files.convert_urdf_to_usd_locked(cfg)

    assert usd_path == str(usd_dir / "robot.usd")
