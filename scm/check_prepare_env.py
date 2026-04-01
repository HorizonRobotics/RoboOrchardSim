import argparse
import functools
import re
import subprocess
import sys

from packaging.version import Version

BUILD_DEPS_TORCH = {
    "torch": ("2.5.1", "2.5.1+cu118"),
    "torchvision": ("0.20.1", "0.20.1+cu118"),
}

BUILD_DEPS_ISAAC = {
    "isaacsim": ("4.5.0.0", None),
    "isaacsim-extscache-physics": ("4.5.0.0", None),
    "isaacsim-extscache-kit": ("4.5.0.0", None),
    "isaacsim-extscache-kit-sdk": ("4.5.0.0", None),
    "isaacsim-replicator": ("4.5.0.0", None),
    "isaacsim-app": ("4.5.0.0", None),
    "isaaclab": ("2.0.2", None),
}


@functools.lru_cache(maxsize=None)
def get_package_version(package_name: str) -> str | None:
    try:
        reqs = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", package_name]
        )
    except subprocess.CalledProcessError:
        return None

    reg_find_res = re.search(r"Version: (.+)", reqs.decode("utf-8"))
    if reg_find_res is None:
        raise ValueError(
            f"Cannot get the version of package {package_name}. "
            f"pip show result: {reqs!r}"
        )

    return reg_find_res.group(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", action="store_true", default=None)
    args = parser.parse_args()
    if args.user is None:
        args.user = True
    return args


def ensure_package(
    package_name: str,
    min_version: str,
    to_install: str | None,
    user: bool,
) -> None:
    install_version = to_install or min_version
    installed_version = get_package_version(package_name)
    print("searching for", package_name, "found", installed_version)

    if installed_version is not None and Version(installed_version) >= Version(
        min_version
    ):
        return

    print(
        f"Package {package_name} version {installed_version} is not "
        f"compatible with version {min_version}. "
        "Now install on the fly..."
    )

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"{package_name}=={install_version}",
        "-i",
        "https://art-internal.hobot.cc/artifactory/api/pypi/pypi/simple",
        "--timeout",
        "600",
        "--retries",
        "3",
    ]
    if user:
        cmd.append("--user")
    subprocess.check_call(cmd)


if __name__ == "__main__":
    args = parse_args()
    build_deps = {}
    build_deps.update(BUILD_DEPS_TORCH)
    build_deps.update(BUILD_DEPS_ISAAC)

    for pkg, (version, to_install) in build_deps.items():
        ensure_package(pkg, version, to_install, args.user)
