#!/usr/bin/env python3
"""Check actual MuJoCo binaries and optional LeRobot VCS provenance."""
import argparse
import ctypes
from importlib import metadata
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def verify(require_lerobot=False, require_vendor=False):
    manifest = yaml.safe_load((ROOT / "config/runtime_manifest.yaml").read_text())
    expected = str(manifest["mujoco"]["version"])
    import mujoco
    actual = mujoco.mj_versionString()
    print(f"MuJoCo Python/native: {mujoco.__version__} / {actual}")
    if mujoco.__version__ != expected or actual != expected:
        raise RuntimeError(f"expected MuJoCo {expected}")
    prefix = os.environ.get("CONDA_PREFIX")
    libraries = [] if not prefix else list((Path(prefix) / "lib").glob("libmujoco.so*"))
    if libraries:
        library = ctypes.CDLL(str(sorted(libraries)[0]))
        library.mj_versionString.restype = ctypes.c_char_p
        native = library.mj_versionString().decode()
        print(f"MuJoCo ROS vendor library: {native}")
        if native != expected:
            raise RuntimeError(f"ROS vendor library is {native}, expected {expected}")
    elif require_vendor or prefix:
        raise RuntimeError("ROS libmujoco not found in CONDA_PREFIX/lib")
    else:
        print("ROS vendor library: not checked outside the Pixi environment")
    if require_lerobot:
        if sys.version_info < (3, 12):
            raise RuntimeError("LeRobot requires Python 3.12+")
        distribution = metadata.distribution("lerobot")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id")
        if commit != manifest["lerobot"]["commit"]:
            raise RuntimeError("LeRobot must be installed from the manifest's exact Git commit")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
        print(f"LeRobot: {distribution.version}, commit {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-lerobot", action="store_true")
    parser.add_argument("--require-vendor", action="store_true")
    args = parser.parse_args()
    try:
        verify(args.require_lerobot, args.require_vendor)
    except (ImportError, RuntimeError, OSError, metadata.PackageNotFoundError) as exc:
        raise SystemExit(f"Runtime verification failed: {exc}") from exc
