#!/usr/bin/env python3
"""Install optional ML/video dependencies without altering the ROS Pixi solve."""
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    if sys.version_info < (3, 12):
        raise SystemExit("Pinned LeRobot requires Python >=3.12; run through pixi")
    destination = ROOT / ".venv-recording"
    if not (destination / "pyvenv.cfg").exists():
        venv.EnvBuilder(with_pip=True).create(destination)
    subprocess.run([str(destination / "bin/python"), "-m", "pip", "install", "-r",
                    str(ROOT / "requirements-runtime.txt")], check=True)
    subprocess.run([str(destination / "bin/python"), str(ROOT / "scripts/verify_runtime.py"),
                    "--require-lerobot"], check=True)
