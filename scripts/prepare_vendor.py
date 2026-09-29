#!/usr/bin/env python3
"""Apply the tracked portability patch to the pinned upstream checkout."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_pins.py")], check=True)
    repo = ROOT / "src/mujoco_ros2_control"
    patch = ROOT / "patches/mujoco-ros2-control-librt.patch"
    command = ["git", "-C", str(repo), "apply"]
    if subprocess.run([*command, "--reverse", "--check", str(patch)], capture_output=True).returncode == 0:
        print("Linux librt portability patch already applied")
    else:
        subprocess.run([*command, "--check", str(patch)], check=True)
        subprocess.run([*command, str(patch)], check=True)
        print("Applied Linux librt portability patch")
