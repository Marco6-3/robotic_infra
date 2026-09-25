#!/usr/bin/env python3
"""Verify that the checked-out model repositories match the v1 manifest."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    ROOT / "third_party/mujoco_menagerie": "32224735cf004df068cd29ba099459f71bcd7d21",
    ROOT / "third_party/franka_description": "7aeeddc449edf8d62b594f9e36a81da53e7796f9",
    ROOT / "src/mujoco_ros2_control": "0b6b57b9afd2ea6f95e37738ca8f63b8be7f15f5",
}


def main() -> int:
    failed = False
    for path, expected in EXPECTED.items():
        if not (path / ".git").exists():
            print(f"MISSING {path} (run vcs import first)")
            failed = True
            continue
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != expected:
            print(f"MISMATCH {path}: expected {expected}, got {actual}")
            failed = True
        else:
            print(f"OK      {path}: {actual}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
