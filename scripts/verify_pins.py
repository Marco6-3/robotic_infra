#!/usr/bin/env python3
"""Verify that the checked-out model repositories match the v1 manifest."""

from __future__ import annotations

import subprocess
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    ROOT / path: spec["version"]
    for manifest in ("models.repos", "third_party.repos")
    for path, spec in yaml.safe_load((ROOT / "repos" / manifest).read_text())["repositories"].items()
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
