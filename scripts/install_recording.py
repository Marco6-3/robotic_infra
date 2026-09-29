#!/usr/bin/env python3
"""Install optional ML/video dependencies without altering the ROS Pixi solve."""
from pathlib import Path
import argparse
import os
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", help="install the tested x86_64 CUDA 12.8 stack")
    parser.add_argument("--index-url", help="optional PyPI mirror for this installation only")
    parser.add_argument("--torch-index-url", default="https://download.pytorch.org/whl/cu128")
    parser.add_argument("--lerobot-source", type=Path, help="optional clean checkout at the manifest commit")
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        raise SystemExit("Pinned LeRobot requires Python >=3.12; run through pixi")
    destination = ROOT / ".venv-recording"
    if not (destination / "pyvenv.cfg").exists():
        venv.EnvBuilder(with_pip=True).create(destination)
    constraints = []
    # RoboStack activation prepends its site-packages to PYTHONPATH. A venv
    # alone does not isolate those imports (or pip's installed-package scan).
    child_env = os.environ.copy()
    child_env.pop("PYTHONPATH", None)
    child_env.pop("PYTHONHOME", None)
    child_env["PYTHONNOUSERSITE"] = "1"
    index = ["--index-url", args.index_url] if args.index_url else []
    requirements = ["-r", str(ROOT / "requirements-runtime.txt")]
    if args.lerobot_source:
        import yaml
        source = args.lerobot_source.resolve()
        expected = yaml.safe_load((ROOT / "config/runtime_manifest.yaml").read_text())["lerobot"]["commit"]
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        if actual != expected or dirty:
            raise SystemExit("LeRobot source must be clean and match runtime_manifest.yaml")
        requirements = ["mujoco==3.10.0", "pyyaml>=6,<7", f"{source}[dataset]"]
    if args.cuda:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise SystemExit("--cuda is validated for Linux x86_64 only")
        subprocess.run([str(destination / "bin/python"), "-m", "pip", "install",
                        "torch==2.8.0", "torchvision==0.23.0", "--index-url",
                        args.torch_index_url], check=True, env=child_env)
        constraints = ["-c", str(ROOT / "config/constraints-cu128.txt")]
    subprocess.run([str(destination / "bin/python"), "-m", "pip", "install",
                    *requirements, *constraints, *index], check=True, env=child_env)
    subprocess.run([str(destination / "bin/python"), "-m", "pip", "check"], check=True, env=child_env)
    subprocess.run([str(destination / "bin/python"), str(ROOT / "scripts/verify_runtime.py"),
                    "--require-lerobot", *(["--require-cuda"] if args.cuda else [])], check=True, env=child_env)
