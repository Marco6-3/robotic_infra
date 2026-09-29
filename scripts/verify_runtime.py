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


def verify(require_lerobot=False, require_vendor=False, require_cuda=False):
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
        # Offline installation from the pinned local checkout retains a file URL,
        # rather than VCS metadata. Verify both its HEAD and clean tracked files.
        if commit is None and direct.get("url", "").startswith("file:"):
            from urllib.parse import unquote, urlparse
            import subprocess
            source = Path(unquote(urlparse(direct["url"]).path))
            if (source / ".git").exists():
                commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
                if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
                    raise RuntimeError("LeRobot local source has tracked modifications")
        if commit != manifest["lerobot"]["commit"]:
            raise RuntimeError("LeRobot must be installed from the manifest's exact Git commit")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401
        print(f"LeRobot: {distribution.version}, commit {commit}")
    if require_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        matrix = torch.randn(128, 128, device="cuda", requires_grad=True)
        loss = (matrix @ matrix.T).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or not torch.isfinite(matrix.grad).all():
            raise RuntimeError("CUDA forward/backward produced non-finite values")
        print(f"CUDA forward/backward: passed; torch={torch.__version__}, CUDA={torch.version.cuda}, "
              f"GPU={torch.cuda.get_device_name()}, capability={torch.cuda.get_device_capability()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-lerobot", action="store_true")
    parser.add_argument("--require-vendor", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    try:
        verify(args.require_lerobot, args.require_vendor, args.require_cuda)
    except (ImportError, RuntimeError, OSError, metadata.PackageNotFoundError) as exc:
        raise SystemExit(f"Runtime verification failed: {exc}") from exc
