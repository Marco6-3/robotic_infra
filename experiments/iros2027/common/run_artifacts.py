"""Inspectable and resumable run-directory lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


class RunArtifacts:
    """Create or resume a run while preserving config and attempt history."""

    def __init__(self, root: Path, run_path: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.run_path = run_path
        self.manifest = manifest
        self.attempt_path: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        output_root: Path,
        experiment_id: str,
        config_path: Path,
        resume: Path | None = None,
    ) -> "RunArtifacts":
        config_path = config_path.resolve()
        config_sha256 = _sha256(config_path)
        if resume is not None:
            run_path = resume.resolve()
            manifest_path = run_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["config_sha256"] != config_sha256:
                raise ValueError("resume config hash does not match the original run")
            instance = cls(root, run_path, manifest)
            instance._start_attempt(resumed=True)
            return instance

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_path = output_root.resolve() / f"{timestamp}-{experiment_id}"
        run_path.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_path / "config.yaml")
        dirty = bool(_git_value(root, "status", "--porcelain"))
        manifest = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "created_at": _now(),
            "config_source": str(config_path),
            "config_sha256": config_sha256,
            "git_commit": _git_value(root, "rev-parse", "HEAD"),
            "git_dirty": dirty,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "attempts": [],
            "checkpoints": [],
        }
        instance = cls(root, run_path, manifest)
        instance._write_manifest()
        instance._start_attempt(resumed=False)
        return instance

    def _write_manifest(self) -> None:
        _atomic_json(self.run_path / "manifest.json", self.manifest)

    def _start_attempt(self, *, resumed: bool) -> None:
        index = len(self.manifest["attempts"]) + 1
        self.attempt_path = self.run_path / "attempts" / f"{index:04d}"
        self.attempt_path.mkdir(parents=True, exist_ok=False)
        self.manifest["attempts"].append(
            {
                "index": index,
                "started_at": _now(),
                "resumed": resumed,
                "status": "running",
            }
        )
        self._write_manifest()

    def append_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        if self.attempt_path is None:
            raise RuntimeError("attempt has not started")
        with (self.attempt_path / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def register_checkpoint(
        self, checkpoint_path: Path, *, step: int, metric: dict[str, float]
    ) -> None:
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        entry = {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "step": int(step),
            "metric": metric,
            "registered_at": _now(),
        }
        self.manifest["checkpoints"].append(entry)
        self._write_manifest()

    def finalize(self, summary: dict[str, Any], *, success: bool) -> None:
        if self.attempt_path is None:
            raise RuntimeError("attempt has not started")
        _atomic_json(self.attempt_path / "summary.json", summary)
        attempt = self.manifest["attempts"][-1]
        attempt["finished_at"] = _now()
        attempt["status"] = "completed" if success else "failed"
        attempt["summary"] = str(
            (self.attempt_path / "summary.json").relative_to(self.run_path)
        )
        self._write_manifest()
