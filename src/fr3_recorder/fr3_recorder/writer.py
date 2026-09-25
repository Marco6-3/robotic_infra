"""Lazy LeRobotDataset adapter.

LeRobot's public constructor has changed slightly across releases. Keeping the
import and constructor here isolates that compatibility work from the Robot
interface and makes the required dependency explicit at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .schema import build_lerobot_features


class LeRobotUnavailable(RuntimeError):
    """Raised when recording is requested without the optional LeRobot install."""


class LeRobotDatasetWriter:
    def __init__(
        self,
        repo_id: str,
        root: Path,
        *,
        fps: int = 30,
        robot_type: str = "fr3",
        task: str = "fr3_simulation",
        use_videos: bool = True,
        features: Optional[Mapping[str, Any]] = None,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - exercised on runtime host
            raise LeRobotUnavailable(
                "LeRobot is required to record datasets; install the pinned runtime dependency"
            ) from exc
        self.task = task
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=str(root),
            robot_type=robot_type,
            features=dict(features or build_lerobot_features()),
            use_videos=use_videos,
        )

    def add_frame(self, frame: Mapping[str, Any]) -> None:
        payload = dict(frame)
        # LeRobot treats task as a required special field rather than a user
        # feature. Keep it out of the platform schema while supplying the
        # stable task label expected by DatasetWriter.add_frame().
        payload.setdefault("task", self.task)
        self.dataset.add_frame(payload)

    def save_episode(self) -> None:
        self.dataset.save_episode()

    def finalize(self) -> None:
        finalize = getattr(self.dataset, "finalize", None)
        if finalize is not None:
            finalize()
