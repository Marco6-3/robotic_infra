"""Camera abstraction preserving source timestamps for later sim-to-real effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .types import CameraFrame


@dataclass(frozen=True)
class CameraConfig:
    name: str
    model: str = "intel_realsense_d435_rgb"
    width: int = 640
    height: int = 480
    fps: int = 30
    fov_y_deg: float = 42.5
    depth: bool = False
    noise: bool = False
    latency_s: float = 0.0
    jitter_s: float = 0.0
    dropped_frames: bool = False

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fps <= 0 or self.fov_y_deg <= 0:
            raise ValueError("camera dimensions and fps must be positive")
        if self.depth:
            raise ValueError("depth is disabled in FR3 platform v1")
        if self.noise or self.latency_s or self.jitter_s or self.dropped_frames:
            raise ValueError("camera realism effects are disabled in FR3 platform v1")


class Camera(ABC):
    config: CameraConfig

    @abstractmethod
    def read(self) -> CameraFrame:
        """Return the next RGB frame with its source timestamp."""


class ArrayCamera(Camera):
    """Small test/runtime adapter for a caller-provided RGB frame source."""

    def __init__(self, config: CameraConfig, source: Any) -> None:
        self.config = config
        self._source = source

    def read(self) -> CameraFrame:
        timestamp_ns, rgb = self._source()
        return CameraFrame(timestamp_ns, rgb, self.config.name)
