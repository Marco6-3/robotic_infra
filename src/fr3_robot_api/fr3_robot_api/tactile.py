"""Vision-based tactile acquisition contracts for simulation and hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .types import VisionTactileFrame


@dataclass(frozen=True)
class VisionTactileConfig:
    name: str
    model: str = "digit_reference"
    placement: str = "parallel_jaw_inner_fingertip"
    width: int = 640
    height: int = 480
    fps: int = 60
    color_order: str = "rgb"
    marker_gel: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.model or not self.placement:
            raise ValueError("name, model, and placement must not be empty")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("tactile image dimensions and fps must be positive")
        if self.color_order != "rgb":
            raise ValueError("the policy-facing tactile contract is RGB")


class VisionTactileSource(ABC):
    config: VisionTactileConfig

    @abstractmethod
    def read(self) -> VisionTactileFrame:
        """Return the next frame with source and optional arrival timestamps."""

    def close(self) -> None:
        """Release the USB/camera backend when necessary."""


class ArrayVisionTactileSource(VisionTactileSource):
    """Test/adapter source; hardware backends can implement the same contract."""

    def __init__(self, config: VisionTactileConfig, source: Any) -> None:
        self.config = config
        self._source = source

    def read(self) -> VisionTactileFrame:
        timestamp_ns, sequence, rgb = self._source()
        return VisionTactileFrame(timestamp_ns, rgb, self.config.name, sequence)
