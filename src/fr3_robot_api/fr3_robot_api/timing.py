"""Deterministic policy-to-low-level target scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class ScheduledTarget:
    timestamp_ns: int
    target: np.ndarray


class TargetInterpolator:
    """Evaluate 30 Hz targets from a 1 kHz simulation clock.

    The first target is held until a second policy target arrives. Afterwards,
    causal ``linear`` mode ramps from the current low-level value to the new
    target over one inferred policy period; ``hold`` returns the newest target
    immediately throughout the interval.
    """

    def __init__(self, size: int, mode: str = "linear") -> None:
        if mode not in {"hold", "linear"}:
            raise ValueError("mode must be 'hold' or 'linear'")
        self._size = size
        self._mode = mode
        self._previous: Optional[ScheduledTarget] = None
        self._current: Optional[ScheduledTarget] = None
        self._ramp_start_ns: Optional[int] = None
        self._ramp_end_ns: Optional[int] = None
        self._ramp_start_value: Optional[np.ndarray] = None
        self._policy_period_ns: Optional[int] = None

    def set_target(self, timestamp_ns: int, target: np.ndarray) -> None:
        value = np.asarray(target, dtype=np.float64)
        if value.shape != (self._size,):
            raise ValueError(f"target must have shape ({self._size},), got {value.shape}")
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self._current is not None and timestamp_ns < self._current.timestamp_ns:
            raise ValueError("policy timestamps must be monotonic")
        if self._current is None:
            self._current = ScheduledTarget(timestamp_ns, value.copy())
            self._ramp_start_ns = timestamp_ns
            self._ramp_start_value = value.copy()
            return

        period = timestamp_ns - self._current.timestamp_ns
        if period > 0:
            self._policy_period_ns = period
        if self._policy_period_ns is None:
            raise ValueError("repeated policy timestamps cannot establish an interpolation period")
        self._previous = self._current
        # A future policy target is not known until its 30 Hz message arrives.
        # Start the ramp at that arrival and finish it one policy period later;
        # this gives deterministic, causal interpolation instead of pretending
        # the future target was available in the preceding interval.
        start_value = self.evaluate(timestamp_ns)
        self._current = ScheduledTarget(timestamp_ns, value.copy())
        self._ramp_start_ns = timestamp_ns
        self._ramp_end_ns = timestamp_ns + self._policy_period_ns
        self._ramp_start_value = start_value

    def evaluate(self, timestamp_ns: int) -> np.ndarray:
        if self._current is None:
            raise RuntimeError("no policy target has been set")
        if self._mode == "hold" or self._ramp_end_ns is None or self._ramp_start_value is None:
            return self._current.target.copy()
        if timestamp_ns <= self._ramp_start_ns:
            return self._ramp_start_value.copy()
        if timestamp_ns >= self._ramp_end_ns:
            return self._current.target.copy()
        fraction = (timestamp_ns - self._ramp_start_ns) / (self._ramp_end_ns - self._ramp_start_ns)
        return self._ramp_start_value + fraction * (self._current.target - self._ramp_start_value)
