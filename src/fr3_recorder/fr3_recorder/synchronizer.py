"""Timestamp matching for the 30 Hz dataset timeline."""

from __future__ import annotations

from typing import Iterable, Optional, TypeVar

T = TypeVar("T")


def nearest_by_timestamp(
    samples: Iterable[T],
    timestamp_ns: int,
    *,
    timestamp=lambda sample: sample.timestamp_ns,
    max_delta_ns: Optional[int] = None,
) -> Optional[T]:
    """Return the sample closest to ``timestamp_ns``.

    The iterable may contain 1 kHz state samples or 30 Hz camera frames. A
    tolerance can be supplied to make missing/stale modalities explicit.
    """

    best = None
    best_delta = None
    for sample in samples:
        delta = abs(int(timestamp(sample)) - int(timestamp_ns))
        if best_delta is None or delta < best_delta:
            best, best_delta = sample, delta
    if best_delta is None or (max_delta_ns is not None and best_delta > max_delta_ns):
        return None
    return best
