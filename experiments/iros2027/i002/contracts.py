"""Wire-level contracts shared by learned and scripted I002 policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticMessage:
    sequence: int
    source_timestamp_ns: int
    coarse_action: tuple[float, ...]
    tokens: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.source_timestamp_ns < 0:
            raise ValueError("sequence and source_timestamp_ns must be non-negative")
        if not self.coarse_action:
            raise ValueError("coarse_action must not be empty")

    @property
    def wire_bytes(self) -> int:
        # Two uint64 fields plus float32 payloads. Protocol framing is kept
        # separate so comparisons use the same payload accounting.
        return 16 + 4 * (len(self.coarse_action) + len(self.tokens))


@dataclass(frozen=True)
class TactileFeedbackMessage:
    sequence: int
    source_timestamp_ns: int
    contact_event: bool
    summary: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.source_timestamp_ns < 0:
            raise ValueError("sequence and source_timestamp_ns must be non-negative")

    @property
    def wire_bytes(self) -> int:
        return 17 + 4 * len(self.summary)
