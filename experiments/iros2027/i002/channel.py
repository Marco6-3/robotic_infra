"""Deterministic delayed latest-value channel with causal delivery semantics."""

from __future__ import annotations

import heapq
import random
from dataclasses import asdict, dataclass
from typing import Generic, Protocol, TypeVar


class WirePayload(Protocol):
    source_timestamp_ns: int

    @property
    def wire_bytes(self) -> int: ...


PayloadT = TypeVar("PayloadT", bound=WirePayload)


@dataclass(frozen=True)
class ChannelConfig:
    latency_ms: float
    jitter_ms: float
    drop_probability: float
    max_queue: int
    seed: int

    def __post_init__(self) -> None:
        if self.latency_ms < 0 or self.jitter_ms < 0:
            raise ValueError("latency and jitter must be non-negative")
        if not 0 <= self.drop_probability <= 1:
            raise ValueError("drop_probability must be in [0, 1]")
        if self.max_queue <= 0:
            raise ValueError("max_queue must be positive")


@dataclass(frozen=True)
class Received(Generic[PayloadT]):
    payload: PayloadT
    delivered_at_ns: int
    age_ns: int


@dataclass
class ChannelStats:
    sent: int = 0
    enqueued: int = 0
    random_drops: int = 0
    overflow_drops: int = 0
    delivered: int = 0
    stale_reads: int = 0
    bytes_enqueued: int = 0
    bytes_delivered: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class DelayedLatestChannel(Generic[PayloadT]):
    """Deliver due messages and retain the newest causal value for reuse."""

    def __init__(self, config: ChannelConfig) -> None:
        self.config = config
        self.stats = ChannelStats()
        self._rng = random.Random(config.seed)
        self._queue: list[tuple[int, int, PayloadT]] = []
        self._enqueue_sequence = 0
        self._latest: tuple[PayloadT, int] | None = None

    def send(self, payload: PayloadT, *, now_ns: int) -> bool:
        if now_ns < 0 or payload.source_timestamp_ns > now_ns:
            raise ValueError("messages cannot be sent before their source timestamp")
        self.stats.sent += 1
        if self._rng.random() < self.config.drop_probability:
            self.stats.random_drops += 1
            return False
        if len(self._queue) >= self.config.max_queue:
            self.stats.overflow_drops += 1
            return False
        jitter_ns = round(self.config.jitter_ms * 1_000_000)
        jitter = self._rng.randint(-jitter_ns, jitter_ns) if jitter_ns else 0
        delivery_ns = max(
            now_ns,
            now_ns + round(self.config.latency_ms * 1_000_000) + jitter,
        )
        self._enqueue_sequence += 1
        heapq.heappush(
            self._queue, (delivery_ns, self._enqueue_sequence, payload)
        )
        self.stats.enqueued += 1
        self.stats.bytes_enqueued += payload.wire_bytes
        return True

    def read_latest(self, *, now_ns: int, max_age_ns: int) -> Received[PayloadT] | None:
        if now_ns < 0 or max_age_ns < 0:
            raise ValueError("now_ns and max_age_ns must be non-negative")
        while self._queue and self._queue[0][0] <= now_ns:
            delivered_at_ns, _, payload = heapq.heappop(self._queue)
            self.stats.delivered += 1
            self.stats.bytes_delivered += payload.wire_bytes
            if (
                self._latest is None
                or payload.source_timestamp_ns >= self._latest[0].source_timestamp_ns
            ):
                self._latest = (payload, delivered_at_ns)
        if self._latest is None:
            return None
        payload, delivered_at_ns = self._latest
        age_ns = now_ns - payload.source_timestamp_ns
        if age_ns < 0:
            raise RuntimeError("channel exposed a future message")
        if age_ns > max_age_ns:
            self.stats.stale_reads += 1
            return None
        return Received(payload, delivered_at_ns, age_ns)
