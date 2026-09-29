"""Dependency-light dry-run runtime for I002 communication experiments."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from experiments.iros2027.i002.channel import ChannelConfig, DelayedLatestChannel
from experiments.iros2027.i002.contracts import SemanticMessage, TactileFeedbackMessage
from experiments.iros2027.stage0 import periodic_timestamps_ns


@dataclass(frozen=True)
class SmokeConfig:
    duration_s: float
    controller_hz: float
    fast_policy_hz: float
    semantic_policy_hz: float
    token_count: int
    max_staleness_ms: float
    contact_event_hz: float
    bidirectional: bool
    semantic_channel: ChannelConfig
    feedback_channel: ChannelConfig

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SmokeConfig":
        rates = raw["rates"]
        smoke = raw["smoke"]
        channel = smoke["channel"]
        feedback = smoke.get("feedback_channel", channel)
        seed = int(raw["experiment"]["seed"])
        return cls(
            duration_s=float(smoke.get("duration_s", raw["experiment"]["duration_s"])),
            controller_hz=float(rates["controller_hz"]),
            fast_policy_hz=float(rates["fast_policy_hz"]),
            semantic_policy_hz=float(rates["semantic_policy_hz"]),
            token_count=int(smoke["token_count"]),
            max_staleness_ms=float(smoke["max_staleness_ms"]),
            contact_event_hz=float(smoke["contact_event_hz"]),
            bidirectional=bool(smoke["bidirectional"]),
            semantic_channel=ChannelConfig(seed=seed, **channel),
            feedback_channel=ChannelConfig(seed=seed + 1, **feedback),
        )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_smoke(config: SmokeConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    semantic_channel: DelayedLatestChannel[SemanticMessage] = DelayedLatestChannel(
        config.semantic_channel
    )
    feedback_channel: DelayedLatestChannel[TactileFeedbackMessage] = DelayedLatestChannel(
        config.feedback_channel
    )
    controller_ticks = periodic_timestamps_ns(config.controller_hz, config.duration_s)
    semantic_ticks = set(periodic_timestamps_ns(config.semantic_policy_hz, config.duration_s))
    fast_ticks = set(periodic_timestamps_ns(config.fast_policy_hz, config.duration_s))
    contact_ticks = set(periodic_timestamps_ns(config.contact_event_hz, config.duration_s))
    event_ticks = sorted(semantic_ticks | fast_ticks | contact_ticks)
    max_age_ns = round(config.max_staleness_ms * 1_000_000)
    semantic_sequence = 0
    feedback_sequence = 0
    semantic_ages_ms: list[float] = []
    decisions_without_context = 0
    feedback_received = 0
    future_context_uses = 0
    events: list[dict[str, Any]] = []

    # Policy and sensor rates need not divide the 1 kHz controller rate. The
    # dry-run therefore advances over exact source event timestamps instead of
    # silently snapping 60 Hz events to integer controller milliseconds.
    for now_ns in event_ticks:
        if now_ns in semantic_ticks:
            feedback = feedback_channel.read_latest(now_ns=now_ns, max_age_ns=max_age_ns)
            if feedback is not None:
                feedback_received += 1
            tokens = tuple(
                math.sin(now_ns / 1_000_000_000 + index) for index in range(config.token_count)
            )
            message = SemanticMessage(
                sequence=semantic_sequence,
                source_timestamp_ns=now_ns,
                coarse_action=(0.0, 0.0, -0.001, 0.0, 0.0, 0.0, 0.5),
                tokens=tokens,
            )
            accepted = semantic_channel.send(message, now_ns=now_ns)
            events.append(
                {
                    "event": "semantic_send",
                    "timestamp_ns": now_ns,
                    "sequence": semantic_sequence,
                    "accepted": accepted,
                    "wire_bytes": message.wire_bytes,
                }
            )
            semantic_sequence += 1

        if config.bidirectional and now_ns in contact_ticks:
            feedback_message = TactileFeedbackMessage(
                sequence=feedback_sequence,
                source_timestamp_ns=now_ns,
                contact_event=True,
                summary=(1.0, 0.1 * math.cos(now_ns / 1_000_000_000)),
            )
            feedback_channel.send(feedback_message, now_ns=now_ns)
            feedback_sequence += 1

        if now_ns in fast_ticks:
            received = semantic_channel.read_latest(now_ns=now_ns, max_age_ns=max_age_ns)
            if received is None:
                decisions_without_context += 1
                age_ms = None
            else:
                if received.payload.source_timestamp_ns > now_ns:
                    future_context_uses += 1
                age_ms = received.age_ns / 1_000_000
                semantic_ages_ms.append(age_ms)
            events.append(
                {
                    "event": "fast_decision",
                    "timestamp_ns": now_ns,
                    "semantic_age_ms": age_ms,
                }
            )

    if future_context_uses:
        raise RuntimeError("future semantic context reached the fast policy")
    metrics = {
        "status": "communication_smoke_passed",
        "claim_boundary": (
            "The deterministic communication runtime passed; no learned policy, "
            "contact task, or task-success comparison was executed."
        ),
        "duration_s": config.duration_s,
        "controller_ticks": len(controller_ticks),
        "semantic_calls": len(semantic_ticks),
        "fast_decisions": len(fast_ticks),
        "contact_events": len(contact_ticks) if config.bidirectional else 0,
        "decisions_without_context": decisions_without_context,
        "future_context_uses": future_context_uses,
        "semantic_age_ms": {
            "mean": statistics.fmean(semantic_ages_ms) if semantic_ages_ms else 0.0,
            "p95": _percentile(semantic_ages_ms, 0.95),
            "max": max(semantic_ages_ms, default=0.0),
        },
        "feedback_messages_received": feedback_received,
        "semantic_channel": semantic_channel.stats.as_dict(),
        "feedback_channel": feedback_channel.stats.as_dict(),
    }
    return metrics, events
