"""Causal I001 contexts from independent streams; never reads labels.jsonl."""
from __future__ import annotations
import bisect
import json
from pathlib import Path
import numpy as np


class EpisodeHistory:
    def __init__(self, episode: Path):
        self.tactile = [json.loads(line) for line in (episode / 'tactile.jsonl').read_text().splitlines()]
        self.control = [json.loads(line) for line in (episode / 'control.jsonl').read_text().splitlines()]
        self.control_times = [r['timestamp_ns'] for r in self.control]
        for times in (self.control_times, [r['source_ns'] for r in self.tactile]):
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError('stream timestamps must strictly increase')
        if any(r['available_ns'] < r['source_ns'] for r in self.tactile):
            raise ValueError('sensor availability precedes acquisition')

    def context(self, decision_ns: int, history_ms: float):
        """Window relative to latest available source; zero means one frame.

        Actions/proprioception are aligned causally at each tactile source time.
        All variants receive source/availability times, so age is observable.
        """
        if decision_ns < 0 or history_ms < 0:
            raise ValueError('decision and history must be non-negative')
        available = [r for r in self.tactile if r['source_ns'] <= decision_ns and r['available_ns'] <= decision_ns]
        if available:
            lower = available[-1]['source_ns'] - round(history_ms * 1e6)
            available = [r for r in available if r['source_ns'] >= lower]
        rows = []
        for sample in available:
            index = bisect.bisect_right(self.control_times, sample['source_ns']) - 1
            if index < 0:
                continue
            rows.append((sample, self.control[index]))
        return {
            'source_ns': np.array([s['source_ns'] for s, _ in rows], dtype=np.int64),
            'available_ns': np.array([s['available_ns'] for s, _ in rows], dtype=np.int64),
            'tactile': np.array([s['values'] for s, _ in rows], dtype=float).reshape(-1, 6),
            'action': np.array([c['arm_target_rad'] + [c['applied_width_target_m']] for _, c in rows], dtype=float).reshape(-1, 8),
            'proprioception': np.array([c['q'] + c['dq'] for _, c in rows], dtype=float).reshape(-1, 18),
        }
