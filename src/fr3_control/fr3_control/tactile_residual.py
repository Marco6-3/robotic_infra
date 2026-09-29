"""Bounded force-proxy rule controller; consumes no object state or slip labels."""
from __future__ import annotations
import numpy as np


class GripResidual:
    def __init__(self, *, max_closure_m=.016, step_m=.002, force_limit_n=12.0, shear_ratio_threshold=.45):
        if min(max_closure_m, step_m, force_limit_n) <= 0:
            raise ValueError('residual bounds must be positive')
        self.max_closure_m = max_closure_m
        self.step_m = step_m
        self.force_limit_n = force_limit_n
        self.shear_ratio_threshold = shear_ratio_threshold
        self.offset_m = 0.0

    def update(self, values):
        if values is None:
            # Keep the last bounded grip on stale input; do not blindly open.
            return self.offset_m
        force = np.asarray(values).reshape(2, 3)
        if not np.isfinite(force).all():
            raise ValueError('non-finite tactile input')
        normal = force[:, 0]
        shear = np.linalg.norm(force[:, 1:], axis=1)
        if max(normal) > self.force_limit_n:
            self.offset_m = min(0.0, self.offset_m + self.step_m)
        elif min(normal) > .15 and max(shear / np.maximum(normal, .15)) > self.shear_ratio_threshold:
            self.offset_m = max(-self.max_closure_m, self.offset_m - self.step_m)
        return self.offset_m

    def compose(self, nominal_width_m):
        return float(np.clip(nominal_width_m + self.offset_m, 0.0, .08))
