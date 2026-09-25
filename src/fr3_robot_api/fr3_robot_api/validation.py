"""Canonical action validation shared by control and recording."""

from dataclasses import replace

import numpy as np

from .gripper import GripperMapper
from .types import Action


def canonical_action(action: Action, mapper: GripperMapper, arm_limits=None) -> Action:
    """Reject invalid commands; never silently clip a recorded target.

    Physical width is authoritative when supplied. The normalized field is
    derived from it, or checked for consistency if both fields were provided.
    arm_limits, when supplied by the model, is a (7, 2) array in radians.
    """
    if not np.isfinite(action.arm_joint_position_target_rad).all():
        raise ValueError("arm target must contain only finite values")
    if arm_limits is not None:
        limits = np.asarray(arm_limits, dtype=float)
        if limits.shape != (7, 2) or not np.isfinite(limits).all():
            raise ValueError("arm_limits must be finite with shape (7, 2)")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("arm lower limits must be below upper limits")
        q = action.arm_joint_position_target_rad
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError("arm target exceeds model joint limits")
    width = action.gripper_width_m
    if width is None:
        value = action.gripper_width_normalized
        if value is None or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("normalized width must be finite and in [0, 1]")
        width = mapper.width_from_normalized(value)
    if not np.isfinite(width) or not mapper.limits.width_lower_m <= width <= mapper.limits.width_upper_m:
        raise ValueError("gripper width exceeds model limits")
    normalized = mapper.normalized_from_width(width)
    if action.gripper_width_normalized is not None and not np.isclose(
        action.gripper_width_normalized, normalized, atol=1e-6, rtol=0
    ):
        raise ValueError("physical and normalized gripper widths disagree")
    return replace(action, gripper_width_m=float(width), gripper_width_normalized=float(normalized))
