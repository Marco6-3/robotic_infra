"""Object-filtered rigid contact proxy; not an optical or calibrated sensor."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass(frozen=True)
class ProxySample:
    source_ns: int
    available_ns: int
    values: tuple[float, ...]  # per finger: normal sum, signed local x/z force


class ContactProxy:
    def __init__(self, model, object_body: int, *, seed=0, noise_std=0.01, delay_ms=0):
        if noise_std < 0 or delay_ms < 0:
            raise ValueError('noise and delay must be non-negative')
        self.model = model
        self.object_body = object_body
        self.sites = [model.site(f'fr3_{s}_fingertip_touch_site').id for s in ('left', 'right')]
        self.bodies = [int(model.site_bodyid[s]) for s in self.sites]
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std
        self.delay_ns = round(delay_ms * 1e6)
        self.pending = deque()
        self.latest = None

    def truth(self, data):
        """Force on each finger in its site frame, and relative tangential speed."""
        force = np.zeros((2, 3))
        speed = 0.0
        contacts = 0
        for index in range(data.ncon):
            c = data.contact[index]
            if c.geom1 < 0 or c.geom2 < 0 or c.efc_address < 0:
                continue
            bodies = [int(self.model.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
            if self.object_body not in bodies:
                continue
            for finger, body in enumerate(self.bodies):
                if body not in bodies:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, data, index, wrench)
                if wrench[0] <= 0:
                    continue
                world = c.frame.reshape(3, 3).T @ wrench[:3]
                world *= 1 if bodies[1] == body else -1
                local = data.site_xmat[self.sites[finger]].reshape(3, 3).T @ world
                force[finger] += [wrench[0], local[0], local[2]]
                jac_obj = np.zeros((3, self.model.nv))
                jac_finger = np.zeros_like(jac_obj)
                mujoco.mj_jac(self.model, data, jac_obj, None, c.pos, self.object_body)
                mujoco.mj_jac(self.model, data, jac_finger, None, c.pos, body)
                relative = (jac_obj - jac_finger) @ data.qvel
                normal = c.frame[:3]
                speed = max(speed, float(np.linalg.norm(relative - normal * (normal @ relative))))
                contacts += 1
        return force, speed, contacts

    def sample(self, now_ns, force):
        values = force + self.rng.normal(0, self.noise_std, (2, 3))
        values[:, 0] = np.maximum(values[:, 0], 0)
        sample = ProxySample(now_ns, now_ns + self.delay_ns, tuple(values.ravel()))
        self.pending.append(sample)
        return sample

    def read(self, now_ns, max_age_ns=50_000_000):
        while self.pending and self.pending[0].available_ns <= now_ns:
            self.latest = self.pending.popleft()
        if self.latest is None or now_ns - self.latest.source_ns > max_age_ns:
            return None
        return self.latest
