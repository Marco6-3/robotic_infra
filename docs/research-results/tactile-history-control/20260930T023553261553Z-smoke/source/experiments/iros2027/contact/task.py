"""Known-pose grasp/lift/disturbance task built from the existing FR3 model."""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from experiments.iros2027.contact.settings import TaskSettings


class DisturbedGrasp:
    def __init__(self, root: Path, *, seed=0, settings=None):
        settings = settings or TaskSettings()
        rng = np.random.default_rng(seed)
        self.mass = float(rng.uniform(*settings.mass_range_kg))
        self.friction = float(rng.uniform(*settings.friction_range))
        self.disturbance_n = float(rng.uniform(*settings.disturbance_range_n))
        model_dir = root / 'src/fr3_description/models'
        arm = ET.parse(model_dir / 'fr3_hand.xml').getroot()
        arm.find('compiler').set('meshdir', str((model_dir / 'assets').resolve()))
        key = arm.find('keyframe')
        home = np.fromstring(key.find('key').get('qpos'), sep=' ')
        arm.remove(key)
        scene = ET.parse(model_dir / 'scene.xml').getroot()
        for tag in ('asset', 'worldbody'):
            for element in scene.find(tag):
                arm.find(tag).append(element)
        body = ET.SubElement(arm.find('worldbody'), 'body', name='task_object', pos='.5545 0 .426')
        ET.SubElement(body, 'freejoint', name='object_free')
        ET.SubElement(body, 'geom', name='task_object_geom', type='box', size='.02 .015 .025',
                      mass=str(self.mass), friction=f'{self.friction} .005 .0001', rgba='.9 .3 .1 1')
        # Set both contacting sides explicitly; MuJoCo's default friction mixing
        # would otherwise override the object's sampled coefficient.
        for side in ('left', 'right'):
            finger = arm.find(f".//body[@name='fr3_{side}_finger']")
            for geom in finger.findall('geom'):
                geom.set('friction', f'{self.friction} .005 .0001')
        self.xml = ET.tostring(arm, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:len(home)] = home
        self.home = home.copy()
        self.object_body = self.model.body('task_object').id
        self.object_qadr = int(self.model.jnt_qposadr[self.model.joint('object_free').id])
        self.tcp = self.model.site('fr3_hand_tcp').id
        mujoco.mj_forward(self.model, self.data)
        self.orientation = self.data.site_xmat[self.tcp].reshape(3, 3).copy()
        self.q_down = self._ik(np.array([.5545, 0, .426]), home[:7])
        self.q_up = self._ik(np.array([.5545, 0, .53]), self.q_down)

    def _ik(self, position, initial):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.data.qpos
        data.qpos[:7] = initial
        for _ in range(160):
            mujoco.mj_forward(self.model, data)
            rotation = data.site_xmat[self.tcp].reshape(3, 3)
            error = np.r_[position - data.site_xpos[self.tcp],
                          .5 * sum(np.cross(rotation[:, i], self.orientation[:, i]) for i in range(3))]
            if np.linalg.norm(error) < 1e-6:
                return data.qpos[:7].copy()
            jp = np.zeros((3, self.model.nv)); jr = np.zeros_like(jp)
            mujoco.mj_jacSite(self.model, data, jp, jr, self.tcp)
            jac = np.vstack([jp[:, :7], jr[:, :7]])
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
            data.qpos[:7] = np.clip(data.qpos[:7] + np.clip(delta, -.05, .05),
                                   self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
        raise RuntimeError('known-pose IK failed')

    def nominal(self, tick):
        if tick < 700:
            return self.home[:7] + (self.q_down - self.home[:7]) * min(tick / 650, 1), .08
        if tick < 1400:
            return self.q_down.copy(), .032
        if tick < 2100:
            return self.q_down + (self.q_up - self.q_down) * min((tick - 1400) / 650, 1), .032
        return self.q_up.copy(), .032

    def step(self, tick, arm_target, width):
        self.data.ctrl[:7] = arm_target
        self.data.ctrl[7] = width / 2
        self.data.xfrc_applied[:] = 0
        if 2500 <= tick < 2800:
            self.data.xfrc_applied[self.object_body, 2] = -self.disturbance_n * min((tick - 2500) / 100, 1.0)
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('non-finite simulation state')

    def object_position(self):
        return self.data.qpos[self.object_qadr:self.object_qadr + 3].copy()
