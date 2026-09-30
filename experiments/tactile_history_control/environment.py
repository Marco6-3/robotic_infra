"""MuJoCo FR3 task. Only observe() crosses the student observation boundary."""
from __future__ import annotations
from collections import deque
from types import SimpleNamespace
import copy
import mujoco
import numpy as np
from experiments.iros2027.contact.run import ROOT
from experiments.iros2027.contact.task import DisturbedGrasp
from experiments.iros2027.contact.settings import TaskSettings
from experiments.iros2027.i001_v2.physics import tactile_geometry
from fr3_sim.contact_proxy import ContactProxy

JOINT_ORDER = tuple([f'fr3_joint{i}' for i in range(1, 8)] + ['fr3_finger_joint1', 'fr3_finger_joint2'])

class RecoveryEnv:
    def __init__(self, config, seed, variant=0, disturbance_scale=1.0):
        self.c = config; self.seed = int(seed); self.variant = int(variant)
        rng = np.random.default_rng(seed)
        mass = float(rng.uniform(*config['mass_kg'])); friction = float(rng.uniform(*config['friction']))
        self.env = DisturbedGrasp(ROOT, seed=seed, settings=TaskSettings(mass_range_kg=(mass,mass), friction_range=(friction,friction)))
        e = self.env
        self.onset = int(rng.integers(*config['onset_ms'])) // config['sample_ms'] * config['sample_ms']
        self.duration = int(rng.integers(*config['duration_ms'])) // config['sample_ms'] * config['sample_ms']
        angle = float(rng.choice([np.pi/2, 3*np.pi/2])) + rng.uniform(-np.pi/6,np.pi/6); magnitude = float(rng.uniform(*config['disturbance_n'])) * disturbance_scale
        self.force = magnitude * np.array([np.cos(angle),np.sin(angle),0.])
        offset = rng.uniform(-config['initial_offset_m'],config['initial_offset_m'],2)
        e.data.qpos[e.object_qadr:e.object_qadr+2] += offset
        self.meta = dict(seed=seed,variant=variant,mass=mass,friction=friction,onset=self.onset,duration=self.duration,force=self.force.tolist(),offset=offset.tolist())
        self.sensor = ContactProxy(e.model,e.object_body,noise_std=0)
        # Same body/site placement, but no force sensor data used for student input.
        self.geometry = SimpleNamespace(sites=self.sensor.sites,bodies=self.sensor.bodies)
        self.qindices = [int(e.model.jnt_qposadr[e.model.joint(n).id]) for n in JOINT_ORDER]
        self.noise = np.random.default_rng([seed,variant,711])
        self.probe_rng = np.random.default_rng([seed,variant,812])
        self.history = deque(maxlen=config['history_ms']//config['sample_ms']+1)
        self.tick=0; self.last_action=0.; self.reference=None
        self.log=[]; self.dropped=False; self.stable_count=0; self.recovery_tick=None
        self.max_slip=0.; self.peak_force=0.; self.action_sum=0.; self.count=0
        self.initial_stable=False

    def physics(self, action):
        c=self.c;e=self.env
        action=float(np.clip(action,0,1))
        width=c['nominal_width_m']-c['max_closure_m']*action
        for _ in range(c['sample_ms']):
            arm,nominal=e.nominal(self.tick)
            e.data.ctrl[:7]=arm
            e.data.ctrl[7]=(nominal if self.tick<2100 else width)/2
            e.data.xfrc_applied[:]=0
            if self.onset<=self.tick<self.onset+self.duration:
                ramp=min((self.tick-self.onset)/c['ramp_ms'],1.)
                e.data.xfrc_applied[e.object_body,:3]=ramp*self.force
            mujoco.mj_step(e.model,e.data);self.tick+=1
        mujoco.mj_forward(e.model,e.data)
        if not np.isfinite(e.data.qpos).all():raise RuntimeError('nonfinite physics')
        self.last_action=action

    def observe(self):
        c=self.c;e=self.env
        tactile=tactile_geometry(e,self.geometry,c)
        tactile=np.maximum(0,tactile+self.noise.normal(0,c['tactile_noise_m'],len(tactile)))
        tactile=np.round(tactile/c['tactile_quantum_m'])*c['tactile_quantum_m']
        q=e.data.qpos[self.qindices].copy()
        quantum=np.r_[np.full(7,c['arm_quantum_rad']),np.full(2,c['finger_quantum_m'])]
        q=np.round(q/quantum)*quantum
        # Last ISSUED bounded residual; absolute width is nominal-max_closure*last_action.
        return np.r_[tactile,q,self.last_action].astype(np.float32)

    def truth(self):
        e=self.env
        forces,speed,contacts=self.sensor.truth(e.data)
        relative=e.object_position()-e.data.site_xpos[e.tcp]
        distance=float(np.linalg.norm(relative-self.reference)) if self.reference is not None else 0.
        return dict(force=float(forces[:,0].max()),min_force=float(forces[:,0].min()),speed=float(speed),contacts=contacts,
                    displacement=distance,height=float(e.object_position()[2]))

    def teacher(self):
        """Privileged displacement/contact velocity -> action ONLY, never features."""
        t=self.truth();p=self.c['teacher']
        return float(np.clip(p['base']+p['displacement_gain']*t['displacement']+p['speed_gain']*t['speed'],0,1))

    def prepare(self):
        c=self.c
        # Shared known-pose grasp/lift; short random, weak pre-policy actions.
        while self.tick<c['start_ms']:
            action=0.
            if 2100<=self.tick<c['start_ms']-50:
                if self.tick%50==0:self.probe=float(self.probe_rng.uniform(0,.12))
                action=getattr(self,'probe',0.)
            if self.tick>=c['start_ms']-c['history_ms']:
                self.history.append(self.observe())
            self.physics(action)
        self.reference=(self.env.object_position()-self.env.data.site_xpos[self.env.tcp]).copy()
        self.history.append(self.observe())
        self.observations=list(self.history)
        t=self.truth();self.initial_stable=t['height']>.49 and t['min_force']>.15 and t['speed']<.005
        return self

    def window(self):
        x=np.stack(self.history)
        if len(x)!=self.history.maxlen:raise RuntimeError('incomplete causal history')
        return x

    def record(self,action):
        t=self.truth();s=self.c['success']
        self.dropped |= t['displacement']>s['drop_displacement_m'] or t['height']<s['min_height_m']
        self.max_slip=max(self.max_slip,t['displacement']);self.peak_force=max(self.peak_force,t['force'])
        stable=t['speed']<s['stable_speed_m_s'] and t['min_force']>s['min_force_n'] and not self.dropped
        if self.tick>=self.onset+self.duration:
            self.stable_count=self.stable_count+1 if stable else 0
            if self.stable_count*self.c['sample_ms']>=s['stable_ms'] and self.recovery_tick is None:self.recovery_tick=self.tick
        self.action_sum+=abs(action)*self.c['max_closure_m'];self.count+=1
        self.log.append([self.tick,action,t['displacement'],t['speed'],t['force'],t['min_force'],t['height'],int(self.dropped)])

    def step(self,action):
        self.record(action);self.physics(action);self.history.append(self.observe())
        self.observations.append(self.history[-1])

    def metrics(self):
        t=self.truth();s=self.c['success']
        self.dropped |= t['displacement']>s['drop_displacement_m'] or t['height']<s['min_height_m']
        self.max_slip=max(self.max_slip,t['displacement']);self.peak_force=max(self.peak_force,t['force'])
        success=self.initial_stable and not self.dropped and self.max_slip<s['max_slip_m'] and self.stable_count*self.c['sample_ms']>=s['stable_ms']
        horizon=(self.c['end_ms']-(self.onset+self.duration))/1000
        latency=(self.recovery_tick-self.onset-self.duration)/1000 if success and self.recovery_tick is not None else horizon
        return dict(success=int(success),drop=int(self.dropped),max_slip_m=self.max_slip,final_slip_m=t['displacement'],
                    recovery_latency_s=latency,latency_censored=int(not success),peak_contact_force_n=self.peak_force,
                    mean_correction_m=self.action_sum/max(self.count,1),initial_stable=int(self.initial_stable))

    def snapshot(self):
        flag=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(self.env.model,flag));mujoco.mj_getState(self.env.model,self.env.data,state,flag)
        return dict(state=state,tick=self.tick,history=self.window(),last_action=self.last_action,reference=self.reference,
                    noise=copy.deepcopy(self.noise.bit_generator.state),initial_stable=self.initial_stable,
                    metrics_state={k:copy.deepcopy(getattr(self,k)) for k in ['dropped','max_slip','peak_force','stable_count','recovery_tick','action_sum','count']})

    def restore(self,snapshot):
        mujoco.mj_setState(self.env.model,self.env.data,snapshot['state'],mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.env.model,self.env.data)
        self.tick=snapshot['tick'];self.history.extend(snapshot['history']);self.last_action=snapshot['last_action']
        self.reference=snapshot['reference'];self.noise.bit_generator.state=snapshot['noise'];self.initial_stable=snapshot['initial_stable']
        for k,v in snapshot['metrics_state'].items():setattr(self,k,v)
        self.observations=list(self.history)
        return self
