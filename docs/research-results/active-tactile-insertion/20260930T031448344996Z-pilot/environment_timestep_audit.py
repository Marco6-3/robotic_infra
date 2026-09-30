"""Physical Cartesian gripper, compliant held peg, faceted circular chamfered hole.

T is pad/shaft contact INDENTATION geometry only. Never a wrench-to-taxel map.
"""
from collections import deque
import copy
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

JOINT_ORDER=('carriage_x','carriage_y','carriage_z')


def scene(c,hole,mass,friction,kp):
    root=ET.Element('mujoco',model='active_tactile_insertion')
    ET.SubElement(root,'compiler',angle='radian',autolimits='true')
    ET.SubElement(root,'option',timestep=str(c['physics_ms']/1000),gravity='0 0 0',integrator='implicitfast',iterations='50')
    default=ET.SubElement(root,'default');ET.SubElement(default,'geom',solref='.015 1',solimp='.9 .95 .0005',friction=f'{friction} .001 .0001',condim='3')
    assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
    ET.SubElement(world,'light',pos='0 -.1 .2',dir='0 0 -1')
    ET.SubElement(world,'camera',name='overview',pos='.065 -.085 .07',xyaxes='.8 .6 0 -.3 .4 .866')
    fixture=ET.SubElement(world,'body',name='hole',pos=f'{hole[0]} {hole[1]} 0')
    # Convex wedges approximate a circular bore; no invisible analytical forces.
    inner=c['hole_radius_m'];depth=c['hole_depth_m'];ch=c['chamfer_m'];chd=c['chamfer_depth_m']
    for i in range(c['hole_facets']):
        angles=2*np.pi*np.array([i,i+1])/c['hole_facets'];vertices=[]
        for a in angles:
            for r,z in [(.025,0),(inner+ch,0),(inner,-chd),(inner,-depth),(.025,-depth)]:vertices.extend([r*np.cos(a),r*np.sin(a),z])
        ET.SubElement(assets,'mesh',name=f'ring{i}',vertex=' '.join(map(str,vertices)))
        ET.SubElement(fixture,'geom',name=f'hole{i}',type='mesh',mesh=f'ring{i}',contype='2',conaffinity='1',rgba='.4 .45 .5 1')
    carrier=ET.SubElement(world,'body',name='carriage',pos='0 0 .0353')
    ET.SubElement(carrier,'inertial',pos='0 0 0',mass='.3',diaginertia='.0002 .0002 .0002')
    for axis,name in zip(np.eye(3),JOINT_ORDER):
        ET.SubElement(carrier,'joint',name=name,type='slide',axis=' '.join(map(str,axis)),damping='80' if name.endswith('z') else '0',range='-.02 .01')
    for side,sign in [('left',-1),('right',1)]:
        x=sign*(c['peg_radius_m']+.00195)
        ET.SubElement(carrier,'geom',name=f'pad_{side}',type='box',pos=f'{x} 0 0',size='.002 .012 .008',mass='.01',contype='4',conaffinity='1',rgba='.25 .5 .85 1')
        ET.SubElement(carrier,'site',name=f'pad_{side}_site',pos=f'{sign*c["peg_radius_m"]} 0 0',size='.001')
    peg=ET.SubElement(carrier,'body',name='peg',pos='0 0 -.015')
    for axis,name in zip(np.eye(3),['mount_x','mount_y','mount_z']):
        ET.SubElement(peg,'joint',name=name,type='slide',axis=' '.join(map(str,axis)),stiffness=str(c['mount_stiffness']),damping=str(c['mount_damping']),range='-.003 .003')
    # Shaft reaches through the gripper; distal flat tip enters hole.
    ET.SubElement(peg,'geom',name='shaft',type='capsule',size=f'{c["peg_radius_m"]} .015',mass=str(mass),contype='1',conaffinity='6',rgba='.9 .55 .2 1')
    ET.SubElement(peg,'site',name='tip',pos='0 0 -.02',size='.0005')
    contact=ET.SubElement(root,'contact')
    for side in ['left','right']:ET.SubElement(contact,'pair',geom1='shaft',geom2=f'pad_{side}',condim='3')
    actuator=ET.SubElement(root,'actuator')
    for name in JOINT_ORDER[:2]:ET.SubElement(actuator,'position',joint=name,kp=str(kp),kv=str(c['motor_damping']),ctrlrange='-.01 .01',forcerange='-15 15')
    ET.SubElement(actuator,'motor',joint='carriage_z',ctrlrange='-20 20')
    return ET.tostring(root,encoding='unicode')


class InsertionEnv:
    def __init__(self,c,seed,mirror=1,force_scale=1.):
        self.c=c;self.seed=int(seed);self.mirror=mirror;rng=np.random.default_rng(seed)
        radius=rng.uniform(*c['offset_radius_m']);angle=rng.uniform(0,2*np.pi)
        self.hole=mirror*radius*np.array([np.cos(angle),np.sin(angle)])
        self.mass=float(rng.uniform(*c['mass_kg']));self.friction=float(rng.uniform(*c['friction']));self.kp=float(rng.uniform(*c['motor_kp']))
        self.xml=scene(c,self.hole,self.mass,self.friction,self.kp)
        self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.tip=self.model.site('tip').id;self.peg=self.model.body('peg').id;self.shaft=self.model.geom('shaft').id
        self.padids=[self.model.geom('pad_'+s).id for s in ['left','right']]
        self.sites=[self.model.site('pad_'+s+'_site').id for s in ['left','right']]
        self.hole_geoms={self.model.geom(f'hole{i}').id for i in range(c['hole_facets'])}
        self.qidx=[int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in JOINT_ORDER]
        self.rng=np.random.default_rng([seed,533]);self.proberng=np.random.default_rng([seed,721])
        self.target=np.zeros(2);self.last_action=np.zeros(2);self.tick=0
        self.history=deque(maxlen=c['history_ms']//c['sample_ms']+1)
        self.force_scale=force_scale;self.peak_force=0.;self.force_failure=False;self.success=False;self.success_tick=None;self.stable_since=None
        self.log=[];self.observations=[];self.actions=[];self.path=0.;self.probes=[];self.start_tip=None
        self.contact_tick=None
        mujoco.mj_forward(self.model,self.data)

    def hole_force(self):
        total=0.
        for i in range(self.data.ncon):
            ct=self.data.contact[i]
            if self.shaft in (ct.geom1,ct.geom2) and (ct.geom1 in self.hole_geoms or ct.geom2 in self.hole_geoms):
                w=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,w);total+=np.linalg.norm(w[:3])
        return float(total)

    def tactile(self):
        c=self.c;g=c['tactile_grid'];out=np.zeros((2,g,g))
        yy,zz=np.meshgrid(np.linspace(-c['tactile_extent_y_m'],c['tactile_extent_y_m'],g),np.linspace(-c['tactile_extent_z_m'],c['tactile_extent_z_m'],g),indexing='ij')
        for i in range(self.data.ncon):
            ct=self.data.contact[i]
            if ct.dist>=0 or self.shaft not in (ct.geom1,ct.geom2):continue
            for side,pad in enumerate(self.padids):
                if pad not in (ct.geom1,ct.geom2):continue
                rel=ct.pos-self.data.site_xpos[self.sites[side]]
                splat=np.exp(-((yy-rel[1])**2+(zz-rel[2])**2)/(2*c['tactile_sigma_m']**2))
                out[side]=np.maximum(out[side],-ct.dist*splat)
        return out.ravel()

    def observe(self):
        c=self.c;t=np.maximum(0,self.tactile()+self.rng.normal(0,c['tactile_noise_m'],32))
        t=np.round(t/c['tactile_quantum_m'])*c['tactile_quantum_m']
        q=np.round(self.data.qpos[self.qidx]/c['q_quantum_m'])*c['q_quantum_m']
        return np.r_[t,q,self.last_action].astype(np.float32)

    def teacher(self):
        error=self.hole-self.data.site_xpos[self.tip,:2]
        delta=np.clip(.15*error,-self.c['max_action_m'],self.c['max_action_m'])
        # A privileged teacher may reduce lateral speed under high contact load.
        if self.hole_force()>.8*self.c['safe_force_n']:delta*=.25
        return delta/self.c['max_action_m']

    def physics(self,action,monitor=True):
        c=self.c;action=np.clip(np.asarray(action,dtype=float),-1,1)
        old=self.target.copy();self.target=np.clip(self.target+c['max_action_m']*action,-c['workspace_m'],c['workspace_m'])
        self.last_action=(self.target-old)/c['max_action_m'];self.path+=float(np.linalg.norm(self.target-old))
        self.data.ctrl[:2]=self.target
        self.data.ctrl[2]=-c['preload_n']*self.force_scale
        if self.tick<c['settle_ms']+c['probe_ms']:
            # Encoder-based insertion-depth stop during probing, common to every policy.
            self.data.ctrl[2]=float(np.clip(-2000*(self.data.qpos[self.qidx[2]]+c['probe_depth_cap_m']),-c['preload_n']*self.force_scale,5))
        for _ in range(int(c['sample_ms']//c['physics_ms'])):
            mujoco.mj_step(self.model,self.data);mujoco.mj_forward(self.model,self.data);self.tick+=c['physics_ms']
            force=self.hole_force();self.peak_force=max(self.peak_force,force)
            if force>.05 and self.contact_tick is None:self.contact_tick=self.tick
            if monitor and force>c['safe_force_n']:
                self.force_failure=True
                break
            pos=self.data.site_xpos[self.tip];depth=-pos[2];aligned=np.linalg.norm(pos[:2]-self.hole)<c['hole_radius_m']-c['peg_radius_m']
            if monitor and depth>=c['insertion_depth_m'] and aligned and not self.force_failure:
                if self.stable_since is None:self.stable_since=self.tick
                if self.tick-self.stable_since>=c['success_hold_ms']:
                    self.success=True
                    if self.success_tick is None:self.success_tick=self.tick
            else:self.stable_since=None
        if not np.isfinite(self.data.qpos).all():raise RuntimeError('nonfinite simulation')

    def step(self,action):
        self.physics(action)
        self.history.append(self.observe());self.observations.append(self.history[-1]);self.actions.append(self.last_action.copy())
        pos=self.data.site_xpos[self.tip]
        self.log.append([self.tick,*self.last_action,*pos,self.hole_force(),self.peak_force,int(self.force_failure),int(self.success)])

    def prepare(self):
        # Settle onto rim first, then execute a variable-duration active probe schedule.
        while self.tick<self.c['settle_ms'] and not self.force_failure:self.physics([0,0])
        self.start_tip=self.data.site_xpos[self.tip].copy();self.probe_start=self.tick
        self.history.append(self.observe())
        goal=np.zeros(2);remaining=0
        while self.tick<self.probe_start+self.c['probe_ms'] and not self.force_failure:
            if remaining<=0:
                duration=int(self.proberng.integers(*self.c['probe_duration_ms']))//self.c['sample_ms']*self.c['sample_ms']
                amplitude=float(self.proberng.uniform(*self.c['probe_amplitude_m']));angle=float(self.proberng.uniform(0,2*np.pi))
                goal=self.mirror*amplitude*np.array([np.cos(angle),np.sin(angle)])
                remaining=duration
                self.probes.append(dict(tick=self.tick,duration_ms=duration,amplitude_m=amplitude,goal=goal.tolist()))
            # Bound the actual command, no action-shape timestamp shortcut.
            action=np.clip((goal-self.target)/(max(remaining,self.c['sample_ms'])/self.c['sample_ms'])/self.c['max_action_m'],-1,1)
            self.step(action);remaining-=self.c['sample_ms']
        while len(self.history)<self.history.maxlen:self.history.appendleft(self.history[0].copy())
        self.control_start=self.tick;self.control_end=self.tick+self.c['horizon_ms']
        self.observations=list(self.history);self.actions=[];self.log=[]
        self.prefix_force_failure=self.force_failure;self.prefix_success=self.success;self.prefix_contact=self.contact_tick is not None
        return self

    def done(self):return self.force_failure or self.success or self.tick>=self.control_end
    def window(self):
        if len(self.history)!=self.history.maxlen:raise RuntimeError('incomplete history')
        return np.stack(self.history)

    def metrics(self):
        pos=self.data.site_xpos[self.tip];elapsed=(self.tick-self.control_start)/1000
        return dict(success=int(self.success and not self.force_failure),force_failure=int(self.force_failure),timeout=int(not self.success and not self.force_failure),
                    insertion_depth_m=float(-pos[2]),lateral_error_m=float(np.linalg.norm(pos[:2]-self.hole)),peak_force_n=self.peak_force,
                    latency_s=elapsed if self.success else self.c['horizon_ms']/1000,search_path_m=self.path,
                    prefix_force_failure=int(self.prefix_force_failure),prefix_success=int(self.prefix_success),prefix_contact=int(self.prefix_contact))

    def snapshot(self):
        flag=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(self.model,flag));mujoco.mj_getState(self.model,self.data,state,flag)
        fields=['target','last_action','tick','force_scale','peak_force','force_failure','success','success_tick','stable_since','path','probes','start_tip','contact_tick','probe_start','control_start','control_end','prefix_force_failure','prefix_success','prefix_contact']
        return dict(state=state,fields={k:copy.deepcopy(getattr(self,k)) for k in fields},history=self.window(),rng=copy.deepcopy(self.rng.bit_generator.state))

    def restore(self,s):
        mujoco.mj_setState(self.model,self.data,s['state'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(self.model,self.data)
        for k,v in s['fields'].items():setattr(self,k,copy.deepcopy(v))
        self.history.extend(s['history']);self.observations=list(self.history);self.rng.bit_generator.state=copy.deepcopy(s['rng'])
        return self
