"""Policy tactile = contact geometry indentation map, never contact wrench."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
import mujoco
from experiments.iros2027.contact.run import ROOT
from experiments.iros2027.contact.task import DisturbedGrasp
from experiments.iros2027.contact.settings import TaskSettings
from fr3_sim.contact_proxy import ContactProxy


def resolution(config):
    o=config['observation'];n=2*config['tactile']['grid']**2
    return np.r_[np.full(n,o['tactile_quantum_m']),np.full(7,o['arm_q_quantum_rad']),
        np.full(2,o['finger_q_quantum_m']),np.full(7,o['arm_dq_quantum_rad_s']),
        np.full(2,o['finger_dq_quantum_m_s']),np.full(7,o['action_arm_quantum_rad']),o['action_width_quantum_m']]


def finite_observation(exact, config, rng):
    out=exact.copy();n=2*config['tactile']['grid']**2
    out[:n]=np.maximum(0,out[:n]+rng.normal(0,config['observation']['tactile_noise_std_m'],n))
    res=resolution(config)
    # Commands are digital known inputs; quantization is shared with pair distance,
    # and equal future/anchor command is separately enforced bit-for-bit.
    return np.round(out/res)*res


def tactile_geometry(env, sensor, config):
    c=config['tactile'];g=c['grid'];out=np.zeros((2,g,g))
    gx,gz=np.meshgrid(np.linspace(-c['extent_x_m'],c['extent_x_m'],g),np.linspace(-c['extent_z_m'],c['extent_z_m'],g),indexing='ij')
    for i in range(env.data.ncon):
        contact=env.data.contact[i]
        if contact.geom1<0 or contact.geom2<0 or contact.dist>=0:continue
        bodies=[int(env.model.geom_bodyid[g]) for g in (contact.geom1,contact.geom2)]
        if env.object_body not in bodies:continue
        for side,body in enumerate(sensor.bodies):
            if body not in bodies:continue
            site=sensor.sites[side]
            local=env.data.site_xmat[site].reshape(3,3).T@(contact.pos-env.data.site_xpos[site])
            shape=np.exp(-((gx-local[0])**2+(gz-local[2])**2)/(2*c['splat_sigma_m']**2))
            out[side]=np.maximum(out[side],(-contact.dist)*shape)
    return out.ravel()


def probe_width(tick,start,duration,amp,direction,nominal):
    x=(tick-start)/duration
    shape=max(0.,min(3*x,1.,3*(1-x))) if 0<=x<=1 else 0.
    return nominal+direction*amp*shape


def slip_label(speed,contacts):
    mask=(np.asarray(speed)>.01)&(np.asarray(contacts)>0)
    return bool(len(mask)>=5 and np.any(np.convolve(mask.astype(int),np.ones(5,dtype=int),'valid')==5))


def simulate(job):
    seed,variant,split,c,directory=job
    path=Path(directory)/f'{split}-{seed}-{variant:02}.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as z:return json.loads(str(z['metadata']))
    begin=time.perf_counter();rng=np.random.default_rng([c['seed'],seed,variant]);shared=np.random.default_rng([c['seed'],seed])
    mass=float(shared.uniform(*c['mass_kg']));solver=float(shared.uniform(*c['solver_scale']))
    fixed=variant<int(c['episodes_per_seed']*c['fixed_fraction'])
    friction=c['fixed_friction'] if fixed else float(rng.uniform(*c['friction']))
    amp=float(rng.uniform(*c['probe_amplitude_m']));direction=int(rng.choice([-1,1]))
    anchor=c['anchor_tick'];duration=int(rng.integers(c['probe_duration_ms'][0],c['probe_duration_ms'][1]+1))
    start=anchor-int(rng.integers(c['probe_start_before_anchor_ms'][0],c['probe_start_before_anchor_ms'][1]+1))
    past_load=float(rng.uniform(*c['past_external_load_n']))
    offset=rng.uniform(-c['initial_offset_m'],c['initial_offset_m'],2)
    env=DisturbedGrasp(ROOT,seed=seed,settings=TaskSettings(mass_range_kg=(mass,mass),friction_range=(friction,friction)))
    env.model.geom_solref[:,0]*=solver
    env.data.qpos[env.object_qadr:env.object_qadr+2]+=offset
    mujoco.mj_forward(env.model,env.data)
    sensor=ContactProxy(env.model,env.object_body,noise_std=0)
    noise=np.random.default_rng([c['seed'],seed,variant,199])
    ticks=[];exact=[];finite=[];labels=[];future_inputs=[];controls=[];reference=None;anchor_state=None
    first=anchor-c['history_ms'];qidx=2*c['tactile']['grid']**2
    for tick in range(anchor+c['future_ms']+1):
        arm,width=env.nominal(tick)
        if tick>=700:width=probe_width(tick,start,duration,amp,direction,c['nominal_width_m'])
        load=(c['future_load_n']*min((tick-anchor)/c['future_ramp_ms'],1.)) if tick>=anchor else (past_load if start<=tick<start+duration else 0.)
        controls.append(np.r_[tick,arm,width,env.data.qpos[:9],env.data.qvel[:9]])
        if tick>=first or tick%c['sample_ms']==0:
            forces,speed,contacts=sensor.truth(env.data);pos=env.object_position();relative=pos-env.data.site_xpos[env.tcp]
            if tick==first:reference=relative.copy()
            displacement=float(np.linalg.norm(relative-reference) if reference is not None else 0.);drop=bool(pos[2]<.46 or displacement>.04)
            object_velocity=np.zeros(6);mujoco.mj_objectVelocity(env.model,env.data,mujoco.mjtObj.mjOBJ_BODY,env.object_body,object_velocity,0)
            labels.append(np.r_[tick,speed,contacts,pos,relative,displacement,int(drop),forces.ravel(),object_velocity,env.data.qpos[env.object_qadr+3:env.object_qadr+7]])
            if tick%c['sample_ms']==0:
                obs=np.r_[tactile_geometry(env,sensor,c),env.data.qpos[:9],env.data.qvel[:9],arm,width]
                ticks.append(tick);exact.append(obs);finite.append(finite_observation(obs,c,noise))
            if tick==anchor:
                eligible=bool(pos[2]>.49 and min(forces[:,0])>.15 and speed<c['max_anchor_speed_m_s'] and displacement<c['max_anchor_displacement_m'])
                anchor_state=np.r_[env.data.qpos,env.data.qvel]
            if tick>=anchor:future_inputs.append(np.r_[arm,width,load])
        env.data.ctrl[:7]=arm;env.data.ctrl[7]=width/2;env.data.xfrc_applied[:]=0;env.data.xfrc_applied[env.object_body,2]=-load
        if tick<anchor+c['future_ms']:
            mujoco.mj_step(env.model,env.data);mujoco.mj_forward(env.model,env.data)
        if not np.isfinite(env.data.qpos).all():raise RuntimeError('nonfinite state')
    truth=np.asarray(labels);future=truth[truth[:,0]>anchor];history=np.asarray(exact)[(np.asarray(ticks)<=anchor)&(np.asarray(ticks)>=first)]
    action_change=float(np.max(np.abs(np.diff(history[:,-8:],axis=0))))
    assert action_change>0,'active probe missing from history'
    meta=dict(seed=seed,variant=variant,split=split,fixed_friction=fixed,friction=friction,mass_kg=mass,solver_scale=solver,
        offset_m=offset.tolist(),probe_amplitude_m=amp,probe_direction=direction,probe_start=start,probe_duration=duration,past_external_load_n=past_load,
        eligible=eligible,label=int(slip_label(future[:,1],future[:,2])),action_change=action_change,
        future_hash=hashlib.sha256(np.asarray(future_inputs).tobytes()).hexdigest(),wall_seconds=time.perf_counter()-begin,episode=path.name)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,ticks_ns=np.asarray(ticks)*1000000,available_ns=np.asarray(ticks)*1000000,
        exact=exact,finite=finite,controls=controls,labels=labels,anchor_state=anchor_state,future_inputs=future_inputs,metadata=json.dumps(meta))
    tmp.replace(path);return meta
