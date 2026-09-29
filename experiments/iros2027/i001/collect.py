"""Real MuJoCo interventions with identical future inputs; no state editing."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import yaml
import numpy as np
import mujoco
from experiments.iros2027.contact.run import ROOT
from experiments.iros2027.contact.task import DisturbedGrasp
from experiments.iros2027.contact.settings import TaskSettings
from experiments.iros2027.common.run_artifacts import RunArtifacts
from fr3_sim.contact_proxy import ContactProxy


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def future_load(tick, config):
    return config['future_load_n'] * np.clip((tick-config['anchor_tick']) / config['future_ramp_ticks'], 0., 1.)


def simulate(job):
    seed, bin_id, probe, split, config, directory = job
    path = Path(directory) / f'{split}-{seed}-friction{bin_id}-probe{int(probe)}.npz'
    if path.exists():
        with np.load(path) as data:
            return json.loads(str(data['metadata']))
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    mass = float(rng.uniform(*config['mass_range_kg']))
    width = float(rng.uniform(*config['width_range_m']))
    probe_n = float(rng.uniform(*config['probe_range_n']))
    friction = float(np.random.default_rng([seed, bin_id, 198]).uniform(*config['friction_bins'][bin_id]))
    env = DisturbedGrasp(ROOT, seed=seed, settings=TaskSettings(mass_range_kg=(mass,mass), friction_range=(friction,friction)))
    noise_seed = int(np.random.SeedSequence([seed,bin_id,int(probe),819]).generate_state(1)[0])
    sensor = ContactProxy(env.model, env.object_body, seed=noise_seed, noise_std=config['noise_std_n'])
    anchor = config['anchor_tick']; period = config['sensor_period_ticks']
    first = anchor - max(config['history_ms'])
    ticks=[]; observations=[]; proprio=[]; actions=[]; truth=[]; labels=[]; loads=[]; future_inputs=[]
    anchor_state=None
    for tick in range(anchor + config['horizon_ticks'] + 1):
        arm, nominal_width = env.nominal(tick)
        target_width = width if tick >= 700 else nominal_width
        load = float(future_load(tick,config)) if tick >= anchor else (
            probe_n if probe and config['probe_start_tick'] <= tick < config['probe_start_tick'] + config['probe_duration_ticks'] else 0.)
        if tick >= first:
            force,speed,contacts = sensor.truth(env.data)
            pos=env.object_position()
            labels.append([tick,speed,contacts,*pos.tolist()])
            if tick % period == 0:
                ticks.append(tick)
                observations.append(sensor.sample(tick*1_000_000, force).values)
                truth.append(force.ravel())
                proprio.append(np.r_[env.data.qpos[:9],env.data.qvel[:9]])
                actions.append(np.r_[arm,target_width])
                loads.append(load)
            if tick == anchor:
                anchor_state=np.r_[env.data.qpos,env.data.qvel,env.data.act]
                stable=bool(pos[2]>.49 and min(force[:,0])>.15 and speed<.003)
            if tick >= anchor:
                future_inputs.append(np.r_[arm,target_width,load])
        env.data.ctrl[:7]=arm; env.data.ctrl[7]=target_width/2
        env.data.xfrc_applied[:]=0
        env.data.xfrc_applied[env.object_body,2]=-load
        if tick < anchor+config['horizon_ticks']:
            mujoco.mj_step(env.model,env.data);mujoco.mj_forward(env.model,env.data)
        if not np.isfinite(env.data.qpos).all():
            raise RuntimeError('nonfinite physics')
    rows=np.asarray(labels); future=rows[rows[:,0]>anchor]
    slip=consecutive_slip(future[:,1],future[:,2])
    meta=dict(seed=seed,bin_id=bin_id,probe=probe,split=split,mass_kg=mass,friction=friction,
              width_m=width,probe_n=probe_n,eligible=stable,label=int(slip),noise_seed=noise_seed,
              future_input_sha256=hashlib.sha256(np.asarray(future_inputs).tobytes()).hexdigest(),
              wall_seconds=time.perf_counter()-start,episode=path.name)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,ticks=ticks,tactile=observations,proprio=proprio,actions=actions,
                            force_truth=truth,labels=labels,external_load=loads,anchor_state=anchor_state,
                            future_inputs=future_inputs,metadata=json.dumps(meta))
    temporary.replace(path)
    return meta


def consecutive_slip(speed, contacts):
    count=0
    for v,c in zip(speed,contacts):
        count=count+1 if v>.01 and c>0 else 0
        if count>=5:return True
    return False


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'))
    p.add_argument('--output',type=Path,default=ROOT/'runs/i001-aliasing')
    p.add_argument('--resume',type=Path)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();config=yaml.safe_load(args.config.read_text())
    source_paths=sorted((ROOT/'experiments/iros2027/i001').glob('*.py')) + [
        ROOT/'experiments/iros2027/contact/task.py',ROOT/'experiments/iros2027/contact/settings.py',
        ROOT/'experiments/iros2027/contact/run.py',ROOT/'experiments/iros2027/common/run_artifacts.py',
        ROOT/'src/fr3_sim/fr3_sim/contact_proxy.py',ROOT/'src/fr3_description/models/fr3_hand.xml',
        ROOT/'src/fr3_description/models/scene.xml',ROOT/'pixi.lock',Path(__file__).with_name('PROTOCOL.md')]
    hashes={str(f.relative_to(ROOT)):digest(f) for f in source_paths}
    if args.resume:
        previous=json.loads((args.resume/'manifest.json').read_text())
        if previous['source_sha256']!=hashes:raise ValueError('source changed; use a new run')
    run=RunArtifacts.create(root=ROOT,output_root=args.output,experiment_id='i001_aliasing',config_path=args.config,resume=args.resume)
    if not args.resume:
        for source in source_paths:
            dest=run.run_path/'source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        run.manifest.update(source_sha256=hashes,mujoco_version=mujoco.__version__,numpy_version=np.__version__,
            asset_sha256={str(f.relative_to(ROOT)):digest(f) for f in sorted((ROOT/'src/fr3_description/models/assets').iterdir()) if f.is_file()})
        run._write_manifest()
    directory=run.run_path/'episodes';directory.mkdir(exist_ok=True)
    jobs=[]
    for split,key in [('train','train_seeds'),('validation','validation_seeds'),('test','test_seeds')]:
        for seed in range(*config[key]):
            for bin_id in range(len(config['friction_bins'])):
                for probe in [False,True]:jobs.append((seed,bin_id,probe,split,config,str(directory)))
    print(f'RUN={run.run_path}',flush=True)
    results=[]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for meta in pool.map(simulate,jobs):
                results.append(meta);run.append_jsonl('episodes.jsonl',meta)
                if len(results)%12==0:print(f'{len(results)}/{len(jobs)} complete; eligible={sum(x["eligible"] for x in results)}',flush=True)
        # Same-seed future controls and forces MUST match across all interventions.
        for seed in {r['seed'] for r in results}:
            assert len({r['future_input_sha256'] for r in results if r['seed']==seed})==1
        run.manifest['episode_sha256']={f.name:digest(f) for f in sorted(directory.glob('*.npz'))};run._write_manifest()
        run.finalize(dict(episodes=len(results),eligible=sum(x['eligible'] for x in results),future_inputs_matched=True),success=True)
    except BaseException as exc:
        run.finalize(dict(error=str(exc),completed=len(results)),success=False);raise
    print(f'COLLECTION_COMPLETE={run.run_path}',flush=True)


if __name__=='__main__':main()
