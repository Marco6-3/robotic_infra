"""One-command resumable collect -> validate -> freeze -> evaluate -> report."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib,json,os,platform,shlex,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import yaml,mujoco
from .physics import ROOT,simulate,slip_label
from .data import load_rows,read_episode,stats,features
from .model import BASELINES,train
from .analysis import evaluate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def shell(*args):
    r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)

def sources():
    roots=[ROOT/'experiments/iros2027/i001_v2',ROOT/'experiments/iros2027/contact',ROOT/'experiments/iros2027/common',ROOT/'experiments/iros2027/i002',ROOT/'src/fr3_sim/fr3_sim',ROOT/'src/fr3_control/fr3_control',ROOT/'src/fr3_robot_api/fr3_robot_api']
    paths=[]
    for directory in roots:
        paths.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.md'])
    paths += [ROOT/'pixi.lock',ROOT/'pixi.toml',ROOT/'src/fr3_description/models/fr3_hand.xml',ROOT/'src/fr3_description/models/scene.xml']
    return sorted(set(paths))


def validate_config(c):
    assert c['train_seed_count']*c['episodes_per_seed']+c['validation_seed_count']*c['episodes_per_seed']<c['max_episodes']
    assert c['history_ms']%c['sample_ms']==0 and c['anchor_tick']%c['sample_ms']==0
    assert max(c['probe_start_before_anchor_ms'])<=c['history_ms']
    assert min(c['probe_start_before_anchor_ms'])>max(c['probe_duration_ms'])
    spans=[set(range(c['seed_starts'][s],c['seed_starts'][s]+c[s+'_seed_count'])) for s in ['train','validation','test']]
    assert all(not spans[i]&spans[j] for i in range(3) for j in range(i))
    assert c['history_ms']>0 and c['future_ms']>=5


def initialize(args,c):
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources()}
    if args.resume:
        run=args.resume.resolve();manifest=json.loads((run/'manifest.json').read_text())
        if manifest['source_sha256']!=hashes or manifest['config_sha256']!=sha(args.config):raise ValueError('resume rejected: source/config drift')
    else:
        run=args.output.resolve()/datetime.now().strftime('%Y%m%d-%H%M%S-%f');run.mkdir(parents=True)
        shutil.copy2(args.config,run/'config.yaml')
        manifest=dict(created=datetime.now(timezone.utc).isoformat(),git_commit=shell('git','rev-parse','HEAD')['stdout'].strip(),git_status=shell('git','status','--short')['stdout'],
            source_sha256=hashes,config_sha256=sha(args.config),environment=dict(python=sys.version,platform=platform.platform(),mujoco=mujoco.__version__,numpy=np.__version__,
                device='CPU; NumPy linear algebra',cuda='not used by this model',batch_size='full ridge fit; inference 256',workers=c['workers'],
                gpu=shell('nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'),
                blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),omp_threads=os.environ.get('OMP_NUM_THREADS')),
            seeds=c['seed_starts'],command=shlex.join([sys.executable,'-m','experiments.iros2027.i001_v2.run',*sys.argv[1:]]),episode_sha256={})
        (run/'git.diff').write_text(shell('git','diff','--binary')['stdout'])
        for source in sources():
            dest=run/'source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        manifest['asset_sha256']={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/fr3_description/models/assets').iterdir() if p.is_file()}
        atomic(run/'manifest.json',manifest)
        (run/'REPRODUCE.txt').write_text('cd '+shlex.quote(str(ROOT))+'\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run i001-v2 --config '+shlex.quote(str((run/'config.yaml').resolve()))+'\n\nResume (same source/config):\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run i001-v2 --resume '+shlex.quote(str(run))+' --config '+shlex.quote(str((run/'config.yaml').resolve()))+'\n')
    # Exclusive lock is released automatically on crash; no stale lock PID logic.
    import fcntl
    lock=(run/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return run,manifest,lock


def progress(run,stage,statistics,**kwargs):
    data=dict(stage=stage,updated=datetime.now(timezone.utc).isoformat(),statistics=statistics,**kwargs)
    atomic(run/'progress.json',data)
    with (run/'events.jsonl').open('a') as f:f.write(json.dumps(data,ensure_ascii=False)+'\n')
    print(json.dumps(data,ensure_ascii=False),flush=True)


def validate_episode(path,c):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));ticks=z['ticks_ns'];lab=z['labels'];future=lab[lab[:,0]>c['anchor_tick']]
        assert len(future)==c['future_ms'] and np.all(np.diff(future[:,0])==1)
        assert np.all(np.diff(ticks)==c['sample_ms']*1000000) and np.array_equal(ticks,z['available_ns'])
        assert ticks[0]==0 and ticks[-1]==(c['anchor_tick']+c['future_ms'])*1000000
        assert np.all(np.isfinite(z['exact'])) and np.all(np.isfinite(z['finite']))
        assert int(slip_label(future[:,1],future[:,2]))==meta['label']
        assert z['controls'].shape[0]==c['anchor_tick']+c['future_ms']+1
        assert np.all(z['future_inputs'][:,-2]==c['nominal_width_m'])
        assert meta['action_change']>0
    return meta


def collect_block(run,c,manifest,split,seeds,statistics):
    directory=run/'episodes'/split;directory.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for seed in seeds:
        for v in range(c['episodes_per_seed']):
            p=directory/f'{split}-{seed}-{v:02}.npz';rel=str(p.relative_to(run))
            if p.exists() and rel in manifest['episode_sha256']:
                if sha(p)!=manifest['episode_sha256'][rel]:raise ValueError('stored episode hash mismatch')
            jobs.append((seed,v,split,c,str(directory)))
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        for i,meta in enumerate(pool.map(simulate,jobs),1):
            path=directory/meta['episode'];validate_episode(path,c)
            manifest['episode_sha256'][str(path.relative_to(run))]=sha(path)
            if i%40==0:
                atomic(run/'manifest.json',manifest)
                # Lightweight crash-visible counter. Full pair statistics at each seed block.
                progress(run,'collect_'+split,statistics,block_completed=i,block_size=len(jobs),persisted_episodes=len(manifest['episode_sha256']))
    atomic(run/'manifest.json',manifest)


def fit_all(run,c):
    tr=[r for r in load_rows(run,c,'train') if r['eligible']];va=[r for r in load_rows(run,c,'validation') if r['eligible']]
    assert not set(r['seed'] for r in tr)&set(r['seed'] for r in va)
    assert len(set(r['label'] for r in tr))==2,'training labels degenerate'
    models=run/'models';models.mkdir(exist_ok=True);selections={}
    yt=np.array([r['label'] for r in tr]);yv=np.array([r['label'] for r in va])
    for regime in ['exact','finite']:
        for b in BASELINES:
            p=models/f'{regime}-{b}.npz'
            if not p.exists():
                target=np.random.default_rng(c['models']['random_seed']+90).permutation(yt) if b=='N3' else yt
                m=train(features(tr,c,regime,b),target,features(va,c,regime,b),yv,c)
                with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**m)
                p.with_suffix('.tmp').replace(p)
            with np.load(p) as z:selections[f'{regime}-{b}']=dict(length=float(z['length']),penalty=float(z['penalty']),validation_brier=float(z['validation_brier']),sha256=sha(p))
            print('MODEL',regime,b,json.dumps(selections[f'{regime}-{b}']),flush=True)
    frozen=dict(time=datetime.now(timezone.utc).isoformat(),config_sha256=sha(run/'config.yaml'),selections=selections,
        train_seeds=sorted({r['seed'] for r in tr}),validation_seeds=sorted({r['seed'] for r in va}),
        test_seed_range=[c['seed_starts']['test'],c['seed_starts']['test']+c['test_seed_count']],history_ms=c['history_ms'],
        episode_sha256={str(p.relative_to(run)):sha(p) for split in ['train','validation'] for p in (run/'episodes'/split).glob('*.npz')})
    atomic(run/'FROZEN.json',frozen)


def audit_and_replay(run,c):
    groups={};checked=0
    for split in ['train','validation','test']:
        for path in (run/'episodes'/split).glob('*.npz'):
            m=validate_episode(path,c);groups.setdefault(m['seed'],set()).add(m['future_hash']);checked+=1
    assert all(len(v)==1 for v in groups.values())
    # Exact seed replay in a new output directory, never overwriting collected evidence.
    seed=c['seed_starts']['test'];directory=run/'replay';directory.mkdir(exist_ok=True)
    m=simulate((seed,0,'test',c,str(directory)));name=m['episode']
    with np.load(run/'episodes/test'/name) as a,np.load(directory/name) as b:
        for k in a.files:
            if k!='metadata':np.testing.assert_array_equal(a[k],b[k])
    result=dict(episodes=checked,seeds=len(groups),future_inputs_identical=True,deterministic_replay=name,labels_timestamps_checked=True)
    atomic(run/'audit.json',result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.yaml'));p.add_argument('--output',type=Path,default=ROOT/'runs/i001-v2');p.add_argument('--resume',type=Path)
    args=p.parse_args();c=yaml.safe_load(args.config.read_text());validate_config(c)
    run,manifest,lock=initialize(args,c);print('RUN='+str(run),flush=True);statistics={}
    try:
        if (run/'COMPLETE.json').exists():print('Already complete; preserved existing outputs');return
        for split in ['train','validation']:
            if (run/'FROZEN.json').exists():
                statistics[split]=stats(load_rows(run,c,split),c);continue
            for offset in range(0,c[split+'_seed_count'],c['batch_seeds']):
                seeds=range(c['seed_starts'][split]+offset,c['seed_starts'][split]+min(offset+c['batch_seeds'],c[split+'_seed_count']))
                collect_block(run,c,manifest,split,seeds,statistics)
                statistics[split]=stats(load_rows(run,c,split),c);progress(run,'collect_'+split,statistics)
        if not (run/'FROZEN.json').exists():
            if (run/'episodes/test').exists() and any((run/'episodes/test').glob('*.npz')):raise RuntimeError('test exists before freeze')
            progress(run,'train',statistics);fit_all(run,c)
        frozen=json.loads((run/'FROZEN.json').read_text())
        for key,val in frozen['selections'].items():assert sha(run/'models'/f'{key}.npz')==val['sha256']
        for rel,value in frozen['episode_sha256'].items():assert sha(run/rel)==value
        # Models are frozen from here onward. Test collection adapts only to pair counts.
        budget_seeds=(c['max_episodes']-sum(statistics[s]['episodes'] for s in ['train','validation']))//c['episodes_per_seed']
        cap=min(c['test_seed_count'],budget_seeds)
        test=load_rows(run,c,'test') if (run/'episodes/test').exists() else []
        statistics['test']=stats(test,c)
        for offset in range(0,cap,c['batch_seeds']):
            st=statistics['test']
            if st['seeds']>=c['minimum_test_seeds'] and st['qualified_pairs']>=c['target_pairs'] and st['qualified_seed_coverage']>=c['target_pair_seeds']:break
            seeds=range(c['seed_starts']['test']+offset,c['seed_starts']['test']+min(offset+c['batch_seeds'],cap))
            collect_block(run,c,manifest,'test',seeds,statistics)
            test=load_rows(run,c,'test');statistics['test']=stats(test,c);progress(run,'collect_test',statistics)
        progress(run,'validate',statistics);audit_and_replay(run,c)
        progress(run,'evaluate_bootstrap_diagnose',statistics);evaluate(run,c)
        # Figures use the host's already-installed matplotlib; simulation remains Pixi-isolated.
        plot=shell('env','-u','PYTHONPATH','-u','PYTHONHOME','PYTHONNOUSERSITE=1','/usr/bin/python3',str(Path(__file__).with_name('plot.py')),str(run));atomic(run/'plot_log.json',plot)
        if plot['returncode']:raise RuntimeError('figure generation failed: '+plot['stderr'])
        progress(run,'regression',statistics)
        checks=shell(sys.executable,'-m','pytest','-q','-o','addopts=');atomic(run/'regression.json',checks)
        (run/'regression.log').write_text(checks['stdout']+checks['stderr'])
        if checks['returncode']:raise RuntimeError('regressions failed; see regression.log')
        progress(run,'complete',statistics)
        atomic(run/'COMPLETE.json',dict(finished=datetime.now(timezone.utc).isoformat(),verdict=json.loads((run/'verdict.json').read_text()),
            artifacts={str(f.relative_to(run)):sha(f) for f in run.iterdir() if f.is_file() and f.name not in ['.lock','manifest.json']}))
        print('COMPLETE='+str(run),flush=True)
    except BaseException as exc:
        progress(run,'failed',statistics,error=repr(exc));raise
    finally:lock.close()

if __name__=='__main__':main()
