"""Before model training: verify task affordances and safety, never use test seeds."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse,json
import numpy as np,yaml
from .environment import InsertionEnv
from .evaluate import simple_action
from .diagnostics import build_pair,pair_metrics
from .utils import atomic


def one(job):
    c,seed,mode=job;e=InsertionEnv(c,seed,force_scale=8 if mode=='brute_down' else 1).prepare()
    while not e.done():e.step(simple_action(e,mode))
    return dict(seed=seed,mode=mode,**e.metrics())


def run(c,path,n=16):
    with ProcessPoolExecutor(c['workers']) as p:rows=list(p.map(one,[(c,c['seed_starts']['development']+i,m) for m in ['teacher','nominal','constant_x','spiral','brute_down'] for i in range(n)]))
    paired=[pair_metrics(*build_pair((c,c['seed_starts']['development']+1000+i)),c) for i in range(8)]
    result=dict(config=c,rollouts=rows,mirror_diagnostics=paired)
    atomic(path,result)
    for m in ['teacher','nominal','constant_x','spiral','brute_down']:
        subset=[r for r in rows if r['mode']==m]
        print(m,'success',np.mean([r['success'] for r in subset]),'forcefailure',np.mean([r['force_failure'] for r in subset]),'prefix_failure',sum(r['prefix_force_failure'] for r in subset),'prefix_success',sum(r['prefix_success'] for r in subset),flush=True)
    print('T/q matched mirror pairs',sum(p['matched'] for p in paired),'/',len(paired),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(yaml.safe_load(Path(__file__).with_name('config.yaml').read_text()),a.output)
