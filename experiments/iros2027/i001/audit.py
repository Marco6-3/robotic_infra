"""Independent checks on saved timestamps, labels, inputs and repeatability."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import yaml
from experiments.iros2027.i001.collect import simulate


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    run=args.run;cfg=yaml.safe_load((run/'config.yaml').read_text());anchor=cfg['anchor_tick']
    manifest=json.loads((run/'manifest.json').read_text());futures={};checked=0
    for path in sorted((run/'episodes').glob('*.npz')):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['episode_sha256'][path.name]
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            lab=z['labels'];future=lab[lab[:,0]>anchor]
            assert len(future)==cfg['horizon_ticks']
            assert np.all(np.diff(lab[:,0])==1)
            assert np.all(np.diff(z['ticks'])==cfg['sensor_period_ticks'])
            active=(future[:,1]>.01)&(future[:,2]>0)
            label=bool(np.any(np.convolve(active.astype(int),np.ones(5,dtype=int),mode='valid')==5))
            assert label==meta['label']
            futures.setdefault(meta['seed'],[]).append(z['future_inputs'].copy())
            checked+=1
    for group in futures.values():
        for array in group[1:]:np.testing.assert_array_equal(array,group[0])
    # Deterministically rerun one held-out witness without using its future to tune anything.
    name='test-324-friction0-probe1.npz'
    with tempfile.TemporaryDirectory() as directory:
        meta=simulate((324,0,True,'test',cfg,directory))
        with np.load(run/'episodes'/name) as a,np.load(Path(directory)/name) as b:
            for key in a.files:
                if key!='metadata':np.testing.assert_array_equal(a[key],b[key])
    result=dict(episodes_checked=checked,independent_seeds=len(futures),labels_recomputed=True,
                timestamps_valid=True,identical_future_arrays=True,deterministic_replay=name)
    (run/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
