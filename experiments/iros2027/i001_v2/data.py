"""Label-blind current matching and explicitly allowlisted causal features."""
from pathlib import Path
import itertools,json
import numpy as np
from .physics import resolution


def read_episode(path,c):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));mask=(z['ticks_ns']<=c['anchor_tick']*1000000)&(z['ticks_ns']>=(c['anchor_tick']-c['history_ms'])*1000000)
        assert np.all(z['available_ns'][mask]<=c['anchor_tick']*1000000)
        r=dict(meta,exact=z['exact'][mask].copy(),finite=z['finite'][mask].copy(),
               anchor_truth=z['labels'][np.flatnonzero(z['labels'][:,0]==c['anchor_tick'])[0]].copy())
    return r


def load_rows(run,c,split):
    return [read_episode(p,c) for p in sorted((run/'episodes'/split).glob('*.npz'))]


def current_matches(current,c):
    """No labels, friction, pose, velocity truth or histories accepted here."""
    delta=np.abs(current[:,None]-current[None,:])/resolution(c)
    linf=delta.max(2);rms=np.sqrt((delta**2).mean(2))
    mask=(linf<=c['observation']['matching_linf']+1e-9)&(rms<c['observation']['matching_rms'])
    # Current command equality, beyond the distance threshold.
    mask &= (current[:,None,-8:]==current[None,:,-8:]).all(2)
    return [(int(i),int(j),float(linf[i,j]),float(rms[i,j])) for i,j in zip(*np.where(np.triu(mask,1)))]


def pair_rows(rows,c,regime):
    pairs=[]
    for seed in sorted({r['seed'] for r in rows}):
        group=[r for r in rows if r['seed']==seed and r['eligible']]
        if len(group)<2:continue
        # Physical group membership only enforces same geometry, mass, future inputs.
        assert len({r['future_hash'] for r in group})==1
        assert len({r['mass_kg'] for r in group})==1
        candidates=current_matches(np.stack([r[regime][-1] for r in group]),c)
        for i,j,linf,rms in candidates:
            a,b=group[i],group[j]
            # Label/privileged annotation happens AFTER matching has finished.
            pairs.append(dict(regime=regime,seed=seed,a=a['episode'],b=b['episode'],linf=linf,rms=rms,
                history_distance=float(np.sqrt(np.mean(((a[regime][:-1]-b[regime][:-1])/resolution(c))**2))),
                label_a=a['label'],label_b=b['label'],opposite=a['label']!=b['label'],
                friction_a=a['friction'],friction_b=b['friction'],fixed=a['fixed_friction'] and b['fixed_friction'],
                object_position_difference=float(np.linalg.norm(a['anchor_truth'][3:6]-b['anchor_truth'][3:6])),
                future_hash=a['future_hash']))
    return pairs


def stats(rows,c):
    pairs=pair_rows(rows,c,'finite') if rows else []
    opposite=[p for p in pairs if p['opposite']]
    return dict(episodes=len(rows),eligible=sum(r['eligible'] for r in rows),seeds=len({r['seed'] for r in rows}),
        positive=sum(r['label'] for r in rows),negative=sum(1-r['label'] for r in rows),
        matched_pairs=len(pairs),qualified_pairs=len(opposite),qualified_seed_coverage=len({p['seed'] for p in opposite}),
        fixed_friction_qualified_pairs=sum(p['fixed'] for p in opposite))


def donor_indices(seeds):
    # Independent random donors, with replacement, drawn only from other seeds.
    # Works even when a split has two unequal seed groups; never uses labels/bin order.
    if len(np.unique(seeds))<2:raise ValueError('need at least two physics seeds')
    rng=np.random.default_rng([731,int(seeds[0])]);donor=np.empty(len(seeds),dtype=int)
    for seed in np.unique(seeds):
        target=np.flatnonzero(seeds==seed);pool=np.flatnonzero(seeds!=seed)
        donor[target]=rng.choice(pool,len(target),replace=True)
    return donor


def features(rows,c,regime,baseline):
    """Allowlisted sensor history; same padded dimensions/compute for every model."""
    obs=np.stack([r[regime] for r in rows]);n=2*c['tactile']['grid']**2
    if baseline=='B0':
        obs=np.repeat(obs[:,-1:,:],obs.shape[1],axis=1);obs[:,:,n:]=0
    elif baseline=='B1':obs=np.repeat(obs[:,-1:,:],obs.shape[1],axis=1)
    elif baseline=='B2':obs[:,:,n:]=0
    elif baseline=='B3':obs[:,:,n:-8]=0
    elif baseline=='N1':
        for i,r in enumerate(rows):
            rng=np.random.default_rng([91,r['seed'],r['variant']]);obs[i,:-1]=obs[i,rng.permutation(obs.shape[1]-1)]
    elif baseline=='N2':obs[:,:-1]=obs[donor_indices(np.array([r['seed'] for r in rows])),:-1]
    elif baseline not in ['B4','N3']:raise ValueError(baseline)
    return obs.reshape(len(rows),-1)
