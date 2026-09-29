import numpy as np
import pytest
from experiments.iros2027.i001.evaluate import features, metrics, squared_distance, bootstrap_delta


def row(label=0):
    return dict(ticks=np.arange(0,301,5),causal=np.arange(0,301,5)<=250,
                tactile=np.arange(61*6).reshape(61,6).astype(float),
                proprio=np.zeros((61,18)),label=label)


def test_features_cannot_use_future_or_privileged_fields():
    a=row();b=row();b['tactile'][~b['causal']]=1e9
    b.update(label=1,friction=100,labels='forbidden',seed=999)
    for ms in [0,20,50,100,200,250]:
        np.testing.assert_equal(features([a],ms),features([b],ms))
    assert features([a],250).shape==(1,306)


def test_repeat_current_does_not_change_kernel_capacity_or_distance():
    a=row();b=row();b['tactile']+=10
    short=features([a,b],0);long=features([a,b],250,kind='repeat')
    np.testing.assert_allclose(squared_distance(short,short),squared_distance(long,long))


def test_metrics_with_ties_and_exact_predictions():
    m=metrics([0,1,0,1],[.5,.5,.5,.5])
    assert m['auroc']==.5 and m['average_precision']==.5 and m['brier']==.25
    assert metrics([0,1],[0,1])['balanced_accuracy']==1


def test_cluster_bootstrap_uses_seeds_and_correct_sign():
    cfg=dict(bootstrap_seed=1,bootstrap_samples=1000)
    y=np.array([0,1,0,1]);seeds=np.array([2,2,3,3])
    out=bootstrap_delta(y,np.full(4,.5),y,seeds,cfg)
    assert out['independent_seeds']==2
    assert out['ci95']==[.25,.25]


@pytest.mark.integration
def test_slip_requires_five_contact_ticks_and_future_input_matches(tmp_path):
    import yaml
    from experiments.iros2027.i001.collect import consecutive_slip, simulate, ROOT
    assert not consecutive_slip([.02]*4,[1]*4)
    assert not consecutive_slip([.02]*5,[1,1,0,1,1])
    assert consecutive_slip([.02]*5,[1]*5)
    cfg=yaml.safe_load((ROOT/'experiments/iros2027/i001/config.yaml').read_text())
    a=simulate((10,0,True,'development',cfg,str(tmp_path)))
    b=simulate((10,2,True,'development',cfg,str(tmp_path)))
    assert a['eligible'] and b['eligible']
    assert a['future_input_sha256']==b['future_input_sha256']
    assert a['label']==1 and b['label']==0
    with np.load(tmp_path/a['episode']) as data:
        assert data['ticks'][-1]==2700 and data['ticks'][0]==2400
        assert data['labels'].shape[0]==301


def test_mismatched_history_is_random_cross_seed_and_keeps_current():
    from experiments.iros2027.i001.evaluate import mismatched_donors
    seeds=np.repeat(np.arange(100,132),3)
    donors=mismatched_donors(seeds)
    assert np.all(seeds[donors]!=seeds)
    assert len(np.unique(donors))==len(seeds)
    # Catch the old cyclic-shift confound that preserved every friction bin.
    assert np.mean(donors%3==np.arange(len(seeds))%3)<.6
    rows=[]
    for i,seed in enumerate(seeds):
        r=row();r['seed']=int(seed);r['tactile'][:]=i;rows.append(r)
    x=features(rows,250,kind='mismatch').reshape(len(rows),51,6)
    np.testing.assert_equal(x[:,-1,0],np.arange(len(rows)))
    np.testing.assert_equal(x[:,0,0],donors)
