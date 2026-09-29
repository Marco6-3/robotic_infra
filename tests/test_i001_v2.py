from pathlib import Path
import numpy as np
import pytest,yaml
from experiments.iros2027.i001_v2.data import features,current_matches,donor_indices
from experiments.iros2027.i001_v2.physics import resolution,probe_width,finite_observation
from experiments.iros2027.i001_v2.analysis import metrics,clustered


def config():return yaml.safe_load((Path(__file__).resolve().parents[1]/'experiments/iros2027/i001_v2/config.yaml').read_text())


def test_action_is_required_by_b3_b4_and_not_b2():
    c=config();x=np.zeros((61,58));y=x.copy();y[:20,-1]=.001
    a=dict(finite=x,exact=x,seed=1,variant=0,label=0,friction=999)
    b=dict(finite=y,exact=y,seed=2,variant=0,label=1,friction=-999)
    np.testing.assert_equal(features([a],c,'finite','B2'),features([b],c,'finite','B2'))
    np.testing.assert_equal(features([a],c,'finite','B1'),features([b],c,'finite','B1'))
    for n in ['B3','B4']:assert not np.array_equal(features([a],c,'finite',n),features([b],c,'finite',n))


def test_pair_matching_accepts_only_current_arrays_and_equal_actions():
    c=config();x=np.zeros((3,58));x[1,0]=resolution(c)[0];x[2,-1]=1e-9
    assert [(i,j) for i,j,_,_ in current_matches(x,c)]==[(0,1)]


def test_observation_model_and_probe_return_to_identical_nominal():
    c=config();r=resolution(c);x=np.arange(58)*r/2
    y=finite_observation(x,c,np.random.default_rng(1));np.testing.assert_allclose(y/r,np.round(y/r),atol=1e-10)
    assert probe_width(2600,2400,70,.001,-1,.032)==.032
    assert probe_width(2430,2400,70,.001,-1,.032)<.032
    assert probe_width(2430,2400,70,.001,1,.032)>.032


def test_negative_control_donors_and_seed_bootstrap():
    s=np.repeat(np.arange(10),3);idx=donor_indices(s);assert np.all(s[idx]!=s)
    out=clustered([.2,.2,.2],[1,1,2],config());np.testing.assert_allclose(out['ci95'],[.2,.2]);assert out['seeds']==2
    m=metrics([0,1],[.5,.5]);assert m['auroc']==.5 and m['brier']==.25


@pytest.mark.integration
def test_real_active_probe_future_identity_and_full_recording(tmp_path):
    from experiments.iros2027.i001_v2.physics import simulate
    from experiments.iros2027.i001_v2.run import validate_episode
    from experiments.iros2027.i001_v2.data import read_episode
    c=config();a=simulate((800000,0,'development',c,str(tmp_path)));b=simulate((800000,1,'development',c,str(tmp_path)))
    assert a['future_hash']==b['future_hash'] and a['mass_kg']==b['mass_kg']
    assert a['action_change']>0 and b['action_change']>0
    validate_episode(tmp_path/a['episode'],c)
    r=read_episode(tmp_path/a['episode'],c);assert r['finite'].shape==(61,58)
    with np.load(tmp_path/a['episode']) as z:
        assert z['controls'].shape==(2701,27)
        assert z['ticks_ns'][0]==0 and z['ticks_ns'][-1]==2700000000
        assert np.max(z['exact'][:,:32])>0


def test_all_baselines_use_same_compute_dimensions_and_no_privileged_inputs():
    c=config();rng=np.random.default_rng(5);rows=[]
    for i in range(8):
        x=rng.normal(size=(61,58));rows.append(dict(exact=x,finite=x,seed=i//2,variant=i%2,label=i%2,friction=i,anchor_truth=np.ones(27)*i))
    from experiments.iros2027.i001_v2.model import BASELINES
    for name in BASELINES:
        assert features(rows,c,'finite',name).shape==(8,61*58)
    before=features(rows,c,'finite','B4')
    for r in rows:r.update(label=999,friction=999,anchor_truth='FORBIDDEN',future_label=999)
    np.testing.assert_array_equal(before,features(rows,c,'finite','B4'))
    # Full-current repetition carries no earlier observations.
    expected=np.repeat(rows[0]['finite'][-1:],61,axis=0).ravel()
    np.testing.assert_array_equal(features(rows,c,'finite','B1')[0],expected)
