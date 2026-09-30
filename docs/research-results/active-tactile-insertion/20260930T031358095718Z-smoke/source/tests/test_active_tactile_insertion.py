from pathlib import Path
import copy
import numpy as np
import pytest
import torch,yaml
from experiments.active_tactile_insertion.environment import InsertionEnv,JOINT_ORDER
from experiments.active_tactile_insertion.policy import Policy,normalize
from experiments.active_tactile_insertion.evaluate import corrupt
from experiments.active_tactile_insertion.diagnostics import pair_metrics,build_pair

C=yaml.safe_load((Path(__file__).parents[1]/'experiments/active_tactile_insertion/config.yaml').read_text())
torch.set_num_threads(2)


def test_identical_parameters_and_information_masks():
    counts=[];x=torch.randn(2,61,37)
    for m in ['M0','M1','M2']:
        torch.manual_seed(17);p=Policy(C,m).eval();counts.append(sum(v.numel() for v in p.parameters()))
        assert p(x).shape==(2,2)
        y=x.clone();y[...,-2:]+=50
        if m!='M2':torch.testing.assert_close(p(x),p(y),rtol=0,atol=0)
        if m=='M0':
            y=x.clone();y[:,:-1]+=5;torch.testing.assert_close(p(x),p(y),rtol=0,atol=0)
    assert len(set(counts))==1


def test_causal_mask():
    p=Policy(C,'M2').eval();x=torch.randn(2,61,37);y=x.clone();y[:,30:]+=100
    with torch.no_grad():torch.testing.assert_close(p.encode(x)[:,:30],p.encode(y)[:,:30],rtol=1e-5,atol=1e-6)


def test_real_teacher_succeeds_and_nominal_not_guaranteed():
    e=InsertionEnv(C,1710000).prepare();assert e.window().shape==(61,37);assert e.prefix_contact
    while not e.done():e.step(e.teacher())
    m=e.metrics();assert m['success'] and m['peak_force_n']<C['safe_force_n']
    assert m['insertion_depth_m']>=C['insertion_depth_m']
    assert not m['prefix_success'] and not m['prefix_force_failure']


def test_force_failure_monitored_at_physics_rate():
    c=copy.deepcopy(C);c['safe_force_n']=.01
    e=InsertionEnv(c,1710000).prepare();assert e.force_failure and not e.success and e.done()
    assert e.peak_force>c['safe_force_n']


def test_geometry_sensor_does_not_read_force_or_teacher_and_action_alignment():
    e=InsertionEnv(C,1710001).prepare();orig=e.hole_force
    e.hole_force=lambda:(_ for _ in ()).throw(AssertionError('privileged sensor read'))
    assert e.observe().shape==(37,);e.hole_force=orig
    e.step([.25,-.35]);np.testing.assert_allclose(e.window()[-1,-2:],[.25,-.35],atol=1e-7)
    assert JOINT_ORDER==('carriage_x','carriage_y','carriage_z')


def test_snapshot_restores_true_dynamics():
    e=InsertionEnv(C,1710001).prepare();s=e.snapshot();f=InsertionEnv(C,1710001).restore(s)
    a=e.teacher();e.step(a);f.step(a)
    np.testing.assert_allclose(e.data.qpos,f.data.qpos,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(e.window(),f.window());assert e.metrics()==f.metrics()


def test_shuffle_preserves_vector_pairing_and_current_observation():
    x=np.arange(61*37,dtype=float).reshape(1,61,37)
    a=corrupt(x,'M2_action_shuffle',[1],[650]);np.testing.assert_array_equal(a[:,:,:-2],x[:,:,:-2])
    assert set(map(tuple,a[0,:,-2:]))==set(map(tuple,x[0,:,-2:]))
    t=corrupt(x,'M2_temporal_shuffle',[1],[650]);np.testing.assert_array_equal(t[:,-1],x[:,-1])


def test_full_history_matching_cannot_ignore_q():
    a=dict(seed=1,teacher=np.array([1.,0.]),snapshot=dict(history=np.zeros((61,37))),metrics=dict(prefix_force_failure=0,prefix_success=0))
    b=copy.deepcopy(a);b['teacher']=-a['teacher']
    assert pair_metrics(a,b,C)['matched']
    b['snapshot']['history'][:,32]=.001
    r=pair_metrics(a,b,C);assert r['tactile_rms']==0 and not r['matched']


def test_normalization_dtype_and_dataset_window_boundary(tmp_path):
    from experiments.active_tactile_insertion.dataset import load_windows
    x=np.arange(63*37,dtype=np.float32).reshape(63,37);y=np.ones((2,2),np.float32)
    p=tmp_path/'data/train';p.mkdir(parents=True);np.savez(p/'x.npz',x=x,y=y)
    w,t=load_windows(tmp_path,C,'train');np.testing.assert_array_equal(w[0],x[:61]);np.testing.assert_array_equal(w[1],x[1:62])
    assert normalize(w,np.zeros(37),np.ones(37)).dtype==np.float32


def test_seed_partitions():
    starts=list(C['seed_starts'].values())
    for i,s in enumerate(starts):
        for t in starts[i+1:]:assert abs(t-s)>1000
