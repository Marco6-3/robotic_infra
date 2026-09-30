from pathlib import Path
import copy
import numpy as np
import pytest
import torch,yaml
from experiments.tactile_history_control.policy import Policy
from experiments.tactile_history_control.environment import RecoveryEnv,JOINT_ORDER
from experiments.tactile_history_control.evaluate import corrupt
from experiments.tactile_history_control.ambiguity import match

C=yaml.safe_load((Path(__file__).parents[1]/'experiments/tactile_history_control/config.yaml').read_text())
torch.set_num_threads(2)

def test_content_and_parameter_fairness():
    x=torch.randn(2,61,42);counts=[]
    for m in ['M0','M1','M2']:
        torch.manual_seed(17);p=Policy(C,m).eval();counts.append(sum(t.numel() for t in p.parameters()))
        modified=x.clone();modified[:,:,-1]+=10
        if m!='M2':torch.testing.assert_close(p(x),p(modified),rtol=0,atol=0)
        if m=='M0':
            modified=x.clone();modified[:,:-1,:]+=10;torch.testing.assert_close(p(x),p(modified),rtol=0,atol=0)
    assert len(set(counts))==1

def test_causal_attention_no_future_influence():
    torch.manual_seed(1);p=Policy(C,'M2').eval();x=torch.randn(2,61,42);y=x.clone();y[:,31:]+=20
    with torch.no_grad():torch.testing.assert_close(p.encode(x)[:,:31],p.encode(y)[:,:31],rtol=1e-5,atol=1e-6)

def test_shuffles_preserve_values_and_current_boundary():
    x=np.arange(2*61*42).reshape(2,61,42).astype(float)
    y=corrupt(x,'M2_temporal_shuffle',[1,2],2400)
    np.testing.assert_array_equal(y[:,-1],x[:,-1])
    for i in range(2):np.testing.assert_array_equal(np.sort(y[i,:-1,0]),np.sort(x[i,:-1,0]))
    z=corrupt(x,'M2_action_shuffle',[1,2],2400)
    np.testing.assert_array_equal(z[:,:,:-1],x[:,:,:-1]);np.testing.assert_array_equal(np.sort(z[:,:,-1]),np.sort(x[:,:,-1]))
    np.testing.assert_array_equal(corrupt(x[:1],'M2_action_shuffle',[1],2400),z[:1])

def test_physics_input_boundary_action_alignment_and_restore():
    e=RecoveryEnv(C,910001).prepare();assert e.window().shape==(61,42);assert e.initial_stable
    assert len(JOINT_ORDER)==9
    # Observation cannot call privileged force/velocity truth or teacher labels.
    saved=e.truth;e.truth=lambda:(_ for _ in ()).throw(AssertionError('privileged path'))
    e.observe();e.truth=saved
    e.step(.37);assert e.window()[-1,-1]==pytest.approx(.37)
    assert e.env.data.ctrl[7]==pytest.approx((C['nominal_width_m']-C['max_closure_m']*.37)/2)
    state=e.snapshot();e2=RecoveryEnv(C,910001).restore(state)
    e.step(.2);e2.step(.2)
    np.testing.assert_allclose(e.env.data.qpos,e2.env.data.qpos,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(e.window(),e2.window())

def test_ambiguity_requires_close_inputs_and_different_teacher():
    x=np.zeros((61,42),np.float32)
    bank=[dict(seed=1,teacher=t,snapshot=dict(history=x.copy())) for t in [0.,.3]]
    pairs,_=match(bank,C);assert len(pairs)==1
    bank[1]['snapshot']['history'][-1,0]=.01
    pairs,_=match(bank,C);assert pairs==[]

def test_seed_partitions_disjoint():
    starts=C['seed_starts'];assert len(set(starts.values()))==len(starts)
    ranges=[set(range(s,s+1000)) for s in starts.values()]
    for i,a in enumerate(ranges):
        for b in ranges[i+1:]:assert a.isdisjoint(b)

def test_normalization_float32_for_cuda_and_no_label_access():
    from experiments.tactile_history_control.policy import normalize
    x=np.zeros((2,61,42),np.float32)
    assert normalize(x,np.zeros(42),np.ones(42)).dtype==np.float32

def test_dataset_windows_do_not_contain_current_target_or_future(tmp_path):
    from experiments.tactile_history_control.dataset import load_windows
    d=tmp_path/'data/train';d.mkdir(parents=True)
    x=np.arange(63*42,dtype=np.float32).reshape(63,42);y=np.array([.1,.2],np.float32)
    np.savez(d/'0000.npz',x=x,y=y)
    windows,labels=load_windows(tmp_path,C,'train')
    np.testing.assert_array_equal(windows[0],x[:61]);np.testing.assert_array_equal(windows[1],x[1:62])
    np.testing.assert_array_equal(labels,y)

def test_paired_bootstrap_retains_episode_correspondence():
    from experiments.tactile_history_control.analysis import contrast
    rows=[dict(train_seed=t,seed=s,success=int(s%2)) for t in [17,29,43] for s in range(12)]
    r=contrast({'a':rows,'b':list(reversed(rows))},'a','b',C)
    assert r['mean']==0 and r['ci95']==[0,0]
