"""Information boundaries and strict ablation invariants, using synthetic data."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')
import yaml
from experiments.iros2027.contactbelief.models import Model, future_loss, parameter_report
from experiments.iros2027.contactbelief.dataset import read_sample, fit_normalizer, transform
from experiments.iros2027.contactbelief.evaluate import weighted_metrics, bootstrap_weights
from experiments.iros2027.contactbelief.representation_analysis import ridge_fit,ridge_predict
from experiments.iros2027.contactbelief.train import fit_all

ROOT=Path(__file__).resolve().parents[1]
C=yaml.safe_load((ROOT/'experiments/iros2027/contactbelief/config.yaml').read_text())

def test_predictive_ablation_identical_initialization_and_inference():
    torch.manual_seed(1);a=Model('B2',C)
    torch.manual_seed(1);b=Model('B4',C)
    assert a.state_dict().keys()==b.state_dict().keys()
    for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    x=torch.randn(3,61,58);torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
    assert parameter_report(a)['outcome_inference_parameters']==parameter_report(b)['outcome_inference_parameters']
    z,_=b.encode(x);control=torch.randn(3,61,3,16);target=torch.randn(3,61,3,50)
    future_loss(b.predictive(z,control),target).backward()
    assert b.temporal.weight_hh_l0.grad.abs().sum()>0

@pytest.mark.parametrize('method',['B2','B3','B4'])
def test_future_suffix_cannot_change_past_latent(method):
    torch.manual_seed(2);model=Model(method,C).eval();x=torch.randn(2,61,58);other=x.clone();other[:,31:]=100*torch.randn_like(other[:,31:])
    with torch.no_grad():a,_=model.encode(x);b,_=model.encode(other)
    torch.testing.assert_close(a[:,:31],b[:,:31],rtol=1e-5,atol=1e-6)

def test_future_and_privileged_data_are_not_encoder_inputs(tmp_path):
    ticks=np.arange(0,2701,5);obs=np.random.default_rng(3).normal(size=(len(ticks),58))
    p=tmp_path/'sample.npz';meta=dict(eligible=True,label=0,seed=1,variant=0,friction=.5)
    def save(data,metadata):np.savez(p,exact=data,finite=data,ticks_ns=ticks*1000000,available_ns=ticks*1000000,metadata=json.dumps(metadata),labels=np.ones((4,27)))
    save(obs,meta);x,f,u,m=read_sample(p,C,'finite')
    changed=obs.copy();changed[ticks>2600,:50]+=10000;meta.update(label=1,friction=99)
    save(changed,meta);xx,ff,uu,mm=read_sample(p,C,'finite')
    np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(u,uu);assert not np.array_equal(f,ff)
    assert x.shape==(61,58) and f.shape==(61,3,50)

def test_cluster_metrics_preserve_ties_and_shared_seed():
    y=np.array([0,1,0,1]);p=np.array([.5,.5,.2,.8]);m=weighted_metrics(y,p,np.ones(4))[0]
    assert m[1]==pytest.approx(.875)
    weights=bootstrap_weights(np.array([1,1,2,2]),dict(bootstrap_samples=30,bootstrap_seed=1))
    np.testing.assert_array_equal(weights[:,0],weights[:,1]);np.testing.assert_array_equal(weights[:,2],weights[:,3])

def test_training_refuses_frozen_run(tmp_path):
    (tmp_path/'FROZEN.json').write_text('{}')
    with pytest.raises(RuntimeError,match='forbidden'):fit_all(tmp_path,C)

def test_ridge_probe_does_not_mutate_representation():
    rng=np.random.default_rng(4);x=rng.normal(size=(100,8));y=x[:,:2];old=x.copy()
    m=ridge_fit(x,y);p=ridge_predict(m,x)
    np.testing.assert_array_equal(x,old);assert np.mean((p-y)**2)<.001
