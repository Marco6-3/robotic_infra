"""Reaggregate exported metrics in temporary copies; never overwrite frozen reports."""
from pathlib import Path
import tempfile,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,yaml
from experiments.active_tactile_insertion.analysis import analyze as insertion
from experiments.tactile_history_control.analysis import analyze as grasp

def equal(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 elif isinstance(a,(int,float)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
 else:assert a==b,(a,b)
for group,func in [('active-tactile-insertion',insertion),('tactile-history-control',grasp)]:
 run=next((ROOT / 'docs/research-results' / group).glob('*-pilot'))
 expected=json.loads((run/'summary.json').read_text())
 with tempfile.TemporaryDirectory(prefix='pilot-public-reaggregate-') as tmp:
  dst=Path(tmp)
  for rel in json.loads((run/'EXPORT_MANIFEST.json').read_text())['files']:
   target=dst/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(run/rel,target)
  equal(expected,func(dst,yaml.safe_load((dst/'config.yaml').read_text())))
 print(group,': published-only metrics, bootstrap contrasts and decisions match original summary')
