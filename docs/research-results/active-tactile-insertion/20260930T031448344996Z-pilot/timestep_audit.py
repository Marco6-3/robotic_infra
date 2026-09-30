"""Development-seed timestep sensitivity; does not modify frozen experiment code."""
from pathlib import Path
import importlib.util,copy,json,yaml
ROOT=Path(__file__).resolve().parent
source=Path('experiments/active_tactile_insertion/environment.py').read_text()
# The production config uses integer milliseconds. Cast the substep COUNT solely
# in this audit copy to permit a half-millisecond simulator step.
audit_source=ROOT/'environment_timestep_audit.py'
audit_source.write_text(source.replace("range(c['sample_ms']//c['physics_ms'])","range(int(c['sample_ms']//c['physics_ms']))"))
spec=importlib.util.spec_from_file_location('step_audit',audit_source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
c=yaml.safe_load((ROOT/'config.yaml').read_text());rows=[]
for dt in [1,.5]:
 d=copy.deepcopy(c);d['physics_ms']=dt
 for seed in range(1711000,1711004):
  e=mod.InsertionEnv(d,seed).prepare()
  while not e.done():e.step(e.teacher())
  rows.append(dict(physics_ms=dt,seed=seed,**e.metrics()))
(ROOT/'timestep-audit.json').write_text(json.dumps(rows,indent=2))
print([(r['physics_ms'],r['seed'],r['success'],round(r['peak_force_n'],3)) for r in rows])
