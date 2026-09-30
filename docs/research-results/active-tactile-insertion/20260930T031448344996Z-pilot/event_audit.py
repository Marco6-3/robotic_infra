"""Replay saved commands to verify 1ms success events versus 5ms log endpoints."""
from pathlib import Path
import json,numpy as np,yaml
from experiments.active_tactile_insertion.environment import InsertionEnv
ROOT=Path(__file__).resolve().parent;c=yaml.safe_load((ROOT/'config.yaml').read_text())
rows=json.loads((ROOT/'event-terminal-differences.json').read_text());out=[]
for row in rows:
    e=InsertionEnv(c,row['seed'],force_scale=8 if row['condition']=='brute_down' else 1).prepare()
    original=e.hole_force;trace=dict(last=None,start=None,events=[])
    def observed_force():
        force=original()
        if trace['last']!=e.tick:
            trace['last']=e.tick;pos=e.data.site_xpos[e.tip]
            depth=-pos[2];error=float(np.linalg.norm(pos[:2]-e.hole))
            good=depth>=c['insertion_depth_m'] and error<c['hole_radius_m']-c['peg_radius_m'] and force<=c['safe_force_n'] and not e.force_failure
            if good:
                if trace['start'] is None:trace['start']=e.tick
                if e.tick-trace['start']>=c['success_hold_ms'] and not trace['events']:
                    trace['events'].append(dict(tick=e.tick,depth_m=float(depth),lateral_error_m=error,force_n=force,held_ms=e.tick-trace['start']))
            else:trace['start']=None
        return force
    e.hole_force=observed_force
    z=np.load(ROOT/'traces'/f"{row['condition']}-{row['train_seed']}"/f"{row['seed']}.npz")
    for command in z['issued']:e.step(command)
    assert trace['events'] and e.success and not e.force_failure and e.peak_force<=c['safe_force_n']
    assert abs(e.metrics()['insertion_depth_m']-row['insertion_depth_m'])<1e-7
    event=trace['events'][0];assert 0<=e.tick-event['tick']<=4
    out.append(dict(condition=row['condition'],train_seed=row['train_seed'],seed=row['seed'],event=event,terminal_tick=e.tick,terminal_lateral_error_m=e.metrics()['lateral_error_m'],verified=True))
(ROOT/'event-audit.json').write_text(json.dumps(out,indent=2))
print('Verified first-hit depth/alignment/20ms hold/force for all',len(out),'event-versus-endpoint cases; no labels changed')
