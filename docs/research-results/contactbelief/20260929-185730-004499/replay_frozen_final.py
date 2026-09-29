"""Regenerate the recorded FINAL physics seeds in a NEW output directory.
Usage: .venv-recording/bin/python /absolute/path/to/this_file.py /new/output/path
No training, hyperparameter selection, or writes to the original run.
"""
from pathlib import Path
import json,shutil,sys
ORIGIN=Path(__file__).resolve().parent
ROOT=ORIGIN.parents[2]
sys.path.insert(0,str(ROOT))
import yaml
from experiments.iros2027.contactbelief.utils import check_frozen,atomic,now
from experiments.iros2027.contactbelief.final_test import collect
from experiments.iros2027.contactbelief.evaluate import evaluate
from experiments.iros2027.contactbelief.representation_analysis import analyze
from experiments.iros2027.contactbelief.report import report

def main():
 if len(sys.argv)!=2:raise SystemExit('Provide exactly one NEW output directory')
 destination=Path(sys.argv[1]).resolve();destination.mkdir(parents=True,exist_ok=False)
 frozen=check_frozen(ORIGIN);probes=json.loads((ORIGIN/'PROBES_FROZEN.json').read_text())
 paths=set(frozen['artifacts'])|set(probes['artifacts'])|{'config.yaml','FROZEN.json','PROBES_FROZEN.json','FINAL_SEEDS.json','source_data_audit.json'}
 for rel in paths:
  target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ORIGIN/rel,target)
 atomic(destination/'REPLAY_ORIGIN.json',dict(origin=str(ORIGIN),time=now(),mode='frozen-model full physical final-set replay'))
 config=yaml.safe_load((destination/'config.yaml').read_text())
 collect(destination,config);evaluate(destination,config);analyze(destination,config);verdict=report(destination,config)
 atomic(destination/'REPLAY_COMPLETE.json',dict(time=now(),gate=verdict))
 print(destination)
if __name__=='__main__':main()
