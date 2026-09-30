from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch,yaml

def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
    tmp.replace(path)

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_pt(path,obj):
    tmp=Path(path).with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def config(run):return yaml.safe_load((Path(run)/'config.yaml').read_text())
def setup(c):
    torch.set_num_threads(c['cpu_threads']);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=True
    if c['device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')

def progress(run,stage,**kw):
    record=dict(time=time.time(),stage=stage,**kw)
    with (Path(run)/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record),flush=True)
