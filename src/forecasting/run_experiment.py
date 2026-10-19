"""Run two CPU horizon workers at a time and preserve logs; resume completed fits."""
# The source tree is separate from input data and generated artifacts.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.dont_write_bytecode = True
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if p.name == 'src')))
from _pipeline_paths import material_source, code_directory
from pathlib import Path
import json, os, subprocess, sys, time
from datetime import datetime, timezone
ROOT=material_source(__file__).parent
def timestamp():return datetime.now(timezone.utc).isoformat()
def state(value):
    p=ROOT/'execution.json';q=p.with_suffix('.pending')
    q.write_text(json.dumps(value,indent=2),encoding='utf-8');q.replace(p)
def main():
    assert json.loads((ROOT/'qa/quality_unit_checks.json').read_text())['passed']
    assert json.loads((ROOT/'qa/preparation_checks.json').read_text())['passed']
    waiting=[1,6,12];active={};completed=[];started=timestamp()
    state({'stage':'running','started_at':started,'horizons':waiting,'workers':2})
    while waiting or active:
        while waiting and len(active)<2:
            h=waiting.pop(0);run=ROOT/'runs'/f'independent_h{h}'
            done=run/'execution.json'
            if done.exists() and json.loads(done.read_text())['stage']=='completed':
                completed.append(h);continue
            env=os.environ.copy();env.update(PM10_EXPERIMENT_DIR=str(run),OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',PYTHONUTF8='1')
            log=(ROOT/'logs'/f'training_h{h}.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-X','utf8',str(code_directory(ROOT)/'code/run_worker.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            active[h]=(p,log);print(f'Started horizon +{h}, pid {p.pid}',flush=True)
        for h,(p,log) in list(active.items()):
            rc=p.poll()
            if rc is None:continue
            log.close();del active[h]
            if rc!=0:
                state({'stage':'failed','failed_horizon':h,'exit_code':rc,'completed':completed,'started_at':started,'at':timestamp()})
                # Existing independent workers may finish; do not erase their work.
                for proc,stream in active.values():proc.wait();stream.close()
                raise SystemExit(rc)
            completed.append(h);print(f'Completed horizon +{h}',flush=True)
        if waiting or active:time.sleep(5)
    state({'stage':'completed','started_at':started,'completed_at':timestamp(),'horizons':completed})
    print('All three horizons trained and evaluated.',flush=True)
if __name__=='__main__':main()
