"""Reference training pipeline. For the dependency-free offline example, run demo.py."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parent
SRC=REPO/'src'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def check_inputs(data_dir):
    manifest=json.loads((REPO/'configs/forecasting/source_manifest.json').read_text(encoding='utf-8'))
    for relative,expected in manifest['source_files'].items():
        path=data_dir/Path(relative).relative_to('data')
        if not path.is_file():raise ValueError('Missing input: '+str(path)+'; see docs/DATA.md.')
        if digest(path)!=expected:raise ValueError('Input differs from the recorded thesis archive: '+str(path))
    metadata=data_dir/'metadata/PM10_metadata.csv'
    if not metadata.is_file():raise ValueError('Missing input: '+str(metadata))
    with metadata.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'Sampling Point Id','Air Quality Station Type','Operational Activity Begin'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Reporting metadata is missing required columns; see docs/DATA.md.')
        rows=list(reader)
    for sid in ['GR0018A','GR0019A','GR0020A','GR0046A']:
        if not any(sid in (row['Sampling Point Id'] or '') and row['Air Quality Station Type'] for row in rows):
            raise ValueError('Reporting metadata is missing station '+sid)
    return manifest

def initialize(destination,data_dir,include_raw):
    manifest=check_inputs(data_dir)
    protected=[REPO/'src',REPO/'configs',REPO/'examples',REPO/'results',REPO/'docs',REPO/'tests',data_dir]
    if destination==REPO or any(destination==p or destination.is_relative_to(p) or p.is_relative_to(destination) for p in protected):
        raise ValueError('Choose a new output folder separate from the source and input files.')
    if destination.exists() and any(destination.iterdir()):raise ValueError('The output folder must be empty.')
    for name in ['forecasting']+(['raw_comparison'] if include_raw else []):
        root=destination/name
        for part in ['data/source','data/prepared','cache','results','figures','qa','logs','runs']:
            (root/part).mkdir(parents=True,exist_ok=True)
        for filename in ['config.json','protocol.json']:
            shutil.copy2(REPO/'configs'/name/filename,root/filename)
        for relative in manifest['source_files']:
            source=data_dir/Path(relative).relative_to('data')
            shutil.copy2(source,root/'data/source'/source.name)
    (destination/'forecasting/source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    metadata=destination/'reports/data/PM10_metadata.csv'
    metadata.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(data_dir/'metadata/PM10_metadata.csv',metadata)

def execute(relative,env,arguments=()):
    subprocess.run([sys.executable,'-B','-X','utf8',str(SRC/relative),*arguments],
        cwd=REPO,env=env,check=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)

def aggregate(destination,include_raw):
    import pandas as pd
    for name,prefix in [('forecasting','independent')]+([('raw_comparison','raw')] if include_raw else []):
        groups={}
        for h in [1,6,12]:
            for p in (destination/name/f'runs/{prefix}_h{h}/results').glob('*.csv'):
                groups.setdefault(p.name,[]).append(pd.read_csv(p))
        for filename,frames in groups.items():
            pd.concat(frames,ignore_index=True).to_csv(destination/name/'results'/filename,index=False)
    if include_raw:
        for h in [1,6,12]:
            a=pd.read_parquet(destination/f'forecasting/runs/independent_h{h}/results/predictions_h{h}.parquet')
            b=pd.read_parquet(destination/f'raw_comparison/runs/raw_h{h}/results/predictions_h{h}.parquet')
            paired=a.merge(b,on=['station_id','input_time','target_time','split'],suffixes=('_processed','_raw'),validate='one_to_one')
            assert len(paired)==len(a)==len(b) and (paired.target_processed==paired.target_raw).all()
        a=pd.read_csv(destination/'forecasting/results/metrics.csv')
        b=pd.read_csv(destination/'raw_comparison/results/metrics.csv')
        table=a.merge(b,on=['horizon','scope','model'],suffixes=('_processed','_raw'),validate='one_to_one')
        for metric in ['RMSE','MAE','R2']:
            table[metric+'_processed_minus_raw']=table[metric+'_processed']-table[metric+'_raw']
        table.to_csv(destination/'raw_comparison/results/before_after.csv',index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=REPO/'data')
    p.add_argument('--check-inputs',action='store_true')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--with-raw',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--llm-mode',choices=['prepare','reports','experiment'],default='prepare')
    a=p.parse_args()
    data_dir=a.data_dir.resolve()
    try:
        if a.check_inputs:
            if a.output_dir or a.with_raw or a.prepare_only or a.llm_mode!='prepare':p.error('--check-inputs is used with --data-dir only.')
            check_inputs(data_dir);print('Recorded input archive verified.');return
        if a.output_dir is None:p.error('Choose --check-inputs or --output-dir. For the offline demo use python demo.py.')
        if a.prepare_only and a.llm_mode!='prepare':p.error('--prepare-only cannot generate reports.')
        if a.llm_mode!='prepare' and not os.environ.get('GEMINI_API_KEY','').strip():
            p.error('Set GEMINI_API_KEY before requesting new LLM reports.')
        destination=a.output_dir.resolve()
        initialize(destination,data_dir,a.with_raw)
    except ValueError as exc:p.error(str(exc))
    env=os.environ.copy()
    env.update(PM10_OUTPUT_DIR=str(destination),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    for script in ['prepare_data.py','validate_quality.py','validate_preparation.py']:
        execute('forecasting/code/'+script,env)
    if a.with_raw:execute('raw_comparison/code/prepare_raw.py',env)
    if a.prepare_only:print('Preparation completed; no training or API requests.');return
    execute('forecasting/run_experiment.py',env)
    if a.with_raw:execute('raw_comparison/run_experiment.py',env)
    aggregate(destination,a.with_raw)
    execute('reports/code/app_reports.py',env,{'prepare':['--prepare'],'reports':[],'experiment':['--experiment']}[a.llm_mode])
    print('Pipeline completed:',destination)

if __name__=='__main__':main()
