# The source tree is separate from input data and generated artifacts.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.dont_write_bytecode = True
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if p.name == 'src')))
from _pipeline_paths import material_source, code_directory
from pathlib import Path
import hashlib, json, os
import numpy as np
import pandas as pd

ROOT = Path(os.environ['PM10_EXPERIMENT_DIR']).resolve()
EXPERIMENT = ROOT.parents[1]
CONFIG = json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
IDS = ['GR0018A','GR0019A','GR0020A','GR0046A','GR0463A']
NAMES = {'GR0018A':'Αγία Σοφία','GR0019A':'Καλαμαριά','GR0020A':'Κορδελιό','GR0046A':'Σίνδος','GR0463A':'GR0463A'}
POLLUTANTS = ['PM10','NO2','O3','CO','PM2.5']
WEATHER = ['temperature','humidity','rain','pressure','wind_speed_10m','wind_speed_100m']
VARIABLES = POLLUTANTS+WEATHER
MODELS = ['TCN','ANN','XGBoost','TCN_XGBoost','ANN_XGBoost','TCN_ANN_XGBoost','TCN_ANN']
HYBRIDS = {'TCN_XGBoost':['TCN','XGBoost'],'ANN_XGBoost':['ANN','XGBoost'],'TCN_ANN_XGBoost':['TCN','ANN','XGBoost'],'TCN_ANN':['TCN','ANN']}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')
    os.replace(pending,path)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def status(stage,**kw):
    write(ROOT/'status.json',dict(stage=stage,at=str(pd.Timestamp.now(tz='UTC')),**kw))
    print(stage,kw,flush=True)
def fingerprint():
    source=code_directory(EXPERIMENT)
    payload={str(p.relative_to(source)):digest(p) for p in (source/'code').rglob('*.py')}
    payload['_pipeline_paths.py']=digest(Path(__import__('_pipeline_paths').__file__))
    for p in [EXPERIMENT/'protocol.json',ROOT/'config.json']:
        payload[str(p.relative_to(EXPERIMENT))]=digest(p)
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload
def source_inventory():
    paths=list((EXPERIMENT/'data/source').glob('*'))+list((ROOT/'data/stations').glob('*.csv'))
    return {str(p.relative_to(EXPERIMENT)):digest(p) for p in paths if p.is_file()}
