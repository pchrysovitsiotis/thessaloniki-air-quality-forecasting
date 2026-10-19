"""Rule behaviour and future-independence checks before expensive model fitting."""
# The source tree is separate from input data and generated artifacts.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.dont_write_bytecode = True
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if p.name == 'src')))
from _pipeline_paths import material_source, code_directory
from pathlib import Path
import json
import numpy as np
import pandas as pd
from quality import VARIABLES, hard_screen, fit_parameters, statistical_screen, fill_inputs

ROOT=material_source(__file__).parents[1]
cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))['qc']

def frame(n=100):
    x=pd.DataFrame(10.,index=pd.date_range('2020-01-01',periods=n,freq='h',tz='UTC'),columns=VARIABLES)
    x['humidity']=50.;x['pressure']=1000.;return x

def run():
    checks=[]
    x=frame();x.loc[x.index[:30],'PM10']=0
    c,r=hard_screen(x,cfg)
    assert c.PM10.iloc[:23].eq(0).all() and c.PM10.iloc[23:30].isna().all()
    checks.append('particulate zero exclusion starts at 24th hour without retrospective masking')
    x=frame();x.loc[:,'CO']=0
    c,r=hard_screen(x,cfg);assert c.CO.eq(0).all() and r.CO.iloc[-1]&8
    checks.append('isolated gaseous zeros retained and flagged')
    x=frame();x.loc[:,['PM10','NO2','O3']]=0
    c,r=hard_screen(x,cfg);assert c.PM10.iloc[:5].eq(0).all() and c.PM10.iloc[5:].isna().all()
    checks.append('shared three-pollutant zero rule starts at sixth hour')
    x=frame();c,r=hard_screen(x,cfg)
    assert c.PM10.eq(10).all() and r.PM10.iloc[11]&16 and not r.PM10.iloc[10]&16
    checks.append('positive flatline flagged without deletion')
    x=frame();x.iloc[40,x.columns.get_loc('PM10')]=500
    c,r=hard_screen(x,cfg);params=fit_parameters(c,c.index[30])
    flagged=statistical_screen(c,r,params,cfg)
    assert flagged.PM10.iloc[40]&32 and c.PM10.iloc[40]==500
    for end in [5,23,24,41,64]:
        prefix,pr=hard_screen(x.iloc[:end],cfg)
        pd.testing.assert_frame_equal(prefix,c.iloc[:end])
        pd.testing.assert_frame_equal(statistical_screen(prefix,pr,params,cfg),flagged.iloc[:end])
    checks.append('future values do not alter earlier hard rules or statistical flags')
    x=frame();x.loc[x.index[2:7],'PM10']=np.nan
    c,r=hard_screen(x,cfg)
    filled=fill_inputs(c,pd.Series(20.,index=VARIABLES),pd.Series(30.,index=VARIABLES),2)
    np.testing.assert_array_equal(filled.PM10.iloc[2:7],[10,10,20,20,20])
    assert c.PM10.iloc[2:7].isna().all()
    checks.append('two-hour causal input fill and unchanged missing target values')
    x=frame();x.iloc[0,x.columns.get_loc('PM10')]=-1;x.iloc[1,x.columns.get_loc('humidity')]=101
    c,r=hard_screen(x,cfg);assert pd.isna(c.PM10.iloc[0]) and pd.isna(c.humidity.iloc[1])
    checks.append('elementary physical-domain exclusions')
    output={'passed':True,'checks':checks}
    (ROOT/'qa/quality_unit_checks.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(output)

if __name__=='__main__':run()
