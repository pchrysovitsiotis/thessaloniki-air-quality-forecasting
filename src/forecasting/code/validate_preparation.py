"""Verify real cohorts, measured targets, chronological purges and fit boundaries."""
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
from quality import VARIABLES, fill_inputs
ROOT=material_source(__file__).parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def run():
    cfg=read(ROOT/'config.json');prep=read(ROOT/'data/prepared/preprocessing.json')
    source=pd.read_parquet(ROOT/'data/source/original_pollutants.parquet')
    assert list(source.columns)==['station','datetime','pollutant','value']
    obs=pd.read_parquet(ROOT/'data/prepared/observations_qc.parquet')
    original=pd.read_parquet(ROOT/'data/prepared/observations_original.parquet')
    samples=pd.read_parquet(ROOT/'data/prepared/samples.parquet')
    assert samples.row_id.is_unique and not samples.duplicated(['station_id','origin_time']).any()
    assert samples.pm10_observed_hours.ge(18).all()
    assert samples.input_time.eq(samples.origin_time).all()
    boundary=pd.Timestamp(cfg['inner_boundary'])
    fit=obs.loc[obs.datetime<boundary,VARIABLES]
    np.testing.assert_allclose(fit.median().fillna(0).to_numpy(),[prep['median'][c] for c in VARIABLES])
    for sid,g in obs.groupby('station_id'):
        g=g.set_index('datetime');s=samples[samples.station_id==sid]
        grid=pd.date_range(g.index.min(),g.index.max(),freq='h');assert g.index.equals(grid)
        count=g.PM10.notna().rolling(24,min_periods=1).sum()
        expected=g.index[g.PM10.notna() & count.ge(18)]
        assert pd.DatetimeIndex(s.origin_time).equals(expected)
        np.testing.assert_array_equal(s.Persistence,g.PM10.reindex(s.origin_time))
        raw=original[original.station_id==sid].set_index('datetime')
        retained=g.PM10.notna()
        np.testing.assert_array_equal(g.PM10[retained],raw.PM10[retained])
        for h in cfg['horizons']:
            t=s[f'target_time_h{h}']
            assert t.eq(s.origin_time+pd.Timedelta(hours=h)).all()
            np.testing.assert_allclose(s[f'target_h{h}'],g.PM10.reindex(t),equal_nan=True)
    counts={}
    for h in cfg['horizons']:
        split=samples[f'split_h{h}'];t=samples[f'target_time_h{h}']
        valid=split.isin(['train','meta','calibration','test'])
        assert np.isfinite(samples.loc[valid,f'target_h{h}']).all()
        for label,end in [('train','meta'),('meta','calibration'),('calibration','test')]:
            assert (t[split==label]<pd.Timestamp(cfg['boundaries'][end])).all()
        assert (samples.loc[split=='test','origin_time']>=pd.Timestamp(cfg['boundaries']['test'])).all()
        assert (t[samples[f'inner_train_h{h}']]<boundary).all()
        assert (samples.loc[samples[f'inner_validation_h{h}'],'origin_time']>=boundary).all()
        assert not (samples[f'inner_train_h{h}'] & samples[f'inner_validation_h{h}']).any()
        counts[str(h)]={str(k):int(v) for k,v in split.value_counts().items()}
    X=np.load(ROOT/'cache/X.npy',mmap_mode='r')
    assert X.shape==(len(samples),727)
    for start in range(0,len(X),8192):assert np.isfinite(X[start:start+8192]).all()
    assert prep['source_validity_flags_used'] is False
    result={'passed':True,'checks':['isolated source has only original-value fields','all eligible station-hours recovered independently','every target matches an unchanged retained measured PM10 value','no imputed targets in fitted or evaluated samples','fixed chronological boundaries and target purges','training-only fitted scaling/medians','unique 24-hour inputs and minimum 18 observed PM10 hours','all 727 features finite'],'cohort_counts':counts}
    (ROOT/'qa/preparation_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
