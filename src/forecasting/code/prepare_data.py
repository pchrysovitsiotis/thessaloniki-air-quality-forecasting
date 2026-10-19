"""Prepare independent QC, causal features and new chronological evaluation cohorts."""
# The source tree is separate from input data and generated artifacts.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.dont_write_bytecode = True
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if p.name == 'src')))
from _pipeline_paths import material_source, code_directory
from pathlib import Path
import hashlib, json, shutil
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from quality import POLLUTANTS, WEATHER, VARIABLES, REASONS, hard_screen, fit_parameters, statistical_screen, fill_inputs

ROOT=material_source(__file__).parents[1]
IDS=['GR0018A','GR0019A','GR0020A','GR0046A','GR0463A']
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.pending');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def robust_center_scale(values,floor):
    x=values.dropna();center=float(x.median()) if len(x) else 0.
    deviation=float((x-center).abs().median())*1.4826 if len(x) else 0.
    return center,max(deviation,floor)

def main():
    completed=ROOT/'data/prepared/preparation_audit.json'
    if completed.exists():
        for rel,digest in read(completed)['prepared_sha256'].items():assert sha(ROOT/rel)==digest,rel
        print('Prepared dataset verified; reuse unchanged.',flush=True);return
    cfg=read(ROOT/'config.json');settings=cfg['qc'];fit_until=pd.Timestamp(cfg['inner_boundary'])
    assert cfg['history_hours']==24 and settings['history_hours']==24
    source=read(ROOT/'source_manifest.json')
    for rel,digest in source['source_files'].items():assert sha(ROOT/rel)==digest,rel
    raw=pd.read_parquet(ROOT/'data/source/original_pollutants.parquet')
    assert list(raw.columns)==['station','datetime','pollutant','value']
    assert not raw.duplicated(['station','datetime','pollutant']).any()
    weather=pd.read_parquet(ROOT/'data/source/station_weather.parquet').rename(columns={'temperature_2m':'temperature','relative_humidity_2m':'humidity','surface_pressure':'pressure'})
    assert not weather.duplicated(['station_id','datetime']).any()
    geo=read(ROOT/'data/source/station_metadata.json')
    original,cleaned,reasons,parameters,periods={},{},{},{},{}
    for sid in IDS:
        s=raw[raw.station==sid].copy();s['datetime']=pd.to_datetime(s.datetime,utc=True)+pd.Timedelta(hours=1)
        assert s.datetime.eq(s.datetime.dt.floor('h')).all()
        s=s[s.datetime>=pd.Timestamp(cfg['start'],tz='UTC')]
        pm=s[s.pollutant=='PM10'];assert len(pm)
        index=pd.date_range(pm.datetime.min(),pm.datetime.max(),freq='h')
        g=s.pivot(index='datetime',columns='pollutant',values='value').reindex(index=index,columns=POLLUTANTS)
        w=weather[weather.station_id==sid].set_index('datetime')[WEATHER].reindex(index)
        g[WEATHER]=w;g.index.name='datetime'
        original[sid]=g
        cleaned[sid],reasons[sid]=hard_screen(g,settings)
        parameters[sid]=fit_parameters(cleaned[sid],fit_until,settings['rate_quantile'])
        reasons[sid]=statistical_screen(cleaned[sid],reasons[sid],parameters[sid],settings)
        periods[sid]={'first_input_end':str(index.min()),'last_input_end':str(index.max()),'raw_pm10_rows':int(g.PM10.notna().sum()),'retained_pm10_rows':int(cleaned[sid].PM10.notna().sum()),'period_origin':'min/max timestamps of original PM10 records, including zeros; no external quality mask'}
        print('QC completed:',sid,periods[sid]['retained_pm10_rows'],flush=True)
    fit=pd.concat([g.loc[g.index<fit_until,VARIABLES] for g in cleaned.values()])
    median=fit.median().fillna(0);quartiles=fit.quantile([.25,.75])
    scale=((quartiles.loc[.75]-quartiles.loc[.25])/1.349);scale=scale.where(scale>1e-6,1).fillna(1)
    station_medians={sid:g.loc[g.index<fit_until,VARIABLES].median().fillna(median) for sid,g in cleaned.items()}
    channel_names=VARIABLES+[c+'_observed' for c in VARIABLES]+[c+'_suspect' for c in POLLUTANTS]+['PM10_neighbour_disagreement','PM10_drift_suspect']
    summary_names=[f'PM10_{stat}_{n}h' for n in [3,6,12,24] for stat in ['mean','std','max','coverage']]+['PM10_change_1h','PM10_change_6h']
    static_names=['hour_sin','hour_cos','weekday_sin','weekday_cos','month_sin','month_cos']+['station_'+s for s in IDS]+['latitude','longitude']+summary_names
    feature_names=[f'{c}_lag_{lag}' for c in channel_names for lag in range(23,-1,-1)]+static_names
    assert len(feature_names)==727
    pm_grid=pd.DataFrame({s:g.PM10 for s,g in cleaned.items()})
    matrices,frame_parts,decision_parts,count_rows,input_rows,split_rows={},{},[],[],[],[]
    observation_parts=[];row_offset=0;positions={};eligibility_rows=[]
    for sid,g in cleaned.items():
        reason=reasons[sid]
        others=pm_grid.drop(columns=sid)
        buddy=others.median(axis=1).where(others.notna().sum(axis=1)>=2).reindex(g.index)
        residual=g.PM10-buddy
        center,deviation=robust_center_scale(residual.loc[residual.index<fit_until],0)
        limit=max(6*deviation,25.)
        trailing=residual.rolling(24,min_periods=18).median()
        drift_center,drift_scale=robust_center_scale(trailing.loc[trailing.index<fit_until],0)
        drift_limit=max(3*drift_scale,15.)
        disagreement=(residual-center).abs().gt(limit)
        drift=(trailing-drift_center).abs().gt(drift_limit)
        parameters[sid]['spatial']={'residual_center':center,'buddy_limit':limit,'minimum_other_stations':2,'trailing_hours':24,'trailing_minimum_hours':18,'drift_center':drift_center,'drift_limit':drift_limit,'action':'flag only; local emissions can differ'}
        obs=g.copy()
        for c in POLLUTANTS:obs[c+'_suspect']=reason[c].ne(0).astype(float)
        obs['PM10_neighbour_disagreement']=disagreement.astype(float);obs['PM10_drift_suspect']=drift.astype(float)
        observation_parts.append(obs.reset_index().assign(station_id=sid))
        reason.loc[disagreement,'PM10']|=64;reason.loc[drift,'PM10']|=128
        fill=fill_inputs(g,station_medians[sid],median,settings['forward_fill_limit_hours'])
        assert np.isfinite(fill.to_numpy()).all()
        z=((fill-median)/scale).to_numpy('float32')
        observed=g.notna().to_numpy('float32')
        matrices[sid]=np.column_stack([z,observed,obs[[c+'_suspect' for c in POLLUTANTS]+['PM10_neighbour_disagreement','PM10_drift_suspect']].to_numpy('float32')])
        for col in VARIABLES:
            before=original[sid][col];after=g[col];code=reason[col]
            excluded=before.notna() & after.isna();flagged=code.ne(0)
            count_rows.append({'station':sid,'variable':col,'grid_hours':len(g),'original_observed':int(before.notna().sum()),'original_missing':int(before.isna().sum()),'excluded':int(excluded.sum()),'suspect_retained':int((flagged & after.notna()).sum()),'retained':int(after.notna().sum()),'forward_filled_input_hours':int((after.isna() & after.ffill(limit=2).notna()).sum()),'median_filled_input_hours':int(after.ffill(limit=2).isna().sum()),**{name:int((code&bit).ne(0).sum()) for bit,name in REASONS.items()}})
            if flagged.any():
                q=pd.DataFrame({'station_id':sid,'datetime':g.index[flagged],'variable':col,'original_value':before.loc[flagged].to_numpy(),'retained_value':after.loc[flagged].to_numpy(),'reason_bits':code.loc[flagged].to_numpy(),'decision':np.where(excluded.loc[flagged],'excluded_by_rule','suspect_retained')})
                decision_parts.append(q)
        count=g.PM10.notna().rolling(24,min_periods=1).sum()
        eligible=g.PM10.notna() & count.ge(settings['minimum_pm10_hours_in_window'])
        eligibility_rows.append({'station':sid,'raw_pm10_anchors':int(original[sid].PM10.notna().sum()),'hard_screen_retained':int(g.PM10.notna().sum()),'anchors_with_18_of_24_observed':int(eligible.sum()),'retained_anchors_excluded_for_low_history_coverage':int((g.PM10.notna() & ~eligible).sum())})
        anchors=g.index[eligible]
        m=pd.DataFrame({'station_id':sid,'input_time':anchors,'origin_time':anchors,'Persistence':g.PM10.loc[anchors].to_numpy(),'pm10_observed_hours':count.loc[anchors].to_numpy()})
        for h in cfg['horizons']:
            target_time=anchors+pd.Timedelta(hours=h);target=g.PM10.reindex(target_time).to_numpy()
            labels=np.full(len(m),'test',dtype=object)
            bounds=[pd.Timestamp(cfg['boundaries'][k]) for k in ['meta','calibration','test']]
            for label,boundary in reversed(list(zip(['train','meta','calibration'],bounds))):labels[anchors<boundary]=label
            for label,boundary in zip(['train','meta','calibration'],bounds):labels[(labels==label)&(target_time>=boundary)]='purged'
            labels[~np.isfinite(target)]='no_target'
            m[f'target_time_h{h}']=target_time;m[f'target_h{h}']=target
            m[f'target_suspect_h{h}']=reason.PM10.ne(0).reindex(target_time).fillna(False).to_numpy(dtype=bool)
            m[f'split_h{h}']=labels
            m[f'inner_train_h{h}']=(anchors<fit_until)&(target_time<fit_until)&(labels=='train')
            m[f'inner_validation_h{h}']=(anchors>=fit_until)&(labels=='train')
            for label in ['train','meta','calibration','test','purged','no_target']:
                group=m[labels==label]
                split_rows.append({'station':sid,'horizon':h,'split':label,'n':len(group)})
        m['row_id']=np.arange(row_offset,row_offset+len(m));positions[sid]=(row_offset,row_offset+len(m),g.index.get_indexer(anchors));row_offset+=len(m);frame_parts[sid]=m
        (ROOT/'data/prepared/stations').mkdir(exist_ok=True)
        g.reset_index().to_csv(ROOT/'data/prepared/stations'/f'{sid}.csv',index=False)
    metadata=pd.concat(frame_parts.values(),ignore_index=True)
    metadata.to_parquet(ROOT/'data/prepared/samples.parquet',index=False)
    pd.concat(observation_parts,ignore_index=True).to_parquet(ROOT/'data/prepared/observations_qc.parquet',index=False)
    pd.concat([g.reset_index().assign(station_id=sid) for sid,g in original.items()],ignore_index=True).to_parquet(ROOT/'data/prepared/observations_original.parquet',index=False)
    decisions=pd.concat(decision_parts,ignore_index=True)
    decoder={i:';'.join(name for bit,name in REASONS.items() if i&bit) for i in decisions.reason_bits.unique()}
    decisions['reasons']=decisions.reason_bits.map(decoder)
    decisions.to_parquet(ROOT/'data/prepared/quality_decisions.parquet',index=False)
    pd.DataFrame(count_rows).to_csv(ROOT/'results/quality_rule_counts.csv',index=False)
    pd.DataFrame(eligibility_rows).to_csv(ROOT/'results/anchor_eligibility.csv',index=False)
    pd.DataFrame(split_rows).to_csv(ROOT/'results/split_counts.csv',index=False)
    write(ROOT/'data/prepared/station_periods.json',periods)
    preprocessing={'source_validity_flags_used':False,'evaluation_cohort_origin':'independent quality rules and fixed calendar boundaries; no old sample lists or validity masks read','fit_before':str(fit_until),'variables':VARIABLES,'channels':channel_names,'median':median.to_dict(),'scale':scale.to_dict(),'station_imputation_median':{sid:m.to_dict() for sid,m in station_medians.items()},'scale_definition':'pooled independent-QC training IQR / 1.349, constant scale -> 1; no clipping','input_imputation':'forward fill max 2 hours then station training median (global training fallback); observed mask preserved','target_policy':'actual retained PM10 only; no target imputation; statistical spikes and positive flatlines retained','quality_parameters':parameters,'quality_settings':settings,'reason_bits':REASONS,'timestamp_convention':'pollutant export UTC hour start +1h -> observation end; archived weather already end aligned','station_periods':periods,'limitations':['Rules identify unusable or suspicious records; they do not certify instrument validity.','Single-pollutant long gas zeros are flagged and retained; only corroborated multi-pollutant zero periods are excluded.','Historical test previously inspected; one training seed; reported timestamps assumed available without measured release latency.']}
    write(ROOT/'data/prepared/preprocessing.json',preprocessing)
    X=open_memmap(ROOT/'cache/X.npy',mode='w+',dtype='float32',shape=(len(metadata),727))
    for sid,(start,end,indices) in positions.items():
        values=matrices[sid]
        for a in range(start,end,1024):
            b=min(a+1024,end);idx=indices[a-start:b-start,None]+np.arange(-23,1)[None,:]
            seq=values[np.maximum(idx,0)].copy();seq[idx<0]=0
            X[a:b,:696]=seq.transpose(0,2,1).reshape(b-a,-1)
            local=pd.DatetimeIndex(metadata.iloc[a:b].origin_time).tz_convert('Europe/Athens')
            static=np.column_stack([np.sin(2*np.pi*local.hour/24),np.cos(2*np.pi*local.hour/24),np.sin(2*np.pi*local.dayofweek/7),np.cos(2*np.pi*local.dayofweek/7),np.sin(2*np.pi*(local.month-1)/12),np.cos(2*np.pi*(local.month-1)/12)]+[np.full(b-a,float(s==sid)) for s in IDS]+[np.full(b-a,(geo[sid]['latitude']-40.6)*10),np.full(b-a,(geo[sid]['longitude']-22.9)*10)])
            summaries=[]
            for n in [3,6,12,24]:
                val=seq[:,-n:,0];known=seq[:,-n:,11]>0;count=known.sum(1);enough=count>=int(np.ceil(.75*n))
                mean=np.divide(np.where(known,val,0).sum(1),count,out=np.zeros(b-a,dtype='float32'),where=count>0)
                var=np.divide(np.where(known,(val-mean[:,None])**2,0).sum(1),count,out=np.zeros(b-a,dtype='float32'),where=count>0)
                maximum=np.where(known,val,-np.inf).max(1)
                summaries.extend([np.where(enough,mean,0),np.where(enough,np.sqrt(var),0),np.where(enough,maximum,0),count/n])
            for lag in [1,6]:
                known=(seq[:,-1,11]>0)&(seq[:,-lag-1,11]>0)
                summaries.append(np.where(known,seq[:,-1,0]-seq[:,-lag-1,0],0))
            X[a:b,696:]=np.column_stack([static]+summaries);assert np.isfinite(X[a:b]).all()
        for row in [start,min(start+37,end-1),end-1]:
            t=metadata.iloc[row].input_time;idx=cleaned[sid].index.get_loc(t)
            direct=np.zeros((24,29),dtype='float32');valid=values[max(0,idx-23):idx+1];direct[-len(valid):]=valid
            np.testing.assert_array_equal(X[row,:696],direct.T.reshape(-1))
    X.flush();del X
    write(ROOT/'data/prepared/feature_schema.json',{'sequence_channels':channel_names,'history_hours':24,'static_features':static_names,'feature_names':feature_names,'shape':[len(metadata),727],'identical_input_matrix_for_all_models':True,'sequence_order':'oldest to newest, channel-major flatten','QC_flags_available_causally':True,'EEA_flags_used':False})
    for h in cfg['horizons']:
        run=ROOT/'runs'/f'independent_h{h}'
        for d in ['data','cache','models','results']:(run/d).mkdir(parents=True,exist_ok=True)
        # Copy the small required inputs; keep full audit records once at experiment level.
        for name in ['samples.parquet','preprocessing.json','feature_schema.json','station_periods.json']:
            shutil.copy2(ROOT/'data/prepared'/name,run/'data'/name)
        shutil.copytree(ROOT/'data/prepared/stations',run/'data/stations',dirs_exist_ok=True)
        shutil.copy2(ROOT/'cache/X.npy',run/'cache/X.npy')
        write(run/'config.json',{**cfg,'horizons':[h]})
    audit={'source_columns_read':list(raw.columns),'external_quality_flags_used':False,'old_sample_lists_used':False,'input_features':727,'model_history_hours':24,'quality_history_hours':24,'matrix_rows':len(metadata),'raw_grid_rows':sum(map(len,original.values())),'pm10_retained':sum(int(g.PM10.notna().sum()) for g in cleaned.values()),'test_counts':{h:int(metadata[f'split_h{h}'].eq('test').sum()) for h in cfg['horizons']},'prepared_sha256':{str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'data/prepared').rglob('*'))+[ROOT/'cache/X.npy'] if p.is_file()}}
    write(completed,audit)
    print('Dataset prepared:',{k:v for k,v in audit.items() if k!='prepared_sha256'},flush=True)

if __name__=='__main__':main()
