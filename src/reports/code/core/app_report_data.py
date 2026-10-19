"""Deterministic evidence for three citizen reports; no training or LLM calculations."""
from pathlib import Path
import hashlib
import json
import math
from zoneinfo import ZoneInfo
import pandas as pd
from settings import PROJECT

MODEL = 'TCN_ANN_XGBoost'
HORIZONS = [1, 6, 12]
NAMES = {'GR0018A':'Αγία Σοφία', 'GR0019A':'Καλαμαριά', 'GR0020A':'Κορδελιό',
         'GR0046A':'Σίνδος'}
EEA_URL = 'https://airindex.eea.europa.eu/AQI/index.html'
LEVELS = ['Καλή', 'Ικανοποιητική', 'Μέτρια', 'Κακή', 'Πολύ κακή', 'Εξαιρετικά κακή']
# Upper endpoints; decimal values use continuous, non-overlapping intervals.
BANDS = {'PM10':[15,45,120,195,270], 'PM2.5':[5,15,50,90,140],
         'NO2':[10,25,60,100,150], 'O3':[60,100,120,160,180], 'SO2':[20,40,125,190,275]}
GENERAL = ['Συνήθεις εξωτερικές δραστηριότητες.', 'Συνήθεις εξωτερικές δραστηριότητες.',
    'Συνήθεις εξωτερικές δραστηριότητες.',
    'Εξετάστε τη μείωση της έντονης εξωτερικής δραστηριότητας αν εμφανίζετε ερεθισμό ματιών, βήχα ή ερεθισμό λαιμού.',
    'Εξετάστε τη μείωση της έντονης εξωτερικής δραστηριότητας αν εμφανίζετε ερεθισμό ματιών, βήχα ή ερεθισμό λαιμού.',
    'Μειώστε τη σωματική δραστηριότητα σε εξωτερικούς χώρους.']
SENSITIVE = ['Συνήθεις εξωτερικές δραστηριότητες.', 'Συνήθεις εξωτερικές δραστηριότητες.',
    'Εξετάστε τη μείωση της έντονης εξωτερικής δραστηριότητας αν εμφανίζετε συμπτώματα.',
    'Εξετάστε τη μείωση της σωματικής δραστηριότητας, ιδιαίτερα έξω, ειδικά αν εμφανίζετε συμπτώματα.',
    'Μειώστε τη σωματική δραστηριότητα, ιδιαίτερα έξω, ειδικά αν εμφανίζετε συμπτώματα.',
    'Αποφύγετε τη σωματική δραστηριότητα σε εξωτερικούς χώρους.']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def display(v): return f'{float(v):.1f}'.replace('.', ',')
def local(t): return pd.Timestamp(t).tz_convert(ZoneInfo('Europe/Athens')).strftime('%d/%m/%Y %H:%M')

def band(value, pollutant='PM10'):
    if value is None or not math.isfinite(float(value)) or value < 0: return None
    return next((i for i, upper in enumerate(BANDS[pollutant]) if value <= upper), 5)

def guidance(values, station_type):
    indices = {p:band(v,p) for p,v in values.items() if p in BANDS and band(v,p) is not None}
    minimum = 'NO2' in indices and ('PM10' in indices or 'PM2.5' in indices)
    if station_type != 'traffic': minimum = minimum and 'O3' in indices
    level = max(indices.values()) if indices else None
    # An incomplete low index cannot support reassurance. An observed poor pollutant
    # can still support the corresponding precaution (EEA incomplete-data convention).
    usable = minimum or (level is not None and level >= 3)
    return {'level':level, 'category':LEVELS[level] if level is not None else 'Άγνωστη',
        'minimum_pollutant_coverage':minimum, 'basis':'available measured pollutants only',
        'dominant_pollutants':[p for p,i in indices.items() if i==level],
        'missing_pollutants':[p for p in BANDS if p not in indices],
        'general':GENERAL[level] if usable else 'Ανεπαρκής κάλυψη ρύπων για οδηγία δραστηριοτήτων· συμβουλευτείτε την επίσημη ενημέρωση.',
        'sensitive':SENSITIVE[level] if usable else 'Ανεπαρκής κάλυψη ρύπων για οδηγία δραστηριοτήτων· συμβουλευτείτε την επίσημη ενημέρωση.',
        'source':EEA_URL, 'guidance_available':bool(usable)}

def build_evidence(project=PROJECT):
    experiment = project.parent/'forecasting'
    paths = [experiment/f'runs/independent_h{h}/results/predictions_h{h}.parquet' for h in HORIZONS]
    frames = [pd.read_parquet(p).query("split == 'test'") for p in paths]
    keys = ['station_id','input_time']
    common = frames[0][keys].merge(frames[1][keys], on=keys, validate='one_to_one').merge(frames[2][keys], on=keys, validate='one_to_one')
    qpath = experiment/'data/prepared/observations_qc.parquet'
    obs = pd.read_parquet(qpath)
    obs['datetime'] = pd.to_datetime(obs.datetime, utc=True)
    good = obs[obs.PM10.notna() & obs.PM10.ge(0) & obs.PM10_suspect.eq(0)]
    common = common.merge(good[['station_id','datetime']], left_on=keys,
                          right_on=['station_id','datetime'], validate='one_to_one')
    counts = common.groupby('input_time').station_id.nunique()
    if counts.empty or counts.max() < 2: raise ValueError('Δεν υπάρχουν δύο σταθμοί με κοινή ώρα.')
    origin = counts[counts==counts.max()].index.max()
    ids = sorted(common[common.input_time==origin].station_id.unique())
    metadata = project/'data/PM10_metadata.csv'
    meta = pd.read_csv(metadata)
    meta['begin'] = pd.to_datetime(meta['Operational Activity Begin'], dayfirst=True, errors='coerce')
    facts = {'origin':local(origin), 'unit':'μg/m³'}
    stations = []
    for sid in ids:
        o = obs[(obs.station_id==sid) & (obs.datetime==origin)].iloc[0]
        m = meta[meta['Sampling Point Id'].str.contains(sid,regex=False)].sort_values('begin')
        kind = str(m.iloc[-1]['Air Quality Station Type']).lower()
        values = {p:float(o[p]) for p in ['PM10','PM2.5','NO2','O3']
                  if pd.notna(o[p]) and o[p]>=0 and o[p+'_suspect']==0}
        s = {'id':sid,'name':NAMES[sid],'station_type':kind,'time':local(origin),
             'observed_pm10':float(o.PM10), 'measured_pollutants':values,
             'guidance':guidance(values,kind),'forecasts':[]}
        facts[sid+'.observed'] = display(o.PM10)
        for h,f in zip(HORIZONS,frames):
            r = f[(f.station_id==sid) & (f.input_time==origin)].iloc[0]
            assert pd.Timestamp(r.target_time)-pd.Timestamp(origin)==pd.Timedelta(hours=h)
            assert abs(float(r.Persistence)-float(o.PM10))<1e-6
            pred = float(r[MODEL]); delta = pred-float(o.PM10)
            # This is a display tolerance, not a statistically significant change.
            direction = 'περίπου σταθερή' if abs(delta)<1 else 'αύξηση' if delta>0 else 'μείωση'
            s['forecasts'].append({'horizon':h,'time':local(r.target_time),'pm10':display(pred),
                'difference':display(delta),'direction':direction,
                'lower':display(r[MODEL+'_lower']),'upper':display(r[MODEL+'_upper'])})
            facts[sid+f'.h{h}'] = display(pred)
        facts[sid+'.category'] = s['guidance']['category']
        facts[sid+'.general'] = s['guidance']['general']
        facts[sid+'.sensitive'] = s['guidance']['sensitive']
        stations.append(s)
    ranking = sorted(stations,key=lambda s:(s['observed_pm10'],s['id']))
    gap = ranking[-1]['observed_pm10']-ranking[0]['observed_pm10']
    facts['comparison.minimum'] = ranking[0]['name']
    facts['comparison.maximum'] = ranking[-1]['name']
    facts['comparison.gap'] = display(gap)
    evidence = {'mode':'historical_replay','origin_utc':pd.Timestamp(origin).isoformat(),
        'origin_local':local(origin),'timezone':'Europe/Athens','time_definition':'end of hourly interval',
        'unit':'μg/m³','history_hours':24,'forecast_horizons':HORIZONS,'model':MODEL,
        'selection':'Latest test origin with maximum simultaneous station coverage and non-suspect observed PM10; predictions not used to select date.',
        'stations':stations, 'unavailable':[{'id':sid,'name':name,'reason':'Δεν υπάρχουν κοινές αξιολογήσιμες προβλέψεις και μέτρηση σε αυτή την ώρα.'} for sid,name in NAMES.items() if sid not in ids],
        'observed_ranking':[{'name':s['name'],'pm10':display(s['observed_pm10'])} for s in ranking],
        'observed_gap':display(gap), 'facts':facts,
        'guidance_source':{'url':EEA_URL,'checked_on':'2026-10-09','basis':'Hourly concentrations, EEA methodology; Greek paraphrase of health messages.',
            'bands_upper':BANDS,'decimal_convention':'Continuous intervals: previous upper < concentration <= upper. No rounding before categorisation.'},
        'constraints':[
            'Historical replay, never current/live/today; explicit full date and local time in each report.',
            'Only +1/+6/+12 point forecasts, no interpolation or entire best time window.',
            'No future observed targets are supplied; forecast is not a measurement.',
            'Activities: describe only the supplied measured-pollutant guidance, including missing coverage; no forecast of an overall index from PM10 alone.',
            'Do not diagnose, prescribe medication, promise safe exposure or infer a pollution source.',
            'No legal daily-limit claims from hourly values; no personalised medical advice.',
            'Comparison: same origin for all stations; a station does not represent every street; missing is not zero.',
            'Intervals are estimates with imperfect historical coverage, never guaranteed probabilities.',
            'The 1 μg/m³ direction tolerance is a display convention, not statistical significance.'
        ]}
    provenance = {'sources_sha256':{str(p.relative_to(project.parent)):digest(p) for p in paths+[qpath,metadata]},
                  'retraining_performed':False,'source_code_sha256':digest(Path(__file__))}
    return evidence, provenance
