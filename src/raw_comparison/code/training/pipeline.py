# Εκπαίδευση βάσεων, συνδυασμός με βάρη και διαστήματα πρόβλεψης.
# Τα βάρη επιλέγονται στο meta και τα διαστήματα στο calibration.
import gc, time, importlib.metadata
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .common import *
from .models import fit_base, predict_base, seed

def weights(predictions, target):
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(target, dtype=float)
    k = p.shape[1]
    result = minimize(lambda w: np.mean((p @ w - y) ** 2), np.full(k, 1 / k), jac=lambda w: 2 * p.T @ (p @ w - y) / len(y), bounds=[(0, 1)] * k, constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1, 'jac': lambda w: np.ones(k)}, method='SLSQP', options={'ftol': 1e-10, 'maxiter': 1000})
    if not result.success:
        raise RuntimeError(result.message)
    w = np.maximum(0, result.x)
    return w / w.sum()

def conformal_quantile(errors, alpha):
    values = np.sort(np.asarray(errors, float))
    i = min(len(values) - 1, int(np.ceil((len(values) + 1) * (1 - alpha))) - 1)
    return float(values[max(0, i)])

def run():
    seed()
    key, payload = fingerprint()
    schema = read(ROOT / 'data/feature_schema.json')
    metadata = pd.read_parquet(ROOT / 'data/samples.parquet')
    X = np.load(ROOT / 'cache/X.npy', mmap_mode='r')
    assert X.shape == tuple(schema['shape']) and np.isfinite(X[:100]).all()
    data_hashes = {name: digest(ROOT / name) for name in ['cache/X.npy', 'data/samples.parquet', 'data/preprocessing.json', 'data/feature_schema.json']}
    previous = ROOT / 'run_manifest.json'
    manifest = {'run_key': key, 'inputs': data_hashes, 'source_and_code': payload, 'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'torch', 'xgboost', 'scikit-learn', 'scipy']}}
    if previous.exists():
        assert read(previous) == manifest, 'Run inputs changed'
    else:
        write(previous, manifest)
    tick = time.monotonic()
    for h in CONFIG['horizons']:
        folder = ROOT / 'models' / f'h{h}'
        folder.mkdir(exist_ok=True)
        target = metadata[f'target_h{h}'].to_numpy()
        split = metadata[f'split_h{h}']
        train = np.flatnonzero(split == 'train')
        inner = np.flatnonzero(metadata[f'inner_train_h{h}'])
        validation = np.flatnonzero(metadata[f'inner_validation_h{h}'])
        prediction_ids = np.flatnonzero(split.isin(['meta', 'calibration', 'test']))
        f = metadata.iloc[prediction_ids][['row_id', 'station_id', 'input_time', 'origin_time', 'Persistence', f'target_time_h{h}', f'target_h{h}', f'target_suspect_h{h}', f'split_h{h}']].rename(columns={f'target_time_h{h}': 'target_time', f'target_h{h}': 'target', f'target_suspect_h{h}': 'target_suspect', f'split_h{h}': 'split'}).copy()
        model_info = {}
        inference = {}
        for kind in ['ANN', 'XGBoost', 'TCN']:
            status('horizon', horizon=h, model=kind, train_rows=len(train), test_rows=int((split == 'test').sum()))
            model_info[kind] = fit_base(kind, X, target, train, inner, validation, schema, folder / kind, key)
            f[kind], inference[kind] = predict_base(kind, X, prediction_ids, schema, folder / kind)
            f.to_parquet(folder / 'partial_predictions.parquet', index=False)
            gc.collect()
        mix = {}
        meta = f[f.split == 'meta']
        cal = f[f.split == 'calibration']
        test = f[f.split == 'test']
        for name, parts in HYBRIDS.items():
            w = weights(meta[parts], meta.target)
            f[name] = f[parts].to_numpy() @ w
            mix[name] = {'components': parts, 'weights': w.tolist(), 'fitted_rows': len(meta), 'constraint': 'nonnegative weights sum to one, no intercept'}
        intervals = {}
        cal = f[f.split == 'calibration']
        for model in MODELS + ['Persistence']:
            e = np.abs(cal[model] - cal.target)
            pooled = conformal_quantile(e, CONFIG['interval_alpha'])
            by_station = {}
            for sid, g in cal.groupby('station_id'):
                by_station[sid] = conformal_quantile(abs(g[model] - g.target), CONFIG['interval_alpha']) if len(g) >= 60 else pooled
            q = f.station_id.map(by_station).fillna(pooled).to_numpy()
            f[model + '_lower'] = np.maximum(0, f[model] - q)
            f[model + '_upper'] = f[model] + q
            intervals[model] = {'pooled_radius': pooled, 'station_radius': by_station, 'calibration_rows': len(cal), 'nominal_coverage': 1 - CONFIG['interval_alpha'], 'method': 'fixed split conformal absolute residuals; time dependence means no guaranteed prospective coverage'}
        f.to_parquet(ROOT / 'results' / f'predictions_h{h}.parquet', index=False)
        write(folder / 'ensemble.json', {'run_key': key, 'hybrids': mix, 'intervals': intervals, 'model_seconds': {k: v['seconds'] for k, v in model_info.items()}, 'inference_seconds': inference, 'inference_rows': len(f)})
        status('horizon_completed', horizon=h)
    assert source_inventory() == read(ROOT / 'source_integrity_before.json'), 'Original source changed'
    write(ROOT / 'training_execution.json', {'run_key': key, 'wall_seconds': time.monotonic() - tick, 'completed_at': str(pd.Timestamp.now(tz='UTC'))})
    status('trained', run_key=key)
