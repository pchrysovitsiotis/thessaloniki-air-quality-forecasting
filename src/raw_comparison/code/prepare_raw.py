"""Build original-value inputs; freeze common validation and evaluation targets."""
# The source tree is separate from input data and generated artifacts.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.dont_write_bytecode = True
_layout_sys.path.insert(0, str(next(p for p in _LayoutPath(__file__).resolve().parents if p.name == 'src')))
from _pipeline_paths import material_source, code_directory
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

ROOT = material_source(__file__).parents[1]
REF = ROOT.parent / 'forecasting'
IDS = ['GR0018A', 'GR0019A', 'GR0020A', 'GR0046A', 'GR0463A']
POLS = ['PM10', 'NO2', 'O3', 'CO', 'PM2.5']
WEATHER = ['temperature', 'humidity', 'rain', 'pressure', 'wind_speed_10m', 'wind_speed_100m']
VARIABLES = POLS + WEATHER


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')
    pending.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    completed = ROOT / 'data/prepared/preparation_audit.json'
    if completed.exists():
        print('Prepared input already exists; verify hashes before reuse.', flush=True)
        audit = read(completed)
        for rel, value in audit['prepared_sha256'].items():
            assert sha(ROOT / rel) == value, rel
        return
    cfg = read(ROOT / 'config.json')
    protocol = {'frozen_before_training': True, 'purpose': 'Paired raw-versus-processed experiment with wholly independent QC selection of validation and evaluation rows.', 'history_hours': 24, 'horizons': [1, 6, 12], 'features': '11 original variables x 24 lags = 264; no masks, quality flags, calendar, station/coordinates or rolling summaries.', 'raw_pollutants': 'Original local values; zeros retained; no EEA fields are read.', 'minimal_preparation': 'Hourly alignment, rejection of nonfinite/negative pollutant values, training-only pooled median imputation and IQR scaling.', 'targets': 'Raw measured PM10 in training; targets never imputed.', 'validation_and_evaluation': 'Exactly the independent-QC arm inner validation, meta, calibration and complete test rows. Their target values must equal the original observations.', 'selection': 'Same optimizer budgets, hidden sizes, losses, seed, and chronological boundaries; all final models refitted from scratch.', 'comparison': 'Paired scores on the entire independent test, not an intersection with a previous externally selected test.', 'limits': ['Combined intervention includes QC, features, input imputation and training sample selection.', 'One seed and previously inspected historical period; not a new untouched external test.', 'Raw training includes more zero-placeholder observations; numerical preparation is still required.']}
    write(ROOT / 'protocol.json', protocol)
    raw = pd.read_parquet(ROOT / 'data/source/original_pollutants.parquet').rename(columns={'value':'original'})
    weather = pd.read_parquet(REF / 'data/source/station_weather.parquet').rename(columns={
        'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity', 'surface_pressure': 'pressure'})
    reference = pd.read_parquet(REF / 'data/prepared/samples.parquet')
    periods = read(REF / 'data/prepared/station_periods.json')
    grids, changes = {}, []
    for sid in IDS:
        src = raw[raw.station == sid].copy()
        assert not src.duplicated(['datetime', 'pollutant']).any()
        table = src.pivot(index='datetime', columns='pollutant', values='original').reindex(columns=POLS)
        table.index = pd.to_datetime(table.index, utc=True) + pd.Timedelta(hours=1)
        bounds = periods[sid]
        hours = pd.date_range(bounds['first_input_end'], bounds['last_input_end'], freq='h')
        g = table.reindex(hours).copy()
        for pol in POLS:
            invalid = ~np.isfinite(g[pol]) | (g[pol] < 0)
            changes.append({'station': sid, 'variable': pol, 'technical_invalid_nonmissing': int((invalid & g[pol].notna()).sum()),
                            'original_zeros_retained': int(g[pol].eq(0).sum())})
            g[pol] = g[pol].mask(invalid)
        w = weather[weather.station_id == sid].set_index('datetime')[WEATHER].reindex(hours)
        assert np.isfinite(w.to_numpy()).all(), 'Reference weather must be complete.'
        g[WEATHER] = w
        g.index.name = 'datetime'
        grids[sid] = g
        destination = ROOT / 'data/prepared/stations' / f'{sid}.csv'
        destination.parent.mkdir(parents=True, exist_ok=True)
        g.reset_index().to_csv(destination, index=False)
    inner_boundary = pd.Timestamp(cfg['inner_boundary'])
    fit = pd.concat([g.loc[g.index < inner_boundary, VARIABLES] for g in grids.values()])
    median = fit.median().fillna(0)
    quantiles = fit.quantile([.25, .75])
    scale = ((quantiles.loc[.75] - quantiles.loc[.25]) / 1.349)
    scale = scale.where(scale > 1e-6, 1).fillna(1)
    preprocessing = {'variables': VARIABLES, 'channels': VARIABLES, 'median': median.to_dict(), 'scale': scale.to_dict(),
                     'fit_before': str(inner_boundary), 'scale_definition': 'raw-training IQR / 1.349, zero or missing scale -> 1; no clipping',
                     'external_quality_flags_used': False, 'engineered_features': False, 'technical_changes': changes,
                     'time_convention': 'original UTC hour-start +1h -> observation end; weather already observation-end aligned',
                     'target_policy': protocol['targets'], 'validation_policy': protocol['validation_and_evaluation']}
    write(ROOT / 'data/prepared/preprocessing.json', preprocessing)
    frames, positions, row_start, counts = [], {}, 0, []
    for sid, g in grids.items():
        anchors = g.index[g.PM10.notna()]
        ref = reference[reference.station_id == sid].set_index('input_time').reindex(anchors)
        m = pd.DataFrame({'station_id': sid, 'input_time': anchors, 'origin_time': anchors,
                          'Persistence': g.PM10.reindex(anchors).to_numpy()})
        for h in cfg['horizons']:
            target_time = anchors + pd.Timedelta(hours=h)
            target = g.PM10.reindex(target_time).to_numpy(copy=True)
            labels = np.full(len(m), 'test', dtype=object)
            bounds = [pd.Timestamp(cfg['boundaries'][k]) for k in ['meta', 'calibration', 'test']]
            for label, boundary in reversed(list(zip(['train', 'meta', 'calibration'], bounds))):
                labels[anchors < boundary] = label
            for label, boundary in zip(['train', 'meta', 'calibration'], bounds):
                labels[(labels == label) & (target_time >= boundary)] = 'purged'
            labels[~np.isfinite(target)] = 'no_target'
            refsplit = ref[f'split_h{h}'].to_numpy()
            for label in ['meta', 'calibration', 'test']:
                intended = labels == label
                shared = refsplit == label
                assert np.all(~shared | intended), (sid, h, label)
                labels[intended & ~shared] = 'excluded_noncommon'
                assert np.allclose(target[shared], ref.loc[shared, f'target_h{h}'].to_numpy(), atol=0, rtol=0)
                target[shared] = ref.loc[shared, f'target_h{h}'].to_numpy()
            m[f'target_time_h{h}'] = target_time
            m[f'target_h{h}'] = target
            m[f'target_suspect_h{h}'] = ref[f'target_suspect_h{h}'].fillna(False).astype(bool).to_numpy()
            m[f'split_h{h}'] = labels
            m[f'inner_train_h{h}'] = (anchors < inner_boundary) & (target_time < inner_boundary) & (labels == 'train')
            m[f'inner_validation_h{h}'] = ref[f'inner_validation_h{h}'].eq(True).to_numpy()
            assert np.all(~m[f'inner_validation_h{h}'].to_numpy() | (labels == 'train'))
            for label in ['train', 'meta', 'calibration', 'test', 'excluded_noncommon']:
                counts.append({'station': sid, 'horizon': h, 'split': label, 'n': int((labels == label).sum()),
                               'reference_n': int((refsplit == label).sum())})
        m['row_id'] = np.arange(row_start, row_start + len(m))
        positions[sid] = (row_start, row_start + len(m), g.index.get_indexer(anchors))
        row_start += len(m)
        frames.append(m)
    metadata = pd.concat(frames, ignore_index=True)
    metadata.to_parquet(ROOT / 'data/prepared/samples.parquet', index=False)
    pd.DataFrame(counts).to_csv(ROOT / 'results/sample_counts.csv', index=False)
    observations = pd.concat([g.reset_index().assign(station_id=sid) for sid, g in grids.items()], ignore_index=True)
    observations.to_parquet(ROOT / 'data/prepared/observations_raw.parquet', index=False)
    length = 24
    names = [f'{v}_lag_{lag}' for v in VARIABLES for lag in range(length-1, -1, -1)]
    X = open_memmap(ROOT / 'cache/X.npy', mode='w+', dtype='float32', shape=(len(metadata), len(names)))
    for sid, (start, end, indices) in positions.items():
        g = grids[sid][VARIABLES]
        values = ((g.fillna(median) - median) / scale).to_numpy('float32')
        for a in range(start, end, 2048):
            b = min(end, a + 2048)
            idx = indices[a-start:b-start, None] + np.arange(-23, 1)[None, :]
            seq = values[np.maximum(idx, 0)].copy()
            seq[idx < 0] = 0
            X[a:b] = seq.transpose(0, 2, 1).reshape(b-a, -1)
            assert np.isfinite(X[a:b]).all()
        # Verify exact source-time alignment, last-hour position and channel-major layout.
        for row in [start, min(start+51, end-1), end-1]:
            t = pd.Timestamp(metadata.iloc[row].input_time)
            check = g.reindex(pd.date_range(t-pd.Timedelta(hours=23), t, freq='h'))
            expected = ((check.fillna(median)-median)/scale).to_numpy('float32').T.reshape(-1)
            assert np.allclose(X[row], expected, atol=0, rtol=0), (sid, row)
    X.flush()
    del X
    schema = {'sequence_channels': VARIABLES, 'history_hours': 24, 'static_features': [], 'feature_names': names,
              'shape': [len(metadata), 264], 'identical_input_matrix_for_all_models': True,
              'sequence_order': 'oldest to newest; channel-major flatten', 'source_quality_flags_used_as_features': False}
    write(ROOT / 'data/prepared/feature_schema.json', schema)
    for h in cfg['horizons']:
        for label in ['meta', 'calibration', 'test']:
            actual = metadata[metadata[f'split_h{h}'] == label]
            expected = reference[reference[f'split_h{h}'] == label]
            keys = ['station_id', 'input_time', f'target_time_h{h}', f'target_h{h}']
            pd.testing.assert_frame_equal(actual[keys].reset_index(drop=True), expected[keys].reset_index(drop=True), check_dtype=False)
        actual = metadata[metadata[f'inner_validation_h{h}']]
        expected = reference[reference[f'inner_validation_h{h}']]
        keys = ['station_id', 'input_time', f'target_h{h}']
        pd.testing.assert_frame_equal(actual[keys].reset_index(drop=True), expected[keys].reset_index(drop=True), check_dtype=False)
    # Each run receives an identical input snapshot, wholly inside this experiment.
    for h in cfg['horizons']:
        run = ROOT / 'runs' / f'raw_h{h}'
        for folder in ['data/stations', 'data/metadata', 'cache', 'models', 'results']:
            (run/folder).mkdir(parents=True, exist_ok=True)
        for path in (ROOT/'data/prepared').rglob('*'):
            if path.is_file():
                shutil.copy2(path, run/'data'/path.relative_to(ROOT/'data/prepared'))
        shutil.copy2(ROOT/'cache/X.npy', run/'cache/X.npy')
        current = {**cfg, 'horizons': [h]}
        write(run/'config.json', current)
    audit = {'raw_rows': len(raw), 'hourly_grid_rows': len(observations), 'raw_anchors': len(metadata),
             'input_dimension': 264, 'reference_input_dimension': 727, 'common_validation_meta_calibration_test_verified': True,
             'original_zeros_retained': int(observations[POLS].eq(0).sum().sum()),
             'prepared_sha256': {str(p.relative_to(ROOT)): sha(p) for p in list((ROOT/'data/prepared').rglob('*'))+[ROOT/'cache/X.npy'] if p.is_file()}}
    write(completed, audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='prepared_sha256'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
