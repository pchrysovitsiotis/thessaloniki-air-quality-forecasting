# Αξιολόγηση στο κοινό test: σφάλματα, αιχμές, διαστήματα, κόστος.
"""Paired test scores without modifying or trimming high target values."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from .common import *

def scores(f, model):
    y = f.target.to_numpy()
    p = f[model].to_numpy()
    e = p - y
    event = y > CONFIG['event_threshold']
    alarm = p > CONFIG['event_threshold']
    tp = int((event & alarm).sum())
    fp = int((~event & alarm).sum())
    fn = int((event & ~alarm).sum())
    tn = int((~event & ~alarm).sum())
    return {'n': len(f), 'RMSE': float(np.sqrt(mean_squared_error(y, p))), 'MAE': float(mean_absolute_error(y, p)), 'R2': float(r2_score(y, p)) if len(y) > 1 else None, 'bias': float(e.mean()), 'events': int(event.sum()), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'precision': tp / (tp + fp) if tp + fp else None, 'recall': tp / (tp + fn) if tp + fn else None, 'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None, 'false_positive_rate': fp / (fp + tn) if fp + tn else None, 'coverage90': float(((y >= f[model + '_lower']) & (y <= f[model + '_upper'])).mean()), 'mean_interval_width': float((f[model + '_upper'] - f[model + '_lower']).mean()), 'peak_MAE': float(abs(e[event]).mean()) if event.any() else None}

def bootstrap(f, model, reference):
    day = f.target_time.dt.floor('D')
    block = (day - day.min()).dt.days // 7
    sums = pd.DataFrame({'block': block, 'a': (f[model] - f.target) ** 2, 'b': (f[reference] - f.target) ** 2, 'n': 1}).groupby('block').sum().to_numpy()
    sampled = sums[np.random.default_rng(CONFIG['seed']).integers(0, len(sums), size=(CONFIG['bootstrap_replicates'], len(sums)))].sum(1)
    delta = np.sqrt(sampled[:, 0] / sampled[:, 2]) - np.sqrt(sampled[:, 1] / sampled[:, 2])
    return {'model': model, 'reference': reference, 'delta_RMSE': float(np.sqrt(np.mean((f[model] - f.target) ** 2)) - np.sqrt(np.mean((f[reference] - f.target) ** 2))), 'ci95_low': float(np.quantile(delta, 0.025)), 'ci95_high': float(np.quantile(delta, 0.975)), 'blocks': len(sums), 'block_days': 7, 'replicates': CONFIG['bootstrap_replicates']}

def evaluate():
    rows = []
    station_rows = []
    monthly = []
    comparisons = []
    weights = []
    cost = []
    selection = []
    conditional = []
    correlations = []
    summary = {}
    for h in CONFIG['horizons']:
        f = pd.read_parquet(ROOT / 'results' / f'predictions_h{h}.parquet')
        test = f[f.split == 'test'].copy()
        assert test.row_id.is_unique and np.isfinite(test[['target'] + MODELS + ['Persistence']]).all().all()
        scopes = {'all_test': test, 'unflagged_target': test[test.target_suspect == 0], 'flagged_target': test[test.target_suspect > 0]}
        for scope, g in scopes.items():
            if len(g):
                for model in MODELS + ['Persistence']:
                    rows.append({'horizon': h, 'scope': scope, 'model': model, **scores(g, model)})
        for sid, g in test.groupby('station_id'):
            for model in MODELS + ['Persistence']:
                station_rows.append({'horizon': h, 'station': sid, 'model': model, **scores(g, model)})
        for month, g in test.groupby(test.target_time.dt.strftime('%Y-%m')):
            for model in MODELS + ['Persistence']:
                monthly.append({'horizon': h, 'month': month, 'model': model, **scores(g, model)})
        for level, g in [('PM10_below_50', test[test.target < 50]), ('PM10_50_to_120', test[(test.target >= 50) & (test.target <= 120)]), ('PM10_above_120', test[test.target > 120])]:
            if len(g):
                for model in MODELS + ['Persistence']:
                    conditional.append({'horizon': h, 'target_level': level, 'model': model, **scores(g, model)})
        for split in ['meta', 'test']:
            g = f[f.split == split]
            for a, b in [('TCN', 'ANN'), ('TCN', 'XGBoost'), ('ANN', 'XGBoost')]:
                correlations.append({'horizon': h, 'split': split, 'first': a, 'second': b, 'n': len(g), 'residual_Pearson': float(np.corrcoef(g[a] - g.target, g[b] - g.target)[0, 1])})
        for model in MODELS + ['Persistence']:
            if model != 'XGBoost':
                comparisons.append({'horizon': h, **bootstrap(test, model, 'XGBoost')})
        for name, parts in HYBRIDS.items():
            for component in parts:
                comparisons.append({'horizon': h, **bootstrap(test, name, component)})
        ensemble = read(ROOT / 'models' / f'h{h}/ensemble.json')
        for model in MODELS:
            components = HYBRIDS.get(model, [model])
            cost.append({'horizon': h, 'model': model, 'training_seconds_including_selection': sum((ensemble['model_seconds'][c] for c in components)), 'inference_ms_per_row': 1000 * sum((ensemble['inference_seconds'][c] for c in components)) / ensemble['inference_rows']})
        for name, info in ensemble['hybrids'].items():
            weights.append({'horizon': h, 'model': name, **{c: float(info['weights'][info['components'].index(c)]) if c in info['components'] else 0.0 for c in ['TCN', 'ANN', 'XGBoost']}})
        for name in ['ANN', 'XGBoost', 'TCN']:
            info = read(ROOT / 'models' / f'h{h}' / name / 'manifest.json')
            for candidate in info['candidates']:
                selection.append({'horizon': h, 'model': name, 'loss': candidate['loss'], 'validation_RMSE': candidate['validation_RMSE'], 'selected': candidate['loss'] == info['selected_loss'], 'epochs_or_trees': candidate['epochs']})
        ranking = sorted([{'model': model, **scores(test, model)} for model in MODELS], key=lambda d: d['RMSE'])
        best = ranking[0]
        best_base = min([d for d in ranking if d['model'] in ['TCN', 'ANN', 'XGBoost']], key=lambda d: d['RMSE'])
        summary[str(h)] = {'n': len(test), 'first_origin': str(test.origin_time.min()), 'last_origin': str(test.origin_time.max()), 'first_target': str(test.target_time.min()), 'last_target': str(test.target_time.max()), 'stations': sorted(test.station_id.unique()), 'target_max': float(test.target.max()), 'target_events': int((test.target > CONFIG['event_threshold']).sum()), 'flagged_targets': int(test.target_suspect.sum()), 'best_by_test_RMSE': best, 'best_base_by_test_RMSE': best_base, 'ranking': ranking, 'best_vs_best_base': bootstrap(test, best['model'], best_base['model'])}
    for name, values in [('metrics', rows), ('station_metrics', station_rows), ('monthly_metrics', monthly), ('conditional_metrics', conditional), ('residual_correlations', correlations), ('paired_comparisons', comparisons), ('hybrid_weights', weights), ('compute_cost', cost), ('training_selection', selection)]:
        table = pd.DataFrame(values)
        if name == 'paired_comparisons':
            table = table.drop_duplicates(['horizon', 'model', 'reference'])
        table.to_csv(ROOT / 'results' / f'{name}.csv', index=False)
    write(ROOT / 'results/summary.json', {'run_key': read(ROOT / 'run_manifest.json')['run_key'], 'horizons': summary, 'same_test_rows_for_all_models': True, 'uncertainty': 'Paired seven-day bootstrap, one fitted seed, no multiplicity correction; intervals describe conditional test differences, not prospective guarantees.', 'limits': ['Historical period already inspected in prior project versions, not a new untouched external test.', 'No EEA validity or matched-value fields used. Original values screened by frozen independent rules; statistical flags remain unconfirmed anomalies, not certified instrument failures.', 'All models use the same 24h tensor and derived context, but architecture capacity and inductive biases differ.', 'Long particulate or corroborated multi-pollutant zero runs excluded prospectively by explicit heuristic rules; positive flatlines and spikes retained.', 'GR0463A has no final test/calibration data because its series ends in 2024.', 'Historical weather is not an archived as-issued forecast; publication delay is assumed.', 'Performance against station readings is not proof of accuracy against latent true pollution.', 'Previous raw and processed metrics used different test cohorts. These new metrics do not measure the causal effect of QC alone.', 'Forecast origins require retained measured PM10 and at least 18 observed hours within the last 24; complete outages and low-coverage histories are not evaluated.']})
    status('evaluated')
    for h, d in summary.items():
        print(h, d['best_by_test_RMSE']['model'], d['best_by_test_RMSE']['RMSE'], flush=True)
