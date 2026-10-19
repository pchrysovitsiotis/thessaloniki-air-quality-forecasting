# Μοντέλα: TCN, ANN, XGBoost. Νέα εκπαίδευση με ανεξάρτητο έλεγχο ποιότητας.
# fit_base: εκπαίδευση. predict_base: χρήση αποθηκευμένου μοντέλου.
"""Raw-control estimators receive the same 24-hour original-variable history."""
import copy, gc, time, random
import joblib
import numpy as np
import torch
from torch import nn
from xgboost import XGBRegressor
from .common import *

def seed():
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    torch.set_num_threads(CONFIG['threads'])
    torch.use_deterministic_algorithms(True)

class CausalBlock(nn.Module):

    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.first = nn.Conv1d(inputs, outputs, 3, dilation=dilation)
        self.second = nn.Conv1d(outputs, outputs, 3, dilation=dilation)
        self.shortcut = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y = self.dropout(torch.relu(self.first(nn.functional.pad(x, (self.padding, 0)))))
        y = self.dropout(torch.relu(self.second(nn.functional.pad(y, (self.padding, 0)))))
        return torch.relu(y + self.shortcut(x))

class Network(nn.Module):

    def __init__(self, kind, dimension, channels, length, static):
        super().__init__()
        self.kind = kind
        self.channels = channels
        self.length = length
        self.sequence_size = channels * length
        if kind == 'TCN':
            blocks = []
            inputs = channels
            for d in [1, 2, 4, 8, 16]:
                blocks.append(CausalBlock(inputs, CONFIG['tcn_channels'], d))
                inputs = CONFIG['tcn_channels']
            self.temporal = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.Linear(inputs + static, 32), nn.ReLU(), nn.Linear(32, 1))
        else:
            a, b = CONFIG['ann_hidden']
            self.head = nn.Sequential(nn.Linear(dimension, a), nn.ReLU(), nn.Dropout(0.1), nn.Linear(a, b), nn.ReLU(), nn.Dropout(0.1), nn.Linear(b, 1))

    def forward(self, x):
        if self.kind == 'TCN':
            z = self.temporal(x[:, :self.sequence_size].reshape(-1, self.channels, self.length))[:, :, -1]
            x = torch.cat([z, x[:, self.sequence_size:]], 1)
        return self.head(x).squeeze(1)

def make_network(kind, schema):
    return Network(kind, schema['shape'][1], len(schema['sequence_channels']), schema['history_hours'], len(schema['static_features']))

def neural_predict(net, X, indices, mean, scale):
    net.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(indices), CONFIG['batch_size']):
            b = indices[a:a + CONFIG['batch_size']]
            p = net(torch.from_numpy(np.asarray(X[b]).copy())).numpy()
            out.append(np.maximum(0, p.astype(float) * scale + mean))
    return np.concatenate(out)

def neural_fit(kind, loss_name, X, y, indices, validation, schema, mean, scale, epochs, context):
    seed()
    net = make_network(kind, schema)
    opt = torch.optim.AdamW(net.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.001)
    target = ((y - mean) / scale).astype('float32')
    rng = np.random.default_rng(CONFIG['seed'])
    best = np.inf
    state = None
    chosen = epochs
    bad = 0
    history = []
    for epoch in range(1, epochs + 1):
        tick = time.monotonic()
        net.train()
        order = rng.permutation(indices)
        total = 0.0
        for a in range(0, len(order), CONFIG['batch_size']):
            ids = order[a:a + CONFIG['batch_size']]
            opt.zero_grad(set_to_none=True)
            p = net(torch.from_numpy(np.asarray(X[ids]).copy()))
            truth = torch.from_numpy(target[ids])
            loss = nn.functional.mse_loss(p, truth) if loss_name == 'MSE' else nn.functional.huber_loss(p, truth, delta=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(ids)
        row = {'epoch': epoch, 'loss': total / len(indices), 'seconds': time.monotonic() - tick}
        if validation is not None:
            pred = neural_predict(net, X, validation, mean, scale)
            value = float(np.sqrt(np.mean((pred - y[validation]) ** 2)))
            row['validation_RMSE'] = value
            if value < best - 0.0001:
                best = value
                state = copy.deepcopy(net.state_dict())
                chosen = epoch
                bad = 0
            else:
                bad += 1
        history.append(row)
        status('training', model=kind, loss=loss_name, phase=context, epoch=epoch, seconds=round(row['seconds'], 2), validation_RMSE=row.get('validation_RMSE'))
        if validation is not None and bad >= CONFIG['patience']:
            break
    if state is not None:
        net.load_state_dict(state)
    return (net, {'history': history, 'epochs': chosen, 'validation_RMSE': best if np.isfinite(best) else None})

def fit_base(kind, X, y, train, inner, validation, schema, destination, key):
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / 'manifest.json'
    if manifest.exists():
        info = read(manifest)
        assert info['run_key'] == key, 'Cache belongs to different run'
        for name, h in info['files'].items():
            assert digest(destination / name) == h, 'Damaged cache'
        return info
    mean = float(np.mean(y[inner]))
    scale = max(1.0, float(np.std(y[inner])))
    start = time.monotonic()
    candidates = []
    if kind in ['TCN', 'ANN']:
        for loss_name in ['MSE', 'Huber']:
            net, diag = neural_fit(kind, loss_name, X, y, inner, validation, schema, mean, scale, CONFIG['neural_max_epochs'], 'inner selection')
            candidates.append({'loss': loss_name, **diag})
            del net
            gc.collect()
        chosen = min(candidates, key=lambda d: d['validation_RMSE'])
        net, refit = neural_fit(kind, chosen['loss'], X, y, train, None, schema, mean, scale, chosen['epochs'], 'refit all train')
        torch.save(net.state_dict(), destination / 'weights.pt')
        del net
        filename = 'weights.pt'
    else:
        Xtrain = np.asarray(X[inner])
        Xval = np.asarray(X[validation])
        ys = (y - mean) / scale
        options = dict(n_estimators=CONFIG['xgb_max_trees'], max_depth=CONFIG['xgb_depth'], learning_rate=0.04, subsample=0.85, colsample_bytree=0.85, tree_method='hist', max_bin=128, n_jobs=CONFIG['threads'], random_state=CONFIG['seed'], eval_metric='rmse', early_stopping_rounds=30)
        for loss_name, objective in [('MSE', 'reg:squarederror'), ('Huber', 'reg:pseudohubererror')]:
            status('training', model=kind, loss=loss_name, phase='inner selection')
            model = XGBRegressor(**options, objective=objective)
            tick = time.monotonic()
            model.fit(Xtrain, ys[inner], eval_set=[(Xval, ys[validation])], verbose=False)
            pred = np.maximum(0, model.predict(Xval).astype(float) * scale + mean)
            candidates.append({'loss': loss_name, 'objective': objective, 'epochs': int(model.best_iteration) + 1, 'validation_RMSE': float(np.sqrt(np.mean((pred - y[validation]) ** 2))), 'seconds': time.monotonic() - tick})
            del model
            gc.collect()
        del Xtrain, Xval
        gc.collect()
        chosen = min(candidates, key=lambda d: d['validation_RMSE'])
        options.pop('early_stopping_rounds')
        options['n_estimators'] = chosen['epochs']
        status('training', model=kind, phase='refit all train', loss=chosen['loss'], trees=chosen['epochs'])
        model = XGBRegressor(**options, objective=chosen['objective'])
        model.fit(np.asarray(X[train]), ys[train], verbose=False)
        joblib.dump(model, destination / 'model.joblib')
        del model
        filename = 'model.joblib'
        refit = None
    info = {'run_key': key, 'model': kind, 'target_mean': mean, 'target_scale': scale, 'selected_loss': chosen['loss'], 'selected_epochs_or_trees': chosen['epochs'], 'selection_RMSE': chosen['validation_RMSE'], 'candidates': candidates, 'refit': refit, 'seconds': time.monotonic() - start, 'train_rows': len(train), 'inner_rows': len(inner), 'validation_rows': len(validation), 'files': {filename: digest(destination / filename)}}
    write(manifest, info)
    gc.collect()
    return info

def predict_base(kind, X, indices, schema, destination):
    info = read(destination / 'manifest.json')
    start = time.monotonic()
    if kind == 'XGBoost':
        model = joblib.load(destination / 'model.joblib')
        chunks = []
        for a in range(0, len(indices), 2048):
            chunks.append(np.maximum(0, model.predict(np.asarray(X[indices[a:a + 2048]])).astype(float) * info['target_scale'] + info['target_mean']))
        result = np.concatenate(chunks)
    else:
        net = make_network(kind, schema)
        net.load_state_dict(torch.load(destination / 'weights.pt', map_location='cpu', weights_only=True))
        result = neural_predict(net, X, indices, info['target_mean'], info['target_scale'])
    return (result, time.monotonic() - start)
