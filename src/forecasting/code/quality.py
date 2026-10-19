"""Independent, trailing-only quality rules. No external validity flags are read."""
import warnings
import numpy as np
import pandas as pd

POLLUTANTS = ['PM10', 'NO2', 'O3', 'CO', 'PM2.5']
WEATHER = ['temperature', 'humidity', 'rain', 'pressure', 'wind_speed_10m', 'wind_speed_100m']
VARIABLES = POLLUTANTS + WEATHER
REASONS = {1:'technical_unusable', 2:'particle_zero_run_24h', 4:'shared_zero_run_6h',
           8:'zero_run_suspect', 16:'positive_flatline_suspect', 32:'hampel_and_rate_suspect',
           64:'neighbour_disagreement', 128:'sustained_disagreement'}


def consecutive_true(mask):
    group = mask.ne(mask.shift()).cumsum()
    return (mask.groupby(group).cumcount()+1).where(mask, 0).astype('int32')


def hard_screen(frame, settings):
    """Excluded values become NaN. Earlier hours are never relabelled retrospectively."""
    clean = frame[VARIABLES].apply(pd.to_numeric, errors='coerce').copy()
    reason = pd.DataFrame(0, index=frame.index, columns=VARIABLES, dtype='uint16')
    for col in VARIABLES:
        x = clean[col]
        bad = (~np.isfinite(x) & x.notna()) | x.isin([-999, -9999])
        # Negative concentrations are excluded from this nonnegative modelling task;
        # this is not a claim of certified instrument invalidity near detection limits.
        if col in POLLUTANTS+['rain','wind_speed_10m','wind_speed_100m']: bad |= x.lt(0)
        if col == 'humidity': bad |= x.lt(0) | x.gt(100)
        if col == 'pressure': bad |= x.le(0)
        if col == 'temperature': bad |= x.lt(-273.15)
        reason.loc[bad, col] |= 1
        clean.loc[bad, col] = np.nan
    zero_run = pd.DataFrame({c:consecutive_true(clean[c].eq(0)) for c in POLLUTANTS})
    simultaneous = zero_run.ge(settings['simultaneous_zero_hours'])
    shared = simultaneous.sum(axis=1).ge(settings['simultaneous_zero_pollutants'])
    if settings['simultaneous_requires_particle']:
        shared &= simultaneous[['PM10','PM2.5']].any(axis=1)
    for col in POLLUTANTS:
        x = clean[col].copy()
        long_zero = zero_run[col].ge(settings['zero_exclusion_hours']) & (col in settings['zero_exclusion_pollutants'])
        joint_zero = simultaneous[col] & shared
        reason.loc[long_zero, col] |= 2
        reason.loc[joint_zero, col] |= 4
        reason.loc[zero_run[col].ge(settings['zero_warning_hours']), col] |= 8
        changed = x.ne(x.shift()) | x.isna()
        length = x.groupby(changed.cumsum()).cumcount()+1
        reason.loc[length.ge(settings['stuck_hours']) & x.gt(0), col] |= 16
        clean.loc[long_zero | joint_zero, col] = np.nan
    return clean, reason


def fit_parameters(clean, fit_before, quantile=.995):
    fit = clean.loc[clean.index < fit_before]
    result = {}
    for col in POLLUTANTS:
        x = fit[col]
        rate = float(x.diff().abs().quantile(quantile))
        center = float(x.abs().median())
        result[col] = {'rate_threshold':rate if np.isfinite(rate) else 1.,
                       'scale_floor':max((center if np.isfinite(center) else 0.)*.05, .01 if col=='CO' else 1.)}
    return result


def statistical_screen(clean, reason, parameters, settings):
    result = reason.copy()
    for col in POLLUTANTS:
        x = clean[col]
        past = x.shift(1).rolling(settings['history_hours'], min_periods=settings['minimum_history'])
        median = past.median()
        width = settings['history_hours']
        padded = np.concatenate([np.full(width, np.nan), x.to_numpy(float)[:-1]])
        windows = np.lib.stride_tricks.sliding_window_view(padded, width)
        mad_values = np.full(len(x), np.nan)
        for start in range(0, len(x), 8192):
            block = windows[start:start+8192]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                center = np.nanmedian(block, axis=1)
                local = np.nanmedian(np.abs(block-center[:,None]), axis=1)
            local[np.isfinite(block).sum(axis=1)<settings['minimum_history']] = np.nan
            mad_values[start:start+len(block)] = local
        mad = pd.Series(mad_values, index=x.index)
        scale = np.maximum(1.4826*mad, parameters[col]['scale_floor'])
        spike = x.sub(median).abs().gt(settings['mad_multiplier']*scale) & x.diff().abs().gt(parameters[col]['rate_threshold'])
        result.loc[spike.fillna(False), col] |= 32
    return result


def fill_inputs(clean, station_median, global_median, hours=2):
    """Forward fill uses at most two past hours. Targets never use this function."""
    return clean.ffill(limit=hours).fillna(station_median).fillna(global_median)
