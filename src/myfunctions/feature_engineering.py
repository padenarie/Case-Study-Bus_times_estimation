import pandas as pd
from typing import Callable
import logging
import numpy as np


### Feature Engineering
def add_feature(
    df: pd.DataFrame, *,
    src: str, dest: str,
    function: Callable[[pd.Series], pd.Series],
    overwrite: bool = False,
    **kwargs,
) -> pd.DataFrame:
    
    """
    Add a new column to the DataFrame by applying a function to an existing column.
    """

    if dest in df.columns and not overwrite:
        raise ValueError(f"Column '{dest}' already exists")
    out = function(df[src], **kwargs)
    if not isinstance(out, pd.Series):
        raise TypeError("func must return a pandas Series")
    # align & name for safety
    out = out.rename(dest).reindex(df.index)
    return df.assign(**{dest: out})

def parse_time_column(df_column: pd.Series) -> pd.Series:
    """
    Parse a time column to datetime format.
    """
    return pd.to_datetime(df_column, yearfirst=True, format="%Y_%m_%d_%H", errors='coerce', cache=True)


def add_time_delta(df:pd.DataFrame, real_time:str, plan_time: str, return_series: bool=False) -> pd.DataFrame | pd.Series:
    """
    Compute the time delta between real and planned time.
    """
    
    df_c = df.copy()

    if return_series:
        try:
            return pd.to_timedelta(pd.to_timedelta(df_c[real_time]) - pd.to_timedelta(df_c[plan_time])).dt.total_seconds()
        except Exception as e:
            logging.error(f"Error occurred during time delta computation: {e}")
            return pd.Series(dtype='float64')
    else:
        try:
            df_c['time_delta'] = pd.to_timedelta(pd.to_timedelta(df_c[real_time]) - pd.to_timedelta(df_c[plan_time])).dt.total_seconds()
            return df_c
        except Exception as e:
            logging.error(f"Error occurred during time delta computation: {e}")
            return df_c
        

def add_hourlycheckin_rate(df: pd.DataFrame, col_datetime: str = 'id_Datetime', col_checkins: str = 'number_of_check_ins', return_series: bool = False) -> pd.DataFrame|pd.Series:

    """Add a column with the hourly check-in rate."""
    df_copy = df.copy()
    df_copy[col_datetime] = pd.to_datetime(df_copy[col_datetime])
    
    if not return_series:
        df_copy['checkin_rate x1000'] = df_copy[col_checkins] / 3600 * 1000
        return df_copy
    else:
        return df_copy[col_checkins] / 3600 * 1000

def add_hour_ceiling(df: pd.DataFrame | pd.Series, col_datetime: str = 'id_Datetime', return_series: bool = False) -> pd.DataFrame | pd.Series:

    """Add a column with the hourly ceiling of the datetime column."""

    df_copy = df.copy()
    df_copy[col_datetime] = pd.to_datetime(df_copy[col_datetime])
    
    if not return_series:
        df_copy['ceiling_hour'] = df_copy[col_datetime].dt.ceil('h')
        return df_copy
    else:
        return df_copy[col_datetime].dt.ceil('h')
    
# add time features
def add_time_cyclical(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    t = pd.to_datetime(df[ts_col])
    # Localize to avoid DST issues later
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam")
    else:
        t = t.dt.tz_convert("Europe/Amsterdam")
    out = df.copy()

    out["second"]    = t.dt.second
    out["minute"]    = t.dt.minute
    out["hour"]      = t.dt.hour
    out["dow"]       = t.dt.dayofweek               
    out["month"]     = t.dt.month

    out["second_sin"] = np.sin(2*np.pi*out["second"]/60); out["second_cos"] = np.cos(2*np.pi*out["second"]/60)
    out["minute_sin"] = np.sin(2*np.pi*out["minute"]/60); out["minute_cos"] = np.cos(2*np.pi*out["minute"]/60)
    out["hour_sin"]  = np.sin(2*np.pi*out["hour"]/24); out["hour_cos"]  = np.cos(2*np.pi*out["hour"]/24)
    out["dow_sin"]   = np.sin(2*np.pi*out["dow"]/7);   out["dow_cos"]   = np.cos(2*np.pi*out["dow"]/7)
    out["month_sin"] = np.sin(2*np.pi*(out["month"]-1)/12); out["month_cos"] = np.cos(2*np.pi*(out["month"]-1)/12)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out



def add_fourier(df, time_series: pd.Series, period: float, K_harmonics: int, prefix: str) -> pd.DataFrame:
    # t = position within the period [0, P)
    phi = 2*np.pi * (time_series.values[:, None] / period)      # type: ignore
    for k in range(1, K_harmonics+1):
        df[f"F_{prefix}_sin{k}"] = np.sin(k*phi).ravel()
        df[f"F_{prefix}_cos{k}"] = np.cos(k*phi).ravel()
    return df

def add_bus_time_fourier(df: pd.DataFrame, time_col: str) -> pd.DataFrame:

    """Add Fourier series features to a DataFrame based on a time column."""

    ts = pd.to_datetime(df[time_col])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("Europe/Amsterdam")
    else:
        ts = ts.dt.tz_convert("Europe/Amsterdam")

    # Daily clock (hours with fraction)
    t_day = ts.dt.hour + ts.dt.minute/60 + ts.dt.second/3600
    df = add_fourier(df, t_day, period=24.0, K_harmonics=2, prefix="day")

    # Weekly clock (hour-of-week)
    t_week = ts.dt.dayofweek*24 + t_day
    df = add_fourier(df, t_week, period=168.0, K_harmonics=2, prefix="week")

    # Yearly clock (day-of-year with fraction; handle leap year)
    day_of_year = (ts.dt.dayofyear - 1) + t_day/24.0
    P_year = 365.2425
    
    df = add_fourier(df, day_of_year, period=P_year, K_harmonics=3, prefix="year")  # type: ignore

    return df


def add_lags_rolls(df: pd.DataFrame, time_series_col: str, y_col_name: str, group_col=None, lags=[1,24,24*7], windows=[24,24*7]):

    """Add lag and rolling features to a time series DataFrame."""

    # Convert y_col to quantity seconds if not already numeric
    if not pd.api.types.is_numeric_dtype(df[y_col_name]):
        df[y_col_name + '_quant'] = pd.to_timedelta(df[y_col_name]).dt.total_seconds()
        y_col_name = y_col_name + '_quant'

    df = df.sort_values(([group_col] if group_col else []) + [time_series_col]).copy()
    if group_col:
        for L in lags:
            df[f"{y_col_name}_lag{L}"] = df.groupby(group_col, observed=True)[y_col_name].shift(L)
        for W in windows:
            df[f"{y_col_name}_rollmean{W}"] = df.groupby(group_col, observed=True)[y_col_name].transform(
                lambda x: x.shift(1).rolling(W, min_periods=max(3, W//4)).mean()
            )
            df[f"{y_col_name}_rollstd{W}"] = df.groupby(group_col, observed=True)[y_col_name].transform(
                lambda x: x.shift(1).rolling(W, min_periods=max(3, W//4)).std()
            )
    else:
        for L in lags:
            df[f"{y_col_name}_lag{L}"] = df[y_col_name].shift(L)
        for W in windows:
            base = df[y_col_name].shift(1)
            df[f"{y_col_name}_rollmean{W}"] = base.rolling(W, min_periods=max(3, W//4)).mean()
            df[f"{y_col_name}_rollstd{W}"] = base.rolling(W, min_periods=max(3, W//4)).std()
    return df

def add_lags_rolls_with_reliability(
    df: pd.DataFrame,
    *,
    time_series_col: str,                               # timestamp column
    y_col_name: str,                                    # target/series to lag/roll
    group_col: str | list[str] | None = None,           # e.g., "line_dir_code" or ["line","dir"]
    lags: tuple[int, ...] = (1, 24, 168),               # tweak to your seasonality
    roll_windows: tuple[int, ...] = (24, 168),
    horizon: int = 1,                                   # label at t + horizon
) -> pd.DataFrame:
    """Adds lags/rolling features + reliability indicators (past-only, per group).
    
    Added columns:
    - ind_miss_lag{L}: 1 if that lag is missing (start of series / gaps).
    - roll{W}_n: how many past points were in the rolling window (integer).
    - roll{W}_cov: window coverage ratio in [0,1] (1.0 = full window).
    - hist_n: number of prior observations seen (per group) — grows over time.
    - age_since_prev_h: hours since last observation (helps with irregular sampling).
    - lag{L}_age_h: how old the lagged timestamp is in hours.
    - ind_warmup: 1 during the initial period where full history is not available.

    
    """

    # Convert y_col to quantity seconds if not already numeric
    if not pd.api.types.is_numeric_dtype(df[y_col_name]):
        df[y_col_name + '_quant'] = pd.to_timedelta(df[y_col_name]).dt.total_seconds()
        y_col_name = y_col_name + '_quant'

    keys = [group_col] if isinstance(group_col, str) else (group_col or [])
    out = df.sort_values(keys + [time_series_col]).copy()

    # Parse timestamps once
    ts = pd.to_datetime(out[time_series_col])

    # Group helper
    g = out.groupby(keys, observed=True) if keys else None
    def _shift(series: pd.Series, k: int) -> pd.Series:
        return (g[series.name].shift(k) if g is not None else series.shift(k))          # type: ignore

    # --- History depth & gap indicators ---
    # number of past observations seen before current row (per group)
    out["hist_n"] = (g.cumcount() if g is not None
                     else pd.Series(np.arange(len(out)), index=out.index)).astype("int32")
    # time since previous observation (useful if timestamps aren’t perfectly regular)
    prev_ts = _shift(ts.rename(time_series_col), 1)
    out["age_since_prev_h"] = ((ts - prev_ts).dt.total_seconds() / 3600).astype("float32")

    # --- Lags + reliability flags ---
    for L in lags:
        lag = _shift(out[y_col_name], L)
        out[f"{y_col_name}_lag{L}"] = lag
        out[f"ind_miss_lag{L}"] = lag.isna().astype("int8")

        # optional: how old is that lag in hours (helps on irregular sampling)
        lag_ts = _shift(ts.rename(time_series_col), L)
        out[f"lag{L}_age_h"] = ((ts - lag_ts).dt.total_seconds() / 3600).astype("float32")

    # --- Rolling stats on past-only series (+ coverage indicators) ---
    base = _shift(out[y_col_name], 1)  # ends at t-1 → no leakage
    for W in roll_windows:
        r = base.rolling(W, min_periods=1)  # keep early rows; expose coverage below
        out[f"{y_col_name}_rollmean{W}"] = r.mean().astype("float32")
        out[f"{y_col_name}_rollstd{W}"]  = r.std().astype("float32")
        n = r.count().astype("int16")
        out[f"roll{W}_n"]   = n                          # how many points contributed
        out[f"roll{W}_cov"] = (n / W).clip(0, 1).astype("float32")  # coverage ratio in [0,1]

    # --- Warm-up indicator (too little history for full feature set) ---
    max_lag = max(lags) if lags else 0
    max_win = max(roll_windows) if roll_windows else 0
    warmup = max(max_lag, max_win)
    out["ind_warmup"] = (out["hist_n"] < warmup).astype("int8")

    # --- Future label at t + horizon (for training) ---
    fut = (g[y_col_name].shift(-horizon) if g is not None else out[y_col_name].shift(-horizon))
    out[f"{y_col_name}_t+{horizon}"] = fut

    return out


# Create direction column & composite direction/line key
def add_direction_columns(df: pd.DataFrame, col_start: str, col_stop: str, col_line: str) -> pd.DataFrame:

    """Encode directional information into new columns direction & composite direction/line key"""

    df['direction'] = df[col_start] + '_to_' + df[col_stop]
    df['line_key'] = df[col_line].astype(str) + '_' + df['direction']
    df['direction'] = df['direction'].astype('category')
    df['line_key'] = df['line_key'].astype('category')
    return df

def add_total_seconds(df: pd.DataFrame, time_cols: list[str]) -> pd.DataFrame | pd.Series:
    """Convert a timedelta column to total seconds."""

    for col in time_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col + '_quant'] = pd.to_timedelta(df[col]).dt.total_seconds()
    return df