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

    out["hour"]  = t.dt.hour
    out["dow"]   = t.dt.dayofweek               
    out["month"] = t.dt.month

    out["dow_sin"]   = np.sin(2*np.pi*out["dow"]/7);   out["dow_cos"]   = np.cos(2*np.pi*out["dow"]/7)
    out["hour_sin"]  = np.sin(2*np.pi*out["hour"]/24); out["hour_cos"]  = np.cos(2*np.pi*out["hour"]/24)
    out["month_sin"] = np.sin(2*np.pi*(out["month"]-1)/12); out["month_cos"] = np.cos(2*np.pi*(out["month"]-1)/12)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def add_fourier(df: pd.DataFrame, col_cyclic_time: str, period: float, K: int, prefix: str) -> pd.DataFrame:

    """Add Fourier series features to a DataFrame based on a time column."""

    phi = 2*np.pi*df[col_cyclic_time].values[:, None] / period # type: ignore
    for k in range(1, K+1):
        df[f"{prefix}_sin{k}"] = np.sin(k*phi).ravel()
        df[f"{prefix}_cos{k}"] = np.cos(k*phi).ravel()
    return df


def add_lags_rolls(df: pd.DataFrame, time_series_col: str, y_col_name: str, group=None, lags=[1,24,24*7], windows=[24,24*7]):

    """Add lag and rolling features to a time series DataFrame."""

    # Convert y_col to quantity seconds if not already numeric
    if not pd.api.types.is_numeric_dtype(df[y_col_name]):
        df[y_col_name + '_quant'] = pd.to_timedelta(df[y_col_name]).dt.total_seconds()
        y_col_name = y_col_name + '_quant'

    df = df.sort_values(([group] if group else []) + [time_series_col]).copy()
    if group:
        for L in lags:
            df[f"{y_col_name}_lag{L}"] = df.groupby(group)[y_col_name].shift(L)
        for W in windows:
            df[f"{y_col_name}_rollmean{W}"] = df.groupby(group)[y_col_name].transform(
                lambda x: x.shift(1).rolling(W, min_periods=max(3, W//4)).mean()
            )
            df[f"{y_col_name}_rollstd{W}"] = df.groupby(group)[y_col_name].transform(
                lambda x: x.shift(1).rolling(W, min_periods=max(3, W//4)).std()
            )
    else:
        for L in lags:
            df[f"{y_col_name}_lag{L}"] = df[y_col_name].shift(L)
        for W in windows:
            base = df[y_col_name].shift(1)
            df[f"{y_col_name}_rollmean{W}"] = base.rolling(W, min_periods=max(3, W//4)).mean()
            df[f"{y_col_name}_rollstd{W}"] = base.rolling(W, min_periods=max(3, W//4)).std()


# Create direction column & composite direction/line key
def add_direction_column(df: pd.DataFrame, col_start: str, col_stop: str, col_line: str) -> pd.DataFrame:

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