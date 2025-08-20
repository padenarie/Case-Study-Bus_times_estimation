from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Iterable, Mapping, Any, Sequence
import logging
import pandas as pd
import time, logging
import numpy as np


### Pipeline runner
@dataclass(frozen=True)
class Step:
    name: str
    fn: Callable[[pd.DataFrame], pd.DataFrame]  # unified step signature
    kwargs: Dict[str, Any] = field(default_factory=dict)
    when: Callable[[pd.DataFrame], bool] | None = None


def run_pipeline(dfs: Dict[str, pd.DataFrame], plan: Mapping[str, Iterable[Step]], *, log: bool = True, feature_col: bool=False) -> Dict[str, pd.DataFrame]:
    """Apply per-DataFrame step sequences from `plan` and return new dict."""
    out: Dict[str, pd.DataFrame] = {}
    feature_columns: Dict[str, Dict[str, list]] = {}
    for key, df in dfs.items():
        steps = list(plan.get(key, []))
        if log:
            logging.info("▶ %s: %d steps", key, len(steps))
        feature_columns[key] = {}
        for step in steps:
            before_cols = set(df.columns)
            if step.when and not step.when(df):
                if log:
                    logging.info("  - %s: skipped (condition not met)", step.name)
                continue
            t0, before = time.time(), df.shape
            try:
                df = df.pipe(step.fn, **step.kwargs)  
            except Exception as e:
                if log:
                    logging.error("  - %s: failed (%s)", step.name, e)
                    continue
            after_cols = set(df.columns)
            new_cols = list(after_cols - before_cols)
            feature_columns[key][step.name] = new_cols
            if log:
                dt = (time.time() - t0) * 1000  # ms
                logging.info("  - %s: %s -> %s (%d ms)", step.name, before, df.shape, dt)
        out[key] = df
    if feature_col:
        out['feature_columns'] = feature_columns         # type: ignore
    return out


# Predicates
def has_cols(*cols: str) -> Callable[[pd.DataFrame], bool]:
    return lambda df: set(cols).issubset(df.columns)

def has_duplicates() -> Callable[[pd.DataFrame], bool]:
    return lambda df: bool(df.duplicated().any())

def is_timedelta_column(*cols: str) -> Callable[[pd.DataFrame], bool]:
    """Return True if the column is of timedelta dtype."""
    return lambda df: all(not pd.api.types.is_numeric_dtype(df[col]) for col in cols)


def inspect_duplicate_rows(df: pd.DataFrame, example_index: int) -> pd.DataFrame:
    """
    Inspect duplicate rows in a DataFrame by a single example.

    Args:
        df (pd.DataFrame): The DataFrame to inspect.
        example_index (int): The index of the example row to match against.

    Returns:
        pd.DataFrame: A DataFrame containing the duplicate rows.
    """
    return df[(df == df.iloc[example_index]).all(axis=1)]


def duplicate_count_per_row(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
    *,
    treat_na_as_equal: bool = True,
    return_series: bool = True
) -> pd.DataFrame|pd.Series:
    """
    For each original row, return the size of its duplicate group.

    Returns a Series aligned with `df.index`, or a appends the series as a new column to original dataframe.
    """
    cols = list(df.columns) if subset is None else list(subset)

    if treat_na_as_equal:
        sizes = df.groupby(cols, dropna=False)

        any_col = df.columns[0]
        result = sizes[any_col].transform('size')
        result.name = "duplicate_count"

    else:
        mask = df[cols].notna().all(axis=1)
        sizes = df.loc[mask, cols].groupby(cols, dropna=False)    

        any_col = df.columns[0]
        result = sizes[any_col].transform('size').fillna(1)
        result.name = "duplicate_count"
                
    if not return_series:
        df["duplicate_count"] = result
        return df
    else:
        return result

# function to remove duplicates
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame, keeping the first occurrence.

    Args:
        df (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The DataFrame with duplicates removed.
    """
    return df.drop_duplicates(keep='first', inplace=False)

# create date cutoff function 
def data_cutoff_dates(df: pd.DataFrame, col_datetime: str = 'id_Datetime', start_date: str = '2020-01-01', end_date: str = '2024-06-18', return_series: bool = False) -> pd.DataFrame | pd.Series:
    """Filter DataFrame by date range."""
    
    df_copy = df.copy()
    df_copy[col_datetime] = pd.to_datetime(df_copy[col_datetime])
    
    mask = (df_copy[col_datetime] >= start_date) & (df_copy[col_datetime] <= end_date)
    
    if not return_series:
        return df_copy[mask]
    else:
        return df_copy.loc[mask, col_datetime]

        
def merge_on_hour_ceiling (df_left: pd.DataFrame, df_right: pd.DataFrame, left_on: str = 'ceiling_hour', right_on: str = 'id_Datetime', how: str = 'left') -> pd.DataFrame:

    """Merge two DataFrames on the hourly ceiling of their datetime columns."""

    return pd.merge(df_left, df_right, left_on=left_on, right_on=right_on, how=how)          # type: ignore


def multiply_checkins(df: pd.DataFrame, column: str, factor: float) -> pd.DataFrame:
    """
    Multiply the check-ins in the DataFrame by a given factor.

    Args:
        df (pd.DataFrame): The input DataFrame containing check-in data.
        column (str): The name of the column to multiply.
        factor (float): The factor by which to multiply the check-ins.

    Returns:
        pd.DataFrame: A new DataFrame with the multiplied check-ins.
    """
    df = df.copy()
    df[column] *= factor
    return df



def attach_weighted_checkins_fast(
    df_trips: pd.DataFrame,
    df_checkins: pd.DataFrame,
    *,
    trip_start_col: str = "departure_time",     # datetime (naive EU/AMS or tz-aware)
    trip_duration_col: str = "trip_duration",   # Timedelta or 'HH:MM:SS'
    checkin_time_col: str = "when",             # hourly bin START (datetime)
    checkin_value_col: str = "checkins",        # hourly count
    out_col: str = "checkins_weighted",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Vectorized, DST-safe merge of network-wide hourly check-ins into per-trip rows.
    Adds:
      - out_col: weighted average check-ins during [start, start+duration)
      - f"{out_col}_coverage" in [0,1]: fraction of trip covered by available hourly bins

    Assumptions:
      - Naive datetimes are Europe/Amsterdam wall clock.
      - checkin_time_col marks the START of each local hour [H, H+1).
    """
    log = logger or logging.getLogger(__name__)
    if not log.handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ---- Robust EU/AMS localization that never errors on DST fall-back ----
    def localize_ams(series: pd.Series) -> pd.Series:
        s = pd.to_datetime(series, errors="coerce")
        if s.dt.tz is not None:
            return s.dt.tz_convert("Europe/Amsterdam")
        # 1) localize with ambiguous -> NaT (and spring-forward gaps shifted forward)
        s1 = s.dt.tz_localize("Europe/Amsterdam", ambiguous="NaT", nonexistent="shift_forward")
        amb_mask = s1.isna() & s.notna()  # rows that hit the duplicated hour
        if amb_mask.any():
            # 2) For ambiguous rows only, choose the DST instance (the *first* 02:00)
            #    In pandas, ambiguous=True => DST (summer time), ambiguous=False => Standard time
            s2 = s[amb_mask].dt.tz_localize("Europe/Amsterdam", ambiguous=True, nonexistent="shift_forward")    # type: ignore
            s1 = s1.where(~amb_mask, s2)
        return s1

    def to_utc(series: pd.Series) -> pd.Series:
        return localize_ams(series).dt.tz_convert("UTC")

    # ---- Normalize inputs ----
    trips  = df_trips.copy()
    checks = df_checkins.copy()

    # Trips: start/end in UTC
    trips[trip_start_col] = to_utc(trips[trip_start_col])
    dur_td = pd.to_timedelta(trips[trip_duration_col], errors="coerce")
    trips["__dur_s__"] = dur_td.dt.total_seconds()
    trips["__t_end__"] = trips[trip_start_col] + pd.to_timedelta(trips["__dur_s__"], unit="s")

    bad = trips["__dur_s__"].isna() | (trips["__dur_s__"] <= 0)
    if bad.any():
        log.warning("Found %d trips with invalid duration (NaT or <=0s); their weighted value will be NaN.",
                    int(bad.sum()))

    # Check-ins: aggregate duplicates per local hour, then to UTC and align to a full hourly grid
    checks[checkin_time_col] = to_utc(checks[checkin_time_col])
    checks = (checks
              .groupby(checkin_time_col, as_index=False, dropna=False)[checkin_value_col]
              .sum()
              .sort_values(checkin_time_col)) # type: ignore

    # Build continuous hourly grid that covers all trips (UTC → no DST ambiguity)
    start_h = trips[trip_start_col].dt.floor("h")
    end_h   = trips["__t_end__"].dt.floor("h")
    grid_start = start_h.min()
    grid_end   = end_h.max()

    if pd.isna(grid_start) or pd.isna(grid_end):
        out = trips.copy()
        out[out_col] = np.nan
        out[f"{out_col}_coverage"] = 0.0
        return out.drop(columns=["__dur_s__", "__t_end__"])

    idx = pd.date_range(grid_start, grid_end, freq="h", tz="UTC")

    # Reindex check-ins to grid (NaN where missing)
    ci = checks.set_index(checkin_time_col)[checkin_value_col].reindex(idx)

    # Prefix sums for fast integrals; NaNs treated as 0 in numerator; separate coverage mask
    C = ci.to_numpy(dtype=float)                   # hourly values
    mask = ~np.isnan(C)                            # hour has data?
    C_filled = np.where(mask, C, 0.0)
    pref_val = np.concatenate(([0.0], np.cumsum(C_filled)))            # length M+1
    pref_cov = np.concatenate(([0.0], np.cumsum(mask.astype(float))))  # length M+1

    # Map trips to grid positions/fractions
    ap = idx.get_indexer(start_h)   # start hour index
    bp = idx.get_indexer(end_h)     # end   hour index
    a_frac = ((trips[trip_start_col] - start_h).dt.total_seconds() / 3600.0).to_numpy(float)
    b_frac = ((trips["__t_end__"]      - end_h).dt.total_seconds() / 3600.0).to_numpy(float)
    dur_h  = (trips["__dur_s__"] / 3600.0).to_numpy(float)

    same = (ap == bp)
    cross = ~same

    num = np.zeros(len(trips), dtype=float)  # integral of check-ins over trip (checkin-hours)
    den = np.zeros(len(trips), dtype=float)  # integral of coverage weights over trip (hours)

    # Same-hour trips
    if same.any():
        pos = ap[same]
        exists = mask[pos]
        frac = dur_h[same]
        val = C[pos]
        num[same] = frac * np.where(exists, val, 0.0)
        den[same] = frac * exists.astype(float)

    # Cross-hour trips
    if cross.any():
        ap_c, bp_c = ap[cross], bp[cross]
        af, bf = a_frac[cross], b_frac[cross]

        # start hour
        s_exists = mask[ap_c]; s_val = C[ap_c]; s_frac = 1.0 - af
        s_num = s_frac * np.where(s_exists, s_val, 0.0)
        s_den = s_frac * s_exists.astype(float)

        # end hour
        e_exists = mask[bp_c]; e_val = C[bp_c]; e_frac = bf
        e_num = e_frac * np.where(e_exists, e_val, 0.0)
        e_den = e_frac * e_exists.astype(float)

        # interior full hours [ap+1, bp-1]
        interior = (bp_c - ap_c > 1)
        i_num = np.zeros_like(s_num); i_den = np.zeros_like(s_den)
        if interior.any():
            ap1 = ap_c[interior] + 1
            bp0 = bp_c[interior]
            i_num[interior] = pref_val[bp0] - pref_val[ap1]
            i_den[interior] = pref_cov[bp0] - pref_cov[ap1]

        num[cross] = s_num + i_num + e_num
        den[cross] = s_den + i_den + e_den

    # Weighted average & coverage
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = num / den
    coverage = np.divide(den, dur_h, out=np.zeros_like(den), where=np.isfinite(dur_h) & (dur_h > 0))

    out = trips.copy()
    out[out_col] = weighted
    out[f"{out_col}_coverage"] = np.clip(coverage, 0.0, 1.0)

    total = len(out); na_count = int(np.isnan(out[out_col]).sum())
    log.info(
        "Attached '%s' (vectorized, DST-safe). Coverage median=%.2f, mean=%.2f. NaNs in %s: %d/%d (%.1f%%).",
        out_col,
        float(pd.Series(coverage).median(skipna=True)),
        float(pd.Series(coverage).mean(skipna=True)),
        out_col, na_count, total, 100.0 * na_count / max(1, total)
    )

    return out.drop(columns=["__dur_s__", "__t_end__"])
