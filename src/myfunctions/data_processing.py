from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Iterable, Mapping, Any, Sequence
import logging
import pandas as pd
import time, logging


### Pipeline runner
@dataclass(frozen=True)
class Step:
    name: str
    fn: Callable[[pd.DataFrame], pd.DataFrame]  # unified step signature
    kwargs: Dict[str, Any] = field(default_factory=dict)
    when: Callable[[pd.DataFrame], bool] | None = None


def run_pipeline(dfs: Dict[str, pd.DataFrame], plan: Mapping[str, Iterable[Step]], *, log: bool = True) -> Dict[str, pd.DataFrame]:
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