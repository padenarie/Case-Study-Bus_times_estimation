import pandas as pd
from typing import Callable



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