from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Dict, Iterable, Mapping, Any
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
    for key, df in dfs.items():
        steps = list(plan.get(key, []))
        if log:
            logging.info("▶ %s: %d steps", key, len(steps))
        for step in steps:
            if step.when and not step.when(df):
                if log:
                    logging.info("  - %s: skipped (condition not met)", step.name)
                continue
            t0, before = time.time(), df.shape
            try:
                df = df.pipe(step.fn, **step.kwargs)  # always DF -> DF
            except Exception as e:
                if log:
                    logging.error("  - %s: failed (%s)", step.name, e)
                    continue
            if log:
                dt = (time.time() - t0) * 1000  # ms
                logging.info("  - %s: %s -> %s (%d ms)", step.name, before, df.shape, dt)
        out[key] = df
    return out


# Predicates
def has_cols(*cols: str) -> Callable[[pd.DataFrame], bool]:
    return lambda df: set(cols).issubset(df.columns)

# # Example pipeline input
# pipeline = {
#     "check_ins": [
#         Step("add_feature", when=has_cols("id"), fn=partial(add_feature, src="id", dest='id_Datetime', function=parse_time_column))
#     ]
# }

# dfs_processed = run_pipeline(dfs, pipeline)