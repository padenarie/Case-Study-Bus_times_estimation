from __future__ import annotations
from pathlib import Path
import re
import logging
from typing import Dict
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def _slug(name: str) -> str:
    # safe identifier: letters/nums/underscore; no leading digit
    base = re.sub(r"\W+", "_", name).strip("_")
    return f"_{base}" if base and base[0].isdigit() else (base or "unnamed")

def load_csv_folder(
    folder: str | Path,
    *,
    sep: str = ";",
    encoding: str = "utf-8",
) -> Dict[str, pd.DataFrame]:
    """
    Load all CSVs in a folder into a dict mapping safe names -> DataFrames.
    """
    folder = Path(folder)
    dfs: Dict[str, pd.DataFrame] = {}

    for p in sorted(folder.glob("*.csv")):
        key = f"{_slug(p.stem)}"
        # avoid accidental overwrites
        suffix = 1
        while key in dfs:
            suffix += 1
            key = f"{key}_{suffix}"

        try:
            df = pd.read_csv(p, sep=sep, encoding=encoding, dtype_backend="pyarrow")
        except Exception as e:
            logging.error(f"Failed to load {p.name}: {e}")
            continue

        dfs[key] = df
        logging.info(f"Loaded {key} from {p.name} | shape={df.shape}")

    if not dfs:
        logging.warning(f"No CSV files found in {folder.resolve()}")
    return dfs
