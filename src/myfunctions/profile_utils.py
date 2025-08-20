import pandas as pd
from ydata_profiling import ProfileReport
import logging
from typing import Dict
import webbrowser
import os
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def generate_profiling_reports(dfs: Dict, title: str = "Data Profiling Report", timeseries: bool = False, **kwargs) -> Dict[str, ProfileReport]:

    profiles : Dict[str, ProfileReport] = {}
    for df_name, df in dfs.items():
        # Generate Profiling reports
        try:
            if timeseries:
                profiles[df_name] = ProfileReport(df.convert_dtypes(dtype_backend="numpy_nullable"), title=df_name + " Data Profiling Time-Series Report", tsmode=True, **kwargs)
            else:
                profiles[df_name] = ProfileReport(df.convert_dtypes(dtype_backend="numpy_nullable"), title=df_name + " Data Profiling Report", **kwargs)
            logging.info(f"Generated profiling report for {df_name}")
        except Exception as e:
            logging.error(f"Error generating profiling report for {df_name}: {e}")

    return profiles


def display_profiling_reports_web(profiles: Dict[str, ProfileReport], refresh_results : bool = False) -> None:

    REPORTS : str = "Reports"
    PROFILING : str = REPORTS + "/Profiling"

    for df_name, profile in profiles.items():
        # Disable progress bar
        profile.config.progress_bar = False

        # Check if HTML report was previously generated
        if not os.path.exists(PROFILING+ f"/{df_name}_profile.html") or refresh_results:
            try:
                profile.to_file(PROFILING+ f"/{df_name}_profile.html")
                logging.info(f"Saved profiling report for {df_name}")
            except Exception as e:
                logging.error(f"Error saving profiling report for {df_name}: {e}")
                continue

        # Open Profile reports
        try:
            webbrowser.open(os.path.abspath(PROFILING+ f"/{df_name}_profile.html"))
            logging.info(f"Opened profiling report for {df_name}")
        except Exception as e:
            logging.error(f"Error opening profiling report for {df_name}: {e}")


def plot_corr_heatmap(
    df,
    columns=None,
    method='pearson',
    min_value_threshold=None,
    figsize=(10,8),
    cmap='coolwarm',
    annot=True,
    y_axis_columns=None
    ):
    """
    Display a correlation matrix (Pearson or Spearman) as a heatmap for a subset of columns.
    Columns are omitted if all values are below min_value_threshold.
    If a column causes an error, it is omitted and the result is shown without that column.
    Only columns in y_axis_columns are shown on the y-axis if provided.
    """
    logger = logging.getLogger("plot_corr_heatmap")
    if columns is None:
        columns = df.columns.tolist()
    valid_cols = []
    for col in columns:
        try:
            col_data = pd.to_numeric(df[col], errors='raise')
            if min_value_threshold is not None:
                if (col_data > min_value_threshold).any():
                    valid_cols.append(col)
                else:
                    logger.info(f"Omitting column '{col}' (all values below threshold {min_value_threshold})")
            else:
                valid_cols.append(col)
        except Exception as e:
            logger.warning(f"Omitting column '{col}' due to error: {e}")
    if not valid_cols:
        logger.error("No valid columns for correlation heatmap.")
        return None
    try:
        corr = df[valid_cols].corr(method=method)
        # Filter y-axis if requested
        if y_axis_columns is not None:
            y_axis_valid = [col for col in y_axis_columns if col in corr.index]
            if not y_axis_valid:
                logger.error("No valid y_axis_columns found in correlation matrix.")
                return None
            corr = corr.loc[y_axis_valid]
        plt.figure(figsize=figsize)
        sns.heatmap(corr, cmap=cmap, annot=annot, fmt='.2f', square=True)
        plt.title(f'{method.capitalize()} Correlation Matrix Heatmap')
        plt.show()
    except Exception as e:
        logger.error(f"Failed to plot correlation heatmap: {e}")
        return None