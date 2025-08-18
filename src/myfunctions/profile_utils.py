import pandas as pd
from ydata_profiling import ProfileReport
import logging
from typing import Dict
import webbrowser
import os

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