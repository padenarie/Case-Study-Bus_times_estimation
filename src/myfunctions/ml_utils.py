# sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import numpy as np
import time

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def time_cv_score(df, pipeline, n_splits=5):
    require_columns(df, REQUIRED_RAW_FEATURES + [TARGET, TIMESTAMP_COL])
    df = df.sort_values(TIMESTAMP_COL)

    X = df[REQUIRED_RAW_FEATURES]
    y = df[TARGET].to_numpy()

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pipeline.fit(X_tr, y_tr)
        pred = pipeline.predict(X_te)
        scores.append(mean_absolute_error(y_te, pred))
    return scores

def mase(y, yhat, y_baseline):
    mae_b = mean_absolute_error(y, y_baseline)
    return np.inf if mae_b == 0 else mean_absolute_error(y, yhat) / mae_b

def summarize_errors(df, y_col, yhat_col, *,
                     baseline_col=None,
                     group_cols=None,
                     duration_col=None):
    """
    Compute error metrics in *minutes*. Inputs are assumed to be in *seconds*.
    Converts y, yhat, (optional) baseline and duration to minutes internally.

    Returns a dict with micro (overall), optional duration-weighted, and optional macro-by-group metrics.
    """
    # ---- convert seconds -> minutes (do not modify df) ----
    s2m = 1.0 / 60.0
    y_sec    = df[y_col].to_numpy()
    yhat_sec = df[yhat_col].to_numpy()
    y        = y_sec * s2m
    yhat     = yhat_sec * s2m

    # optional baseline in minutes
    y_base = None
    if baseline_col is not None:
        y_base = df[baseline_col].to_numpy() * s2m

    # optional duration (weights) in minutes
    w = None
    if duration_col is not None:
        w = df[duration_col].to_numpy() * s2m

    # ---- micro (overall) ----
    err = yhat - y
    out = {
        "MAE_min": mean_absolute_error(y, yhat),
        "RMSE_min": root_mean_squared_error(y, yhat),
        "MedianAE_min": float(np.median(np.abs(err))),
        "Bias_ME_min": float(np.mean(err)),
        "P90_|e|_min": float(np.quantile(np.abs(err), 0.90)),
        "Within_±1min": float(np.mean(np.abs(err) <= 1.0)),
        "Within_±2min": float(np.mean(np.abs(err) <= 2.0)),
    }
    if y_base is not None:
        out["MASE_vs_baseline"] = mase(y, yhat, y_base)  # unit cancels, still fine

    # ---- duration-weighted MAE (optional) ----
    if w is not None:
        # weights in minutes; scaling by constant would not change the average, but we pass minutes for clarity
        w = np.clip(w, 1e-9, None)
        out["MAE_duration_weighted_min"] = float(np.average(np.abs(err), weights=w))

    # ---- macro by groups (optional) ----
    if group_cols:
        g = (df.assign(abs_err_min=np.abs((df[yhat_col] - df[y_col]) * s2m))
               .groupby(group_cols, dropna=False, observed=True)["abs_err_min"].mean())
        out["MAE_macro_groups_min"] = float(g.mean())  # unweighted mean across groups

    return out


def fit_with_status(pipe, X, y, name):
    print(f"[{name}] fitting on {X.shape[0]} rows / {X.shape[1]} features…", flush=True)
    t0 = time.time()
    pipe.fit(X, y)
    dt = time.time() - t0
    print(f"[{name}] done in {dt:.2f}s", flush=True)

def predict_with_status(pipe, X, name):
    print(f"[{name}] predicting {X.shape[0]} rows…", flush=True)
    t0 = time.time()
    yhat = pipe.predict(X)
    dt = time.time() - t0
    print(f"[{name}] done in {dt:.2f}s", flush=True)
    return yhat