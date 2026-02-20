import numpy as np
import pandas as pd

def enforce_nonneg(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].clip(lower=0.0)
    return out

def round_counts(df: pd.DataFrame, count_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in count_cols:
        out[c] = np.rint(out[c]).astype(int)
        out[c] = out[c].clip(lower=0)
    return out

def enforce_min_mean_max(df: pd.DataFrame, groups: list[tuple[str, str, str]]) -> pd.DataFrame:
    """
    groups: list of (min_col, mean_col, max_col)
    """
    out = df.copy()
    for mn, mean, mx in groups:
        a = out[mn].to_numpy()
        b = out[mean].to_numpy()
        c = out[mx].to_numpy()
        stacked = np.vstack([a, b, c]).T
        stacked.sort(axis=1)
        out[mn] = stacked[:, 0]
        out[mean] = stacked[:, 1]
        out[mx] = stacked[:, 2]
    return out

def violation_rate_nonneg(df: pd.DataFrame, cols: list[str]) -> float:
    if len(cols) == 0:
        return 0.0
    v = (df[cols] < 0).any(axis=1).mean()
    return float(v)
