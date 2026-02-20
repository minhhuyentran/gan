import numpy as np
import pandas as pd

def enforce_nonneg(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].clip(lower=0.0)
    return out

def enforce_monotonic(df: pd.DataFrame, triplets: list[tuple[str,str,str]]) -> pd.DataFrame:
    # (min_col, mean_col, max_col)
    out = df.copy()
    for mn, mean, mx in triplets:
        a = out[mn].values
        b = out[mean].values
        c = out[mx].values
        # sort each row’s trio
        stacked = np.vstack([a,b,c]).T
        stacked.sort(axis=1)
        out[mn] = stacked[:,0]
        out[mean] = stacked[:,1]
        out[mx] = stacked[:,2]
    return out

def round_counts(df: pd.DataFrame, count_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[count_cols] = np.rint(out[count_cols]).astype(int)
    out[count_cols] = out[count_cols].clip(lower=0)
    return out
