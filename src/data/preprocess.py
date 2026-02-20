import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(0.0)
    return out

def clip_outliers(df: pd.DataFrame, cols: list[str], q: float = 0.999) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        hi = out[c].quantile(q)
        lo = out[c].quantile(1 - q)
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out

def fit_scaler(df: pd.DataFrame, cols: list[str], scaler_kind: str = "standard"):
    if scaler_kind == "standard":
        scaler = StandardScaler()
    elif scaler_kind == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_kind: {scaler_kind}")
    scaler.fit(df[cols].values)
    return scaler

def apply_scaler(df: pd.DataFrame, cols: list[str], scaler) -> pd.DataFrame:
    out = df.copy()
    out[cols] = scaler.transform(out[cols].values)
    return out

def inverse_scaler(df: pd.DataFrame, cols: list[str], scaler) -> pd.DataFrame:
    out = df.copy()
    out[cols] = scaler.inverse_transform(out[cols].values)
    return out

def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    out["hour"] = ts.dt.hour.fillna(0).astype(int)
    out["day_name"] = ts.dt.day_name().fillna("Unknown")
    return out
