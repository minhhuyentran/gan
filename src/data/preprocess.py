import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df

def clip_outliers(df: pd.DataFrame, cols: list[str], q: float=0.999) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        hi = out[c].quantile(q)
        lo = out[c].quantile(1-q)
        out[c] = out[c].clip(lower=lo, upper=hi)
    return out

def fit_scaler(df: pd.DataFrame, cols: list[str], kind: str="standard"):
    if kind == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    scaler.fit(df[cols].values)
    return scaler

def transform_scaler(df: pd.DataFrame, cols: list[str], scaler) -> pd.DataFrame:
    out = df.copy()
    out[cols] = scaler.transform(out[cols].values)
    return out

def save_artifact(obj, path: str):
    joblib.dump(obj, path)

def load_artifact(path: str):
    return joblib.load(path)
