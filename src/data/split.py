import pandas as pd

def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    out["hour"] = ts.dt.hour.fillna(0).astype(int)
    out["day_name"] = ts.dt.day_name().fillna("Unknown")
    return out

def time_split(df: pd.DataFrame, train_days, val_days, test_days):
    train = df[df["day_name"].isin(train_days)].copy()
    val = df[df["day_name"].isin(val_days)].copy()
    test = df[df["day_name"].isin(test_days)].copy()
    return train, val, test
