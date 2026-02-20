import numpy as np
import pandas as pd

def random_split(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2, seed=42):
    assert abs(train_ratio + val_ratio - 1.0) < 1e-9 or (train_ratio + val_ratio) < 1.0
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0

    n = len(df)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy(), df.iloc[test_idx].copy()

def time_days_split(df: pd.DataFrame, train_days, val_days, test_days):
    train = df[df["day_name"].isin(train_days)].copy()
    val = df[df["day_name"].isin(val_days)].copy()
    test = df[df["day_name"].isin(test_days)].copy()
    return train, val, test
