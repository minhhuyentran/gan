import os
import pandas as pd

DAY_FILES = [
    ("Monday", "monday.csv"),
    ("Tuesday", "tuesday.csv"),
    ("Wednesday", "wednesday.csv"),
    ("Thursday", "thursday.csv"),
    ("Friday", "friday.csv"),
]

def load_all_days(raw_dir: str) -> pd.DataFrame:
    dfs = []
    for day_name, fname in DAY_FILES:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        df["day_name"] = day_name  # override by filename (stable)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)
