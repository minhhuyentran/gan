import json
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class Schema:
    continuous: list[str]
    categorical: list[str]
    count_like: list[str]
    label_col: str
    timestamp_col: str
    protocol_col: str

def infer_schema(df: pd.DataFrame, label_col: str, timestamp_col: str, protocol_col: str, drop_cols: list[str]) -> Schema:
    cols = [c for c in df.columns if c not in drop_cols]
    # label/timestamp/protocol tách riêng
    base_exclude = {label_col, timestamp_col, protocol_col}
    feature_cols = [c for c in cols if c not in base_exclude]

    # simple heuristic: numeric -> continuous, object -> categorical
    categorical = [c for c in feature_cols if df[c].dtype == "object"]
    numeric = [c for c in feature_cols if c not in categorical]

    # count-like heuristic: contains "Packet", "Count", "Bytes", "Total"
    count_like = [c for c in numeric if any(k in c.lower() for k in ["packet", "count", "bytes", "total"])]
    continuous = [c for c in numeric if c not in count_like]

    # protocol thường numeric (6/17) hoặc string; bạn đưa vào categorical ở cond vector
    return Schema(continuous=continuous, categorical=categorical, count_like=count_like,
                  label_col=label_col, timestamp_col=timestamp_col, protocol_col=protocol_col)

def save_schema(schema: Schema, path: str):
    with open(path, "w") as f:
        json.dump(asdict(schema), f, indent=2)
