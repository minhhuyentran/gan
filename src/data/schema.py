import pandas as pd

def infer_schema(df: pd.DataFrame, label_col: str, timestamp_col: str, protocol_col: str, drop_cols: list[str]) -> dict:
    # columns we keep for modeling (excluding id columns + label/timestamp/protocol)
    keep_cols = [c for c in df.columns if c not in drop_cols]
    exclude = {label_col, timestamp_col, protocol_col}
    feat_cols = [c for c in keep_cols if c not in exclude]

    categorical = []
    numeric = []
    for c in feat_cols:
        if df[c].dtype == "object":
            categorical.append(c)
        else:
            numeric.append(c)

    # heuristic for count-like numeric columns
    count_like = []
    for c in numeric:
        lc = c.lower()
        if any(k in lc for k in ["packet", "bytes", "count", "total"]):
            count_like.append(c)

    continuous = [c for c in numeric if c not in count_like]

    schema = {
        "continuous": continuous,
        "categorical": categorical,
        "count_like": count_like,
        "label_col": label_col,
        "timestamp_col": timestamp_col,
        "protocol_col": protocol_col
    }
    return schema
