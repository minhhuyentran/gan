import numpy as np
import pandas as pd

def protocol_to_onehot(proto_series: pd.Series):
    # handle numeric (6/17/1) or string
    # we map to 3 buckets: TCP/UDP/ICMP/OTHER
    mapped = []
    for v in proto_series.values:
        s = str(v).upper()
        if s in ["6", "TCP"]:
            mapped.append("TCP")
        elif s in ["17", "UDP"]:
            mapped.append("UDP")
        elif s in ["1", "ICMP"]:
            mapped.append("ICMP")
        else:
            mapped.append("OTHER")
    cats = ["TCP", "UDP", "ICMP", "OTHER"]
    onehot = np.zeros((len(mapped), len(cats)), dtype=np.float32)
    idx = {c:i for i,c in enumerate(cats)}
    for i,m in enumerate(mapped):
        onehot[i, idx[m]] = 1.0
    return onehot, cats

def service_bucket_from_port(port_series: pd.Series):
    # buckets: HTTP(80), HTTPS(443), DNS(53), OTHER
    buckets = []
    for p in port_series.values:
        try:
            p = int(p)
        except:
            p = -1
        if p == 80:
            buckets.append("HTTP")
        elif p == 443:
            buckets.append("HTTPS")
        elif p == 53:
            buckets.append("DNS")
        else:
            buckets.append("OTHER")
    cats = ["HTTP", "HTTPS", "DNS", "OTHER"]
    onehot = np.zeros((len(buckets), len(cats)), dtype=np.float32)
    idx = {c:i for i,c in enumerate(cats)}
    for i,b in enumerate(buckets):
        onehot[i, idx[b]] = 1.0
    return onehot, cats

def hour_onehot(hour_series: pd.Series):
    hours = hour_series.astype(int).clip(0, 23).values
    onehot = np.zeros((len(hours), 24), dtype=np.float32)
    onehot[np.arange(len(hours)), hours] = 1.0
    cats = [str(h) for h in range(24)]
    return onehot, cats

def build_cond(df: pd.DataFrame, protocol_col: str, dst_port_col: str, use_protocol=True, use_service_bucket=True, use_time_window=True):
    parts = []
    meta = {}

    if use_protocol:
        p_oh, p_cats = protocol_to_onehot(df[protocol_col])
        parts.append(p_oh)
        meta["protocol_cats"] = p_cats

    if use_service_bucket:
        s_oh, s_cats = service_bucket_from_port(df[dst_port_col])
        parts.append(s_oh)
        meta["service_cats"] = s_cats

    if use_time_window:
        h_oh, h_cats = hour_onehot(df["hour"])
        parts.append(h_oh)
        meta["hour_cats"] = h_cats

    if len(parts) == 0:
        c = np.zeros((len(df), 0), dtype=np.float32)
    else:
        c = np.concatenate(parts, axis=1).astype(np.float32)

    return c, meta
