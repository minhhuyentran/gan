import os
import numpy as np
import pandas as pd

import time
from datetime import datetime


from src.utils.config import load_config, ensure_dirs
from src.utils.io import load_pkl, load_json
from src.models.detectors import ocsvm, iforest
from src.models.detectors.ae import train_ae, score_ae
from src.eval.metrics import anomaly_metrics, threshold_at_fpr, fpr_tpr_at_threshold

def to_xy(df, schema, label_col, benign_label):
    # Features for detectors: only numeric features (continuous+count_like), already scaled
    feat_cols = schema["continuous"] + schema["count_like"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].to_numpy().astype(np.float32)

    y = (df[label_col].astype(str).values != benign_label).astype(int)  # 1 anomaly
    return X, y

def main():
    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    cfg = load_config("configs/base.yaml")
    out_dir = cfg["paths"]["out_dir"]
    art_dir = cfg["paths"]["artifacts_dir"]
    ensure_dirs(out_dir, art_dir)

    log("Loading artifacts...")
    schema = load_json(os.path.join(art_dir, "schema.json"))
    train_df, _ = load_pkl(os.path.join(art_dir, "train.pkl"))
    val_df, _ = load_pkl(os.path.join(art_dir, "val.pkl"))
    test_df, _ = load_pkl(os.path.join(art_dir, "test.pkl"))

    label_col = cfg["data"]["label_col"]
    benign_label = cfg["data"]["benign_label"]

    log("Preparing X/y ...")
    X_train, y_train = to_xy(train_df, schema, label_col, benign_label)
    X_val, y_val = to_xy(val_df, schema, label_col, benign_label)
    X_test, y_test = to_xy(test_df, schema, label_col, benign_label)

    # Train only on benign rows for anomaly detectors
    X_train_benign = X_train[y_train == 0]

    # --- speed guard: OCSVM (RBF) is too slow on hundreds of thousands samples
    max_benign_for_ocsvm = 50000  # try 50k; you can set 30000 if still slow
    rng = np.random.RandomState(cfg["seed"])
    if X_train_benign.shape[0] > max_benign_for_ocsvm:
        idx = rng.choice(X_train_benign.shape[0], max_benign_for_ocsvm, replace=False)
        X_train_benign_ocsvm = X_train_benign[idx]
    else:
        X_train_benign_ocsvm = X_train_benign


    log(f"Train benign size: {X_train_benign.shape}")

    rows = []

    # OCSVM
    t0 = time.time()
    log("Training OCSVM...")

    # m1 = ocsvm.train(X_train_benign, nu=0.05)
    m1 = ocsvm.train(X_train_benign_ocsvm, nu=0.05)
    log(f"OCSVM training samples: {X_train_benign_ocsvm.shape}")

    log(f"OCSVM trained in {time.time()-t0:.1f}s. Scoring val/test...")

    s_val = ocsvm.score(m1, X_val)       # normal score
    thr = threshold_at_fpr(y_val, s_val, target_fpr=0.01)
    s_test = ocsvm.score(m1, X_test)
    log("OCSVM scoring done.")

    met = anomaly_metrics(y_test, -s_test)  # convert to anomaly score
    ft = fpr_tpr_at_threshold(y_test, s_test, thr)
    rows.append({"model": "OCSVM", **met, **ft, "thr_normscore@FPR1%": thr})

    # IForest
    t0 = time.time()
    log("Training IsolationForest...")

    # m2 = iforest.train(X_train_benign, contamination=0.05, random_state=cfg["seed"])
    m2 = iforest.train(X_train_benign, contamination=0.05, random_state=cfg["seed"])

    log(f"IForest trained in {time.time()-t0:.1f}s. Scoring val/test...")

    s_val = iforest.score(m2, X_val)
    thr = threshold_at_fpr(y_val, s_val, target_fpr=0.01)
    s_test = iforest.score(m2, X_test)
    log("IForest scoring done.")

    met = anomaly_metrics(y_test, -s_test)
    ft = fpr_tpr_at_threshold(y_test, s_test, thr)
    rows.append({"model": "IForest", **met, **ft, "thr_normscore@FPR1%": thr})

    # AE
    t0 = time.time()
    log("Training Autoencoder (AE)...")

    device = "cuda" if False else "cpu"
    # m3 = train_ae(X_train_benign, epochs=10, device=device)
    m3 = train_ae(X_train_benign, epochs=3, device=device)

    log(f"AE trained in {time.time()-t0:.1f}s. Scoring val/test...")

    s_val = score_ae(m3, X_val, device=device)
    thr = threshold_at_fpr(y_val, s_val, target_fpr=0.01)
    s_test = score_ae(m3, X_test, device=device)
    log("AE scoring done.")

    met = anomaly_metrics(y_test, -s_test)
    ft = fpr_tpr_at_threshold(y_test, s_test, thr)
    rows.append({"model": "AE", **met, **ft, "thr_normscore@FPR1%": thr})

    res = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "baseline_results.csv")
    res.to_csv(out_path, index=False)
    log("Saving baseline_results.csv ...")

    print("Saved:", out_path)
    print(res)

if __name__ == "__main__":
    main()
