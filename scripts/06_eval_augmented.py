import os
import numpy as np
import pandas as pd

from src.utils.config import load_config, ensure_dirs
from src.utils.io import load_pkl, load_json
from src.models.detectors import ocsvm, iforest
from src.models.detectors.ae import train_ae, score_ae
from src.eval.metrics import anomaly_metrics, threshold_at_fpr, fpr_tpr_at_threshold

def to_xy(df, schema, label_col, benign_label):
    feat_cols = schema["continuous"] + schema["count_like"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].to_numpy().astype(np.float32)
    y = (df[label_col].astype(str).values != benign_label).astype(int)
    return X, y, feat_cols

def main():
    cfg = load_config("configs/base.yaml")
    out_dir = cfg["paths"]["out_dir"]
    art_dir = cfg["paths"]["artifacts_dir"]
    ensure_dirs(out_dir, art_dir)

    schema = load_json(os.path.join(art_dir, "schema.json"))
    meta = load_json(os.path.join(art_dir, "preprocess_meta.json"))
    cont_cols = meta["cont_cols"]
    scaler = None  # không cần ở đây vì train/val/test đã scaled rồi

    train_df, _ = load_pkl(os.path.join(art_dir, "train.pkl"))
    val_df, _ = load_pkl(os.path.join(art_dir, "val.pkl"))
    test_df, _ = load_pkl(os.path.join(art_dir, "test.pkl"))

    # load synthetic (raw scale) -> nhưng detector đang chạy trên scaled features.
    # => quick approach: chỉ dùng synthetic như "raw demo".
    # Chuẩn paper: bạn nên generate synth ở scaled space rồi match preprocessing 1:1.
    synth_path = os.path.join(out_dir, "synthetic_benign_v1.csv")
    synth_raw = pd.read_csv(synth_path)

    # For now: scale synth using TRAIN scaler (bạn có thể mở rộng sau)
    # Nếu bạn muốn chuẩn hoá luôn synth, mình sẽ hướng dẫn bước thêm.
    # (Ở đây tạm bỏ qua để tránh dài.)

    label_col = cfg["data"]["label_col"]
    benign_label = cfg["data"]["benign_label"]

    X_train, y_train, feat_cols = to_xy(train_df, schema, label_col, benign_label)
    X_val, y_val, _ = to_xy(val_df, schema, label_col, benign_label)
    X_test, y_test, _ = to_xy(test_df, schema, label_col, benign_label)

    X_train_benign = X_train[y_train == 0]

    rows = []

    m1 = ocsvm.train(X_train_benign, nu=0.05)
    s_val = ocsvm.score(m1, X_val)
    thr = threshold_at_fpr(y_val, s_val, target_fpr=0.01)
    s_test = ocsvm.score(m1, X_test)
    rows.append({"model": "OCSVM(no-aug)", **anomaly_metrics(y_test, -s_test), **fpr_tpr_at_threshold(y_test, s_test, thr)})

    m2 = iforest.train(X_train_benign, contamination=0.05, random_state=cfg["seed"])
    s_val = iforest.score(m2, X_val)
    thr = threshold_at_fpr(y_val, s_val, target_fpr=0.01)
    s_test = iforest.score(m2, X_test)
    rows.append({"model": "IForest(no-aug)", **anomaly_metrics(y_test, -s_test), **fpr_tpr_at_threshold(y_test, s_test, thr)})

    device = "cpu"
    m3 = train_ae(X_train_benign, epochs=10, device=device)
    s_val = score_ae(m3, X_val, device=device)
    thr = threshold_at_fpr(y_val, s_val, target_fpr=0.01)
    s_test = score_ae(m3, X_test, device=device)
    rows.append({"model": "AE(no-aug)", **anomaly_metrics(y_test, -s_test), **fpr_tpr_at_threshold(y_test, s_test, thr)})

    out = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "eval_no_aug.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print(out)

if __name__ == "__main__":
    main()
