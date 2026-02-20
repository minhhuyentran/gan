import os

from src.utils.config import load_config, ensure_dirs
from src.utils.io import load_json, save_pkl, save_json
from src.data.load import load_all_days
from src.data.preprocess import clean_df, clip_outliers, fit_scaler, apply_scaler, add_time_features
from src.data.split import time_days_split
from src.data.cond import build_cond

def main():
    cfg = load_config("configs/base.yaml")
    out_dir = cfg["paths"]["out_dir"]
    art_dir = cfg["paths"]["artifacts_dir"]
    ensure_dirs(out_dir, art_dir)

    schema = load_json(os.path.join(art_dir, "schema.json"))

    df = load_all_days(cfg["paths"]["raw_dir"])
    df = add_time_features(df, cfg["data"]["timestamp_col"])
    df = clean_df(df)

    # drop id cols from model features
    for c in cfg["data"]["drop_cols"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    cont_cols = schema["continuous"] + schema["count_like"]
    cont_cols = [c for c in cont_cols if c in df.columns]

    df = clip_outliers(df, cont_cols, q=cfg["preprocess"]["clip_quantile"])

    # split by days
    train_df, val_df, test_df = time_days_split(
        df,
        cfg["split"]["train_days"],
        cfg["split"]["val_days"],
        cfg["split"]["test_days"]
    )

    # sanity checks to avoid silent empty splits
    assert len(train_df) > 0, "Train split is empty. Check day_name values / split config."
    assert len(val_df) > 0, "Val split is empty. Check day_name values / split config."
    assert len(test_df) > 0, "Test split is empty. Check day_name values / split config."


    # Fit scaler on TRAIN ONLY
    scaler = fit_scaler(train_df, cont_cols, scaler_kind=cfg["preprocess"]["scaler"])
    train_s = apply_scaler(train_df, cont_cols, scaler)
    val_s = apply_scaler(val_df, cont_cols, scaler)
    test_s = apply_scaler(test_df, cont_cols, scaler)

    # Build conditioning vectors
    c_train, meta = build_cond(
        train_s,
        protocol_col=cfg["data"]["protocol_col"],
        dst_port_col=cfg["data"]["dst_port_col"],
        use_protocol=cfg["gan"]["cond"]["use_protocol"],
        use_service_bucket=cfg["gan"]["cond"]["use_service_bucket"],
        use_time_window=cfg["gan"]["cond"]["use_time_window"],
    )
    c_val, _ = build_cond(val_s, cfg["data"]["protocol_col"], cfg["data"]["dst_port_col"],
                          cfg["gan"]["cond"]["use_protocol"], cfg["gan"]["cond"]["use_service_bucket"], cfg["gan"]["cond"]["use_time_window"])
    c_test, _ = build_cond(test_s, cfg["data"]["protocol_col"], cfg["data"]["dst_port_col"],
                           cfg["gan"]["cond"]["use_protocol"], cfg["gan"]["cond"]["use_service_bucket"], cfg["gan"]["cond"]["use_time_window"])

    save_pkl(scaler, os.path.join(art_dir, "scaler.pkl"))
    save_json({"cont_cols": cont_cols, "cond_meta": meta}, os.path.join(art_dir, "preprocess_meta.json"))

    # Save processed splits (pkl for speed)
    save_pkl((train_s, c_train), os.path.join(art_dir, "train.pkl"))
    save_pkl((val_s, c_val), os.path.join(art_dir, "val.pkl"))
    save_pkl((test_s, c_test), os.path.join(art_dir, "test.pkl"))

    print("Saved artifacts in:", art_dir)

if __name__ == "__main__":
    main()
