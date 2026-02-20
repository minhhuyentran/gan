import os
import numpy as np
import pandas as pd
import torch

from src.utils.config import load_config, ensure_dirs
from src.utils.io import load_pkl, load_json
from src.data.cond import build_cond
from src.data.preprocess import inverse_scaler
from src.models.gan.constraints import enforce_nonneg, round_counts, violation_rate_nonneg

def main():
    cfg = load_config("configs/base.yaml")
    out_dir = cfg["paths"]["out_dir"]
    art_dir = cfg["paths"]["artifacts_dir"]
    ensure_dirs(out_dir, art_dir)

    schema = load_json(os.path.join(art_dir, "schema.json"))
    meta = load_json(os.path.join(art_dir, "preprocess_meta.json"))
    cont_cols = meta["cont_cols"]

    scaler = load_pkl(os.path.join(art_dir, "scaler.pkl"))
    train_df, _ = load_pkl(os.path.join(art_dir, "train.pkl"))

    # load GAN checkpoint
    ckpt = torch.load(os.path.join(out_dir, "gan_cwgan_gp.pt"), map_location="cpu")
    feat_cols = ckpt["feat_cols"]
    x_dim = ckpt["x_dim"]
    cond_dim = ckpt["cond_dim"]

    from src.models.gan.cwgan_gp import CWGANGP
    model = CWGANGP(x_dim=x_dim, cond_dim=cond_dim, z_dim=cfg["gan"]["z_dim"], hidden=cfg["gan"]["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Build a conditioning pool from real benign train (so cond is realistic)
    benign_label = cfg["data"]["benign_label"]
    y = (train_df[cfg["data"]["label_col"]].astype(str).values != benign_label).astype(int)
    benign_df = train_df[y == 0].copy()

    # Create "target https ratio" by resampling benign rows with port 443 more often
    dst_port_col = cfg["data"]["dst_port_col"]
    if dst_port_col not in benign_df.columns:
        raise ValueError(f"dst_port_col '{dst_port_col}' not found. Update configs/base.yaml")

    https_mask = benign_df[dst_port_col].astype(int) == 443
    https_df = benign_df[https_mask]
    other_df = benign_df[~https_mask]

    n = int(cfg["generate"]["n_samples"])
    n_https = int(n * cfg["generate"]["target_https_ratio"])
    n_other = n - n_https

    sample_https = https_df.sample(n=min(n_https, len(https_df)), replace=True, random_state=cfg["seed"])
    sample_other = other_df.sample(n=min(n_other, len(other_df)), replace=True, random_state=cfg["seed"])
    pool = pd.concat([sample_https, sample_other], axis=0).sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)

    # cond vectors from pool
    c_pool, _ = build_cond(
        pool,
        protocol_col=cfg["data"]["protocol_col"],
        dst_port_col=cfg["data"]["dst_port_col"],
        use_protocol=cfg["gan"]["cond"]["use_protocol"],
        use_service_bucket=cfg["gan"]["cond"]["use_service_bucket"],
        use_time_window=cfg["gan"]["cond"]["use_time_window"],
    )
    C = torch.tensor(c_pool, dtype=torch.float32).to(device)

    # generate in batches
    z_dim = cfg["gan"]["z_dim"]
    bs = 4096
    outs = []
    for i in range(0, n, bs):
        cb = C[i:i+bs]
        z = torch.randn(cb.size(0), z_dim, device=device)
        xb = model.gen(z, cb).detach().cpu().numpy()
        outs.append(xb)

    Xs = np.vstack(outs)

    synth_scaled = pd.DataFrame(Xs, columns=feat_cols)

    # invert scaling for numeric cols (continuous+count_like)
    synth_raw = inverse_scaler(synth_scaled, cont_cols, scaler)

    # constraint repair (v1)
    synth_raw = enforce_nonneg(synth_raw, cont_cols)
    synth_raw = round_counts(synth_raw, [c for c in schema["count_like"] if c in synth_raw.columns])

    # add label as BENIGN (since trained benign-only)
    synth_raw[cfg["data"]["label_col"]] = cfg["data"]["benign_label"]

    out_csv = os.path.join(out_dir, "synthetic_benign_v1.csv")
    synth_raw.to_csv(out_csv, index=False)

    vr = violation_rate_nonneg(synth_raw, cont_cols)
    print("Saved:", out_csv)
    print("Nonneg violation rate:", vr)

if __name__ == "__main__":
    main()
