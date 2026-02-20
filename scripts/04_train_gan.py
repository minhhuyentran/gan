import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils.config import load_config, ensure_dirs
from src.utils.io import load_pkl, load_json, save_pkl
from src.utils.seed import set_seed
from src.models.gan.cwgan_gp import CWGANGP, gradient_penalty

def main():
    cfg = load_config("configs/base.yaml")
    set_seed(cfg["seed"])
    out_dir = cfg["paths"]["out_dir"]
    art_dir = cfg["paths"]["artifacts_dir"]
    ensure_dirs(out_dir, art_dir)

    schema = load_json(os.path.join(art_dir, "schema.json"))
    train_df, c_train = load_pkl(os.path.join(art_dir, "train.pkl"))

    # train GAN on BENIGN only (recommended)
    benign_label = cfg["data"]["benign_label"]
    y = (train_df[cfg["data"]["label_col"]].astype(str).values != benign_label).astype(int)
    train_df = train_df[y == 0].copy()
    c_train = c_train[y == 0]

    feat_cols = schema["continuous"] + schema["count_like"]
    feat_cols = [c for c in feat_cols if c in train_df.columns]
    X = train_df[feat_cols].to_numpy().astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_t = torch.tensor(X, dtype=torch.float32)
    C_t = torch.tensor(c_train, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X_t, C_t), batch_size=cfg["gan"]["batch_size"], shuffle=True, drop_last=True)

    x_dim = X.shape[1]
    cond_dim = c_train.shape[1]
    model = CWGANGP(x_dim=x_dim, cond_dim=cond_dim, z_dim=cfg["gan"]["z_dim"], hidden=cfg["gan"]["hidden"]).to(device)

    opt_D = torch.optim.Adam(model.D.parameters(), lr=cfg["gan"]["lr"], betas=(0.5, 0.9))
    opt_G = torch.optim.Adam(model.G.parameters(), lr=cfg["gan"]["lr"], betas=(0.5, 0.9))

    n_critic = cfg["gan"]["n_critic"]
    z_dim = cfg["gan"]["z_dim"]

    for epoch in range(cfg["gan"]["epochs"]):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{cfg['gan']['epochs']}")
        for real_x, c in pbar:
            real_x = real_x.to(device)
            c = c.to(device)

            # --- train critic n_critic times
            for _ in range(n_critic):
                z = torch.randn(real_x.size(0), z_dim, device=device)
                fake_x = model.gen(z, c).detach()

                d_real = model.disc(real_x, c).mean()
                d_fake = model.disc(fake_x, c).mean()
                gp = gradient_penalty(model, real_x, fake_x, c, gp_lambda=cfg["gan"]["gp_lambda"])

                loss_D = (d_fake - d_real) + gp

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # --- train generator
            z = torch.randn(real_x.size(0), z_dim, device=device)
            fake_x = model.gen(z, c)
            loss_G = -model.disc(fake_x, c).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            pbar.set_postfix({"D": float(loss_D.item()), "G": float(loss_G.item()), "gp": float(gp.item())})

    ckpt_path = os.path.join(out_dir, "gan_cwgan_gp.pt")
    torch.save({"state_dict": model.state_dict(), "feat_cols": feat_cols, "cond_dim": cond_dim, "x_dim": x_dim}, ckpt_path)
    print("Saved GAN:", ckpt_path)

if __name__ == "__main__":
    main()
