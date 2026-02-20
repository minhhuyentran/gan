import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class AE(nn.Module):
    def __init__(self, x_dim: int, hidden=(128, 64)):
        super().__init__()
        h1, h2 = hidden
        self.enc = nn.Sequential(
            nn.Linear(x_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, x_dim),
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

def train_ae(X_train, epochs=10, batch_size=1024, lr=1e-3, device="cpu"):
    x = torch.tensor(X_train, dtype=torch.float32)
    dl = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)

    model = AE(x_dim=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (xb,) in dl:
            xb = xb.to(device)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def score_ae(model, X, device="cpu"):
    model.eval()
    x = torch.tensor(X, dtype=torch.float32).to(device)
    recon = model(x)
    # reconstruction error: higher => more anomalous (đổi sign cho unified)
    err = torch.mean((recon - x) ** 2, dim=1).detach().cpu().numpy()
    # convert to "normal score" (higher more normal) for consistent thresholding
    return -err
