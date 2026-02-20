import torch
import torch.nn as nn
from .modules import MLP

class CWGANGP(nn.Module):
    def __init__(self, x_dim: int, cond_dim: int, z_dim: int, hidden: list[int]):
        super().__init__()
        self.G = MLP(z_dim + cond_dim, hidden, x_dim)
        self.D = MLP(x_dim + cond_dim, hidden, 1)

    def gen(self, z, c):
        return self.G(torch.cat([z, c], dim=1))

    def disc(self, x, c):
        return self.D(torch.cat([x, c], dim=1))

def gradient_penalty(model: CWGANGP, real_x, fake_x, c, device, gp_lambda=10.0):
    alpha = torch.rand(real_x.size(0), 1, device=device)
    alpha = alpha.expand_as(real_x)
    interpolates = alpha * real_x + (1 - alpha) * fake_x
    interpolates.requires_grad_(True)

    d_inter = model.disc(interpolates, c)
    grads = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp_lambda * gp
