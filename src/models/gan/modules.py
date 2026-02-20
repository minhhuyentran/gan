import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LeakyReLU(0.2)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
