import torch.nn as nn
from ..layers.coupling import AffineCoupling

class NormalizingFlow(nn.Module):
    def __init__(self, in_channels, hidden_channels, n_blocks=6, cond_channels=0):
        super().__init__()
        self.blocks = nn.ModuleList([AffineCoupling(in_channels, hidden_channels, cond_channels) for _ in range(n_blocks)])

    def forward(self, x, c=None):
        for block in self.blocks:
            x = block(x, c)
        return x
