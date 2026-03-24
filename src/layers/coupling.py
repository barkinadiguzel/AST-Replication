import torch
import torch.nn as nn
from .subnet import Subnet

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, hidden_channels, cond_channels=0):
        super().__init__()
        self.cond_channels = cond_channels
        self.in_channels = in_channels
        self.split_len1 = in_channels // 2
        self.split_len2 = in_channels - self.split_len1
        
        self.s1 = Subnet(self.split_len1 + cond_channels, hidden_channels)
        self.t1 = Subnet(self.split_len1 + cond_channels, hidden_channels)
        self.s2 = Subnet(self.split_len2 + cond_channels, hidden_channels)
        self.t2 = Subnet(self.split_len2 + cond_channels, hidden_channels)

    def forward(self, x, c=None):
        x1, x2 = torch.split(x, [self.split_len1, self.split_len2], dim=1)
        if c is not None:
            x1c = torch.cat([x1, c], dim=1)
            x2c = torch.cat([x2, c], dim=1)
        else:
            x1c, x2c = x1, x2

        y2 = x2 * torch.exp(self.s1(x1c)) + self.t1(x1c)
        y1 = x1 * torch.exp(self.s2(x2c)) + self.t2(x2c)
        y = torch.cat([y1, y2], dim=1)
        return y
