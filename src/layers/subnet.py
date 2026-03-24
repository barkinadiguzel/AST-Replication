import torch.nn as nn

class Subnet(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1)
        )
    
    def forward(self, x):
        return self.net(x)
