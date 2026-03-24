import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class StudentCNN(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks=4):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(n_blocks)])
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.output_conv(x)
        return x
