import torch
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        if channels % 4 != 0:
            raise ValueError("channels must be divisible by 4")
    
    def forward(self, x):
        B, C, H, W = x.size()
        device = x.device

        c = torch.zeros(B, self.channels, H, W, device=device)
        div_term = torch.exp(torch.arange(0, self.channels//2, 2, device=device) * -(math.log(10000.0) / (self.channels//2)))

        pos_w = torch.arange(0, W, device=device).unsqueeze(1)
        pos_h = torch.arange(0, H, device=device).unsqueeze(1)

        c[:, 0:self.channels//2:2, :, :] = torch.sin(pos_w * div_term).permute(1,0).unsqueeze(1).repeat(B,1,H,1)
        c[:, 1:self.channels//2:2, :, :] = torch.cos(pos_w * div_term).permute(1,0).unsqueeze(1).repeat(B,1,H,1)
        c[:, self.channels//2::2, :, :] = torch.sin(pos_h * div_term).permute(1,0).unsqueeze(2).repeat(B,1,1,W)
        c[:, self.channels//2+1::2, :, :] = torch.cos(pos_h * div_term).permute(1,0).unsqueeze(2).repeat(B,1,1,W)

        return torch.cat([x, c], dim=1)  
