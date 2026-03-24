import torch
import torch.nn as nn
from ..backbone.feature_extractor import FeatureExtractor
from ..layers.positional_encoding import PositionalEncoding2D
from ..modules.normalizing_flow import NormalizingFlow
from ..modules.student_cnn import StudentCNN
from ..losses.losses import teacher_loss, student_loss

class ASTModel(nn.Module):
    def __init__(self, feat_channels=512, nf_hidden=256, nf_blocks=6, student_blocks=4, use_pe=True):
        super().__init__()
        self.feature_extractor = FeatureExtractor(out_features=feat_channels)
        self.use_pe = use_pe
        if use_pe:
            self.pos_enc = PositionalEncoding2D(feat_channels)
            pe_channels = feat_channels
        else:
            self.pos_enc = None
            pe_channels = 0
        
        self.teacher = NormalizingFlow(in_channels=feat_channels + pe_channels, hidden_channels=nf_hidden, n_blocks=nf_blocks, cond_channels=0)
        self.student = StudentCNN(in_channels=feat_channels + pe_channels, out_channels=feat_channels, n_blocks=student_blocks)

    def forward(self, x):
        x_feat = self.feature_extractor(x)
        if self.use_pe:
            x_feat = self.pos_enc(x_feat)
        ft_x = self.teacher(x_feat)
        fs_x = self.student(x_feat)
        return fs_x, ft_x

    def compute_losses(self, x):
        fs_x, ft_x = self.forward(x)
        Lt = teacher_loss(ft_x)
        Ls = student_loss(fs_x, ft_x)
        return Lt, Ls
