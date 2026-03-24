import torch
import torch.nn as nn

def teacher_loss(z):
    """
    Lt = - log pX(x) = ||z||^2 /2 - log|det dz/dx|
    simplified here: ||z||^2 /2 (that part can be implemented with NF library)
    """
    return 0.5 * torch.sum(z ** 2)

def student_loss(fs_x, ft_x):
    """
    Ls = ||fs(x) - ft(x)||^2
    """
    return torch.mean((fs_x - ft_x) ** 2)
