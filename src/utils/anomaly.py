import torch

def anomaly_score(fs_x, ft_x, reduction="mean"):
    score_map = torch.norm(fs_x - ft_x, p=2, dim=1)  
    if reduction=="mean":
        return score_map.mean(dim=[1,2])
    elif reduction=="max":
        return score_map.amax(dim=[1,2])
    else:
        return score_map
