import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet18", pretrained=True, out_features=512):
        super().__init__()
        if model_name == "resnet18":
            resnet = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(resnet.children())[:-2])  
            self.out_channels = 512
        else:
            raise NotImplementedError(f"{model_name} not implemented")
        
    def forward(self, x):
        return self.features(x)  
