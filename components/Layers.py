import torch
import torch.nn as nn


class QPU(nn.Module):
    #similar to QPU example github since needs same parameters
    def __init__(self, in_features, out_features):
        super(QPU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self,x):
        x = self.flatten(x)
        
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(device)