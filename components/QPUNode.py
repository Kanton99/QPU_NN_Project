import torch
import torch.nn as nn


class QPU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = self.flatten(x)
        
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(device)