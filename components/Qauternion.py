import torch
from torch import nn


class QPU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = self.flatten(x)
        