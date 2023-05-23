import torch
import torch.nn as nn
from functions import *

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class QPU(nn.Module):
    #similar to QPU example github since needs same parameters
    def __init__(self, in_features, out_features):
        super(QPU, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = torch.Parameters(self.in_features,self.out_features)
        self.bias = 0
    
    def forward(self,x):
        return qpu_forward(x,self.weights,self.bias)
    
class AngelAxisMap(nn.Module):
    def __init__(self):
        super(AngelAxisMap,self).__init__()

    def forward(self,q):
        return angleAxisMap(q)
    
class KeepRealPart(nn.Module):
    def __init__(self) -> None:
        super(KeepRealPart,self).__init__()

    def forward(self,q):
        return q[0]