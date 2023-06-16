import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from components.functions import *


class QPU(nn.Module):
    """
    in_features: number of input data to each node
    out_features: number of nodes in the layer
    """
    #similar to QPU example github since needs same parameters
    def __init__(self, in_features, out_features):
        super(QPU, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.weights = Parameter(torch.Tensor(self.out_features,self.in_features))
        self.bias = Parameter(torch.Tensor(self.out_features))

        nn.init.xavier_uniform_(self.weights)
    
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