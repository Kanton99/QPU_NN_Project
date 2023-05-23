import torch
import torch.nn as nn
from components import Layers

RMLP = nn.Sequential(
    nn.Linear(7*4,128),
    nn.ReLU(),
    nn.Linear(128,128),
    nn.ReLU(),
    nn.Linear(128,32)
)

QMLP = nn.Sequential(
    Layers.QPU(7*4,128),
    Layers.QPU(128,128),
    nn.Linear(128,32)
)

QMLP_RInv = nn.Sequential(
    Layers.QPU(7*4,128),
    Layers.QPU(128,512),
    Layers.KeepRealPart(),
    nn.Linear()
)