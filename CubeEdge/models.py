import torch
import torch.nn as nn

if __name__ == "__main__":
    import sys
    sys.path.append('.')
from  components.Layers import *
RMLP = nn.Sequential(
    nn.Linear(7*4,128),
    nn.ReLU(),
    nn.Linear(128,128),
    nn.ReLU(),
    nn.Linear(128,32)s
)

QMLP = nn.Sequential(
    QPU(7*4,128),
    QPU(128,128),
    nn.Linear(128,32)
)

QMLP_RInv = nn.Sequential(
    QPU(7*4,128),
    QPU(128,512),
    KeepRealPart(),
    nn.Linear()
)