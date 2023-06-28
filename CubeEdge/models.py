import torch
import torch.nn as nn

if __name__ == "__main__":
    import sys
    sys.path.append('.')
from  components.Layers import *

# Taken from paper's code in CubeEdge/models.py
class MLPBase(nn.Module):
    def __init__(self):
        super(MLPBase, self).__init__()

    def forward(self, x):
        # batch_size = x.shape[0]
        # x = x.permute(0, 2, 1)
        # x = x.reshape(batch_size, -1)
        #x = torch.flatten(x)
        return self.stack(x)

class RMLP(MLPBase):
    def __init__(self,num_data,num_cls):
        super(RMLP,self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(num_data*4,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,num_cls)
        )
    
    def forward(self,x):
        x = torch.flatten(x)
        return self.stack(x)

# QMLP_RInv = nn.Sequential(
#     QPU(7,128),
#     QPU(128,512),
#     KeepRealPart(),
#     nn.Linear()
# )

class QMLP(MLPBase):
    def __init__(self,num_data,num_cls):
        super(QMLP,self).__init__()

        self.stack = nn.Sequential(
            QPU(num_data*4,128),
            QPU(128,128),
            nn.Linear(128,num_cls)
        )