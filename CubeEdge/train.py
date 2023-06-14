import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import sys
    sys.path.append('.')
from models import *
from cubeEdgeData import CubeEdge


def train(model,data,epochs,lr):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    train_dataloader = DataLoader(data,batch_size=64,shuffle=True)
    for epoch in range(epochs):
        #Forward pass
        
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs,labels)
            loss.backwards()

            optimizer.step()

if __name__=="__main__":

    data = CubeEdge(train=True, num_edges=4, use_quaternion=True)
    train(QMLP(),data, 100,0.01)