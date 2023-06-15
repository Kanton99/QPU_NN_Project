import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import sys
    sys.path.append('.')
from models import *
from cubeEdgeData import CubeEdge


def train(model,data,epochs,lr,batch_size):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    train_dataloader = DataLoader(data,batch_size=batch_size,shuffle=True)
    for epoch in range(epochs):
        #Forward pass
        for input,label in train_dataloader:
            initial = input[0]
            print(model(initial))
            optimizer.zero_grad()
            outputs = model(input)
            #print(outputs)
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()

if __name__=="__main__":

    data = CubeEdge(train=True, num_edges=7, use_quaternion=True,num_samples=2000)
    rmlp_net = RMLP(num_data=7,num_cls=data.num_shapes)
    train(rmlp_net,data=data,epochs=100,lr=0.01,batch_size=data.num_shapes)

    print(rmlp_net(torch.tensor(data[0][0])))

    