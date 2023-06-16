import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import sys
    sys.path.append('.')
from models import *
from cubeEdgeData import CubeEdge


def train(model,data,epochs,lr,batch_size): 
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    train_dataloader = DataLoader(data,batch_size=batch_size,shuffle=True)
    loss = 0
    net.to(device=device)
    for epoch in range(epochs):
        #Forward pass
        for input,label in train_dataloader:
            input.to(device)
            label.to(device)
            for i in range(input.shape[0]):
                optimizer.zero_grad()
                outputs = model(input[i])
                #print(outputs)
                loss = criterion(outputs,label[i])
                loss.backward()
                optimizer.step()
        
        if epoch+1 % 10 == 0:
            print(f"epoch: {epoch+1}")
            print(loss)

if __name__=="__main__":

    data = CubeEdge(train=True, num_edges=7, use_quaternion=True,num_samples=100)
    net = QMLP(num_data=7,num_cls=data.num_shapes)
    train(model=net,data=data,epochs=100,lr=0.01,batch_size=data.num_shapes)

    #print(rmlp_net(torch.tensor(data[0][0])))

    