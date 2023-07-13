import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import time

if __name__ == "__main__":
    import sys
    sys.path.append('.')
from models import *
from cubeEdgeData import CubeEdge

device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
def train(model,data,epochs,lr,batch_size): 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    train_dataloader = DataLoader(data,batch_size=batch_size,shuffle=True)
    loss = 0
    net.to(device=device)
    for epoch in range(epochs):
        net.train()
        #Forward pass
        for input,label in train_dataloader:
            input.to(device)
            label.to(device)
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs,label)
            loss.backward()
            optimizer.step()
    
        if ((epoch+1) % 10) == 0:
            print(f"epoch: {epoch+1}")
            print(loss)

def test(data, model):
    dataloader = DataLoader(data,batch_size=32,shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__=="__main__":

    start_time = time.time()
    training_data = CubeEdge(train=True, num_edges=7, use_quaternion=True,num_samples=500)
    test_data = CubeEdge(train=False,num_edges=7,use_quaternion=True,num_samples=50)
    net = QMLP_RInv(num_data=7,num_cls=training_data.num_shapes)
    train(model=net,data=training_data,epochs=100,lr=0.01,batch_size=training_data.num_shapes)
    test(model=net,data=test_data)
    print("--- %s seconds ---" % (time.time() - start_time))
    #print(rmlp_net(torch.tensor(data[0][0])))

    