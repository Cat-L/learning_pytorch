import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR=0.01
EPOCHS=20
BATCH_SIZE=256

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           # transforms.Scale((224, 224), 2),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           # transforms.Resize(224*224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


# input 28*28
class mynet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1=nn.Sequential(
            nn.Conv2d(1,16,5),  #output 24*24
            nn.ReLU(),
            nn.MaxPool2d(3,2)      #output 7*7
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(16,32,3,padding=1),  #output 7*7
            nn.ReLU(),
            nn.MaxPool2d(3,2)     #output 3
        )

        self.classifier=nn.Sequential(
            nn.Linear(32*5*5,1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000,10)
        )


    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.view(x.size(0),-1)
        out=self.classifier(x)
        return out




mynet=mynet().to(DEVICE)



optimizer=torch.optim.Adam(mynet.parameters(),LR)
loss_func=nn.CrossEntropyLoss()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss =loss_func(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
            ))



def mytest_(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target).item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))



for epoch in range(1, EPOCHS + 1):
    train(mynet, DEVICE, train_loader, optimizer, epoch)
    mytest_(mynet, DEVICE, test_loader)
