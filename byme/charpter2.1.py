import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1=nn.Conv2d(1,6,3)
        self.fc1=nn.Linear(1350,10)

    def forward(self, x):

        print(x.size())

        x=self.conv1(x)

        x=F.relu(x)
        print(x.size())

        x=F.max_pool2d(x,(2,2))
        x=F.relu(x)
        print(x.size())

        x=x.view(x.size()[0],-1)
        print(x.size())

        x=self.fc1(x)

        return x



net=Net()
print(net)


for name,parameters in net.named_parameters():
    print(name,": ",parameters)

