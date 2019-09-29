import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

LR=0.001
BATCH_SIZE=512 #大概需要2G的显存
EPOCHS=100 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(224*224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.Resize(224*224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)

# for i,data in enumerate(train_loader):
#     print(data[0].size())
# 你会发现图像的形状torch.Size([512,1,28,28])，这表明每个batch中有512个图像，每个图像的尺寸为28 x 28像素。同样，标签的形状为torch.Size([512])，512张图像分别对应有512个标签。



#  input size =224*224*1
class Alexnet(nn.Module):

    def __init__(self):
        super(Alexnet,self).__init__()
        self.conv1=nn.Sequential(  # First Layer
            nn.Conv2d(             # conv(1,64,11) -> Relu -> maxpool(3*3)
                in_channels=1,     # conv out size 216
                out_channels=64,
                kernel_size=11,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(3)       # maxpool output size 71
            #todo if input size could not be divisible, how would it do?
            )
        self.conv2=nn.Sequential( #Second Layer
            nn.Conv2d(            #conv(64,192,5) -> Relu -> maxpool(3*3)
                in_channels=64,   #conv out size  69
                out_channels=192,
                kernel_size=5,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(3)       # maxpool output size
        )
        self.conv3=nn.Sequential( #Third Layer

            nn.Conv2d(            #conv(192,384,3) -> conv(384,256,3) ->conv(256,265,3)
                in_channels=192,  #conv out size  23
                out_channels=384,
                kernel_size=3,
                padding=1
            ),
            nn.Conv2d(
                in_channels=384,  #conv out size  23
                out_channels=256,
                kernel_size=3,
                padding=1

            ),
            nn.Conv2d(
                in_channels=256,  #conv out size  23
                out_channels=256,
                kernel_size=3,
                padding=1
            ),
            nn.MaxPool2d(3)       # maxpool output size 6
        )
        self.classifier=nn.Sequential(
            nn.Dropout(),
            # TODO Liear could not match the out of last conv
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self,x):
        print("input",x.size())
        x=self.conv1(x)
        print("conv1 out ",x.size())
        x=self.conv2(x)
        print("conv2 out",x.size())
        x=self.conv3(x)
        # x=x.view(x.size(0),-1)
        out=self.classifier(x)
        return out

alexnet=Alexnet().to(DEVICE)

optimizer=torch.optim.Adam(alexnet.parameters(),LR)
loss_func=nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))



def mytest_(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            print(data.size())
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



for epoch in range(1, EPOCHS + 1):
    train(alexnet, DEVICE, train_loader, optimizer, epoch)
    mytest_(alexnet, DEVICE, test_loader)

# for epoch in range(EPOCHS):
#     for step, (batch_x,batch_laber) in enumerate(train_loader):
#         b_x=Variable(batch_x)
#         b_y=Variable(batch_laber)
#
#         output=alexnet(b_x)
#         print(output)
#
#         loss=loss_func(output,b_y)
#
#         optimizer.zero_grad()
#
#         loss.backward()
#
#         optimizer.step()
#
#         if step % 50 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, step * len(batch_x), len(train_loader.dataset),
#                        100. * step / len(train_loader), loss.item()))

