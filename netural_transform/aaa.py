import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),  # b, 16(高度), 26, 26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 32, 12, 12
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # b, 64, 10, 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # b, 128, 4, 4
        )

        self.out = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),  # (input, output)
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),  # (input, output)
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)  # (input, output)
        )

    def forward(self, x):
        x = self.layer1(x)  # (batch, 16, 26, 26) -> (batchsize, 输出图片高度, 输出图片长度, 输出图片宽度)
        x = self.layer2(x)  # (batch, 32, 12, 12)
        x = self.layer3(x)  # (batch, 64, 10, 10)
        x = self.layer4(x)  # (batch, 128, 4, 4)
        x = x.view(x.size(0), -1)  # 扩展、展平 -> (batch, 128 * 4 * 4)
        x = self.out(x)
        return x

# 定义超参数
batch_size = 64
learning_rate = 1e-2
num_epoches = 20000

if __name__ == '__main__':
    # 数据预处理
    data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # 下载训练集-MNIST手写数字训练集
    train_dataset = datasets.MNIST(root="./data", train=True, transform=data_tf, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=data_tf)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = CNN()
    if torch.cuda.is_available():
        model = model.cuda()
    # 定义损失函数和优化函数
    criterion = nn.CrossEntropyLoss()  # 损失函数：损失函数交叉熵
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 优化函数：随机梯度下降法
    # 训练网络
    epoch = 0
    for data in train_loader:
        img, label = data
        img = Variable(img)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()  # 梯度归零
        loss.backward()
        optimizer.step()  # 更新参数
        epoch += 1
        if (epoch) % 100 == 0:
            #print('*' * 10)
            print('epoch{} loss is {:.4f}'.format(epoch,loss.item()))
            #print('loss is {:.4f}'.format(loss.item()))
    # 测试网络
    model.eval()
    eval_loss = 0
    eval_acc = 0
    for data in test_loader:
        img, label = data
        # img = img.view(img.size(0), -1)
        img = Variable(img)
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Test Loss:{:.6f}, Acc:{:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))
