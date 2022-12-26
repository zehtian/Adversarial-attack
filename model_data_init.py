# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def model_init(data_name):
    if data_name == 'mnist':
        model = Net_mnist()
    elif data_name == 'cifar10':
        model = ResNet18()

    return model


class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# CIFAR10 LENET(一般CNN)
class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# CIFAR10 采用ResNet18
class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)  # ! (h-3+2)/2 + 1 = h/2 图像尺寸减半
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)  # ! h-3+2*1+1=h 图像尺寸没变化
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),  # ! 这句话是针对原图像尺寸写的，要进行element wise add
            # ! 因此图像尺寸也必须减半，(h-1)/2+1=h/2 图像尺寸减半
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        out = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # short cut
        # ! element wise add [b,ch_in,h,w] [b,ch_out,h,w] 必须当ch_in = ch_out时才能进行相加
        out = x + self.extra(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # ! 图像尺寸不变
            nn.BatchNorm2d(64)
        )
        # 4个ResBlock
        #  [b,64,h,w] --> [b,128,h,w]
        self.block1 = ResBlock(64, 128)
        #  [b,128,h,w] --> [b,256,h,w]
        self.block2 = ResBlock(128, 256)
        #  [b,256,h,w] --> [b,512,h,w]
        self.block3 = ResBlock(256, 512)
        #  [b,512,h,w] --> [b,512,h,w]
        self.block4 = ResBlock(512, 512)

        self.outlayer = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # [b,64,h,w] --> [b,1024,h,w]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # print("after conv:",x.shape)
        # [b,512,h,w] --> [b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # flatten
        x = x.view(x.shape[0], -1)
        x = self.outlayer(x)
        return x

# 数据加载函数
def data_init(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.cuda_state else {}
    trainset, testset = data_set(FL_params.data_name)

    # 构建训练数据加载器
    train_loader = DataLoader(trainset, batch_size=FL_params.train_batch_size, shuffle=True, **kwargs)
    # 构建测试数据加载器
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, drop_last=True, **kwargs)

    return train_loader, test_loader


# 数据设置函数
def data_set(data_name):
    if data_name not in ['mnist', 'cifar10', 'imagenet']:
        raise TypeError('data_name should be a string, including mnist, cifar10, imagenet.')

    # MNIST数据处理
    if data_name == 'mnist':
        trainset = datasets.MNIST('./data', train=True, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

        testset = datasets.MNIST('./data', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))

    # CIFAR10数据处理
    elif data_name == 'cifar10':
        # 使得原始数据范围为[0, 1]
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=False, transform=transform)
        testset = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=False, transform=transform)

    return trainset, testset


def model_train(FL_params, train_loader, test_loader):

    print("Model Training Starting...")

    # 加载初始模型
    model = model_init(FL_params.data_name)

    model.to(FL_params.device)
    model.train()

    # 定义优化器和损失函数 -> 初始：Adam & CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=FL_params.local_lr)
    criteria = nn.CrossEntropyLoss()

    for epoch in range(FL_params.train_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(FL_params.device), target.to(FL_params.device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criteria(pred, target)
            loss.backward()
            optimizer.step()

        if FL_params.train_with_test:
            if epoch % 5 == 4:
                print("Training Epoch:", epoch)
                test(model, train_loader, test_loader, FL_params)

    path = "./model/model_" + FL_params.data_name + "_" + str(FL_params.train_epoch) + "_" + str(FL_params.local_lr) + ".pth"
    torch.save(model.state_dict(), path)

    print("Model Training Successfully!")


# 测试模型在测试集上的性能，在device上进行
def test(model, train_loader, test_loader, FL_params):

    model.eval()

    train_loss = 0
    train_acc = 0
    i = 0
    for data, target in train_loader:

        data, target = data.to(FL_params.device), target.to(FL_params.device)
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        train_loss += criteria(output, target)
        pred = torch.argmax(output, dim=1)
        train_acc += torch.sum(torch.where(
            pred == target, torch.tensor(1, device=FL_params.device), torch.tensor(0, device=FL_params.device))).item() / data.shape[0]
        i += 1
        if i > 3:
            break

    # train_loss /= len(train_loader.dataset)
    # train_acc = train_acc / torch.ceil(len(train_loader.dataset) / train_loader.batch_size).item()
    train_acc = train_acc / 4
    # # print('Train set: Average loss: {:.4f}'.format(train_loss))
    print('Train set: Average acc:  {:.4f}'.format(train_acc))

    test_loss = 0
    test_acc = 0
    i = 0
    for data, target in test_loader:
        data, target = data.to(FL_params.device), target.to(FL_params.device)
        output = model(data)
        criteria = nn.CrossEntropyLoss()
        test_loss += criteria(output, target)
        pred = torch.argmax(output, dim=1)
        test_acc += torch.sum(torch.where(
            pred == target, torch.tensor(1, device=FL_params.device), torch.tensor(0, device=FL_params.device))).item() / data.shape[0]

        i += 1
        if i > 3:
            break

    # test_loss /= len(test_loader.dataset)
    test_acc = test_acc / 4
    # test_acc = test_acc / torch.ceil(len(test_loader.dataset) / test_loader.batch_size).item()
    # print('Test set: Average loss: {:.4f}'.format(test_loss))
    print('Test set: Average acc:  {:.4f}'.format(test_acc))

    model.train()



