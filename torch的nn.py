# python深度学习基于pytorch
# -*- coding: utf-8 -*-
# @Author  : Tangzhao
# @FileName: torch的nn.py
# @Software: PyCharm
# @Blog    ：http://blog.csdn.net/tangzhaotz

"""
nn.functional与nn.Module中的Layer的主要区别是后者继承Module，会自动提取可以学习的参数，而nn.functional更像是纯函数。像卷积层、全连接层，
Droupout等层有学习参数，一般使用nn.Module，而激活函数，池化层不含可学习参数，可以使用nn.functional
"""

# 导入必要的模块
import numpy as np
import torch
from torchvision.datasets import mnist
# 导入预处理模块
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

# 定义一些超参数
train_batch_size = 64
test_batch_size = 128
learning_rate = 0.01
num_epoches = 20
lr = 0.01
momentu = 0.5

# 下载数据并对数据进行预处理
# 定义预处理函数，这些预处理依次放入Compose中
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])  # 对张量进行归一化，两个0.5分别代表对张量进行平均和方差，灰色的图只有一个通道，若是RGB，需要设置
    # 多个通道的值
])

# 下载数据集并对数据进行预处理
train_dataset = mnist.MNIST('./data',train=True,download=True,transform=transform)
test_dataset = mnist.MNIST('./data',download=True,transform=transform)

# dataloader是一个可迭代对象，可以使用迭代器一样使用
train_loader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=test_batch_size,shuffle=False)

# 可视化数据
import matplotlib.pyplot as plt
examples = enumerate(test_loader)
batch_idx,(example_data,example_targets) = next(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0],cmap='gray',interpolation='none')
    plt.title("Ground Truth:{}".format(example_targets[i]))
    plt.xticks()
    plt.yticks()
    plt.show()

# 构建模型
# 1、搭建网络
class Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    # 定义前向传播
    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# 实例化网络
# 检测是否有GPU可用，有则使用，否则使用cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 实例化网络
model = Net(28 * 28,300,100,10)
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=lr,momentum=momentu)

# 训练模型
# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(num_epoches):
    train_loss = 0
    train_acc = 0
    model.train()

    # 动态修改参数学习率
    if epoch % 5 == 0:
        optimizer.param_groups[0]['lr'] *= 0.1
    for img,label in train_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0),-1)

        # 前向传播
        out = model(img)
        loss = criterion(out,label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 更新参数
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _,pred = out.max(1)
        num_correct = (pred == label).sum().item()

        acc = num_correct / img.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))

    # 在测试集上验证效果
    eval_loss = 0
    eval_acc = 0

    # 将模型改为测试模式
    model.eval()
    for img,label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0),-1)
        out = model(img)
        loss = criterion(out,label)

        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _,pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('epoch:{},Train loss:{:.4f},Train acc:{:.4f},Test loss:{:.4f},Test acc:{:.4f}'
          .format(epoch,train_loss / len(train_loader),train_acc / len(train_loader),eval_loss / len(test_loader),
                  eval_acc / len(test_loader)
                  ))

# 可视化训练及测试损失值
plt.title('trainloss')
plt.plot(np.arange(len(losses)),losses)
plt.legend(['Train loss'],loc = 'upper right')



