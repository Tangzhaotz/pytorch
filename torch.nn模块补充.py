# pytorch
# -*- coding: utf-8 -*-
# @Author  : Tangzhao
# @Blog:https://blog.csdn.net/tangzhaotz

"""
构建神经网络，采用了torch.nn.Squential()来构建网络层，如果要对每一个层定义一个名称，可以采用Squential的一种改进方法，在Squential的基础上，
通过add_module()来添加每一层，并且为每一层添加单独的名称
此外，还可以在Squential的基础上，通过字典的形式添加每一层，设置单独的层名称
"""
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv = nn.Sequential(
            OrderedDict(
                [
                    ('conv1',nn.Conv2d(3,32,3,1,1)),
                    ('relu',nn.Relu()),
                    ('pool',nn.MaxPool2d(2))
                ]
            )
        )

        self.dense = nn.Sequential(
            OrderedDict(
                [
                    ('dense1',nn.Linear(32*3*3,128)),
                    ('relu2',nn.Relu()),
                    ('dense2',nn.Linear(128,10))
                ]
            )
        )



# 前向传播
"""
在forward函数中，有些层来自nn.Module，也可以使用nn.functional，来自nn.Module的需要实例化，而使用nn.functional的可以直接使用
"""
# 反向传播：pytorch提供了自动反向传播的功能，使用loss.backward()即可

# x训练模型
"""
训练模型是需要注意使模型处于训练模式，即调用model.train()，调用之后会把所有的module设置为训练模式，在验证或者测试时，需要将模型调用model.eval()
，调用之后会将training属性设置为False
缺省情况下梯度是累加的，需要手工把梯度初始化或者清零，调用optimizer.zero_grad()即可
"""

# 神经网络工具箱nn
"""
nn中主要有两个模块，nn.Module和nn.functional
nn.Module是nn的一个核心数据结构，它可以是神经网络的某个层，也可以包含多层的神经网络，在实际使用中，最常见的是继承nn.Module，生成自己的网络
nn中的层，一类是继承了nn.Module，其命名一般为nn.Xxx(第一个大写），如nn.Linear，nn.Conv2D等，另外一类是nn.functional中的函数，其名称一般为
nn.functional.xxx,如nn.functional.linear,nn.functional.conv2d，两者大体相似，主要区别为：
1、nn.Xxx继承于nn.Module，nn.Xxx需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入数据，它能够很好的与nn.Squential结合使用，而
nn.functional.xxx无法与nn.Squential结合使用
2、nn.Xxx不需要自己定义和管理weight，bias等参数；而nn.functional.xx需要自己定义weight、bias参数，每次调用的时候需要手动传入weight、bias等
参数，不利于代码复用
3、Dropout操作是在训练和测试阶段是有区别的，使用nn.Xxx方式定义Dropout,在调用model.eval()之后，自动实现状态的转换，而使用nn.functional.xx却
无此功能
"""

# 动态修改学习率参数
"""
修改参数的方式可以通过修改参数optimizer.params_groups或者新建optimizer，新建optimizer比较简单，但是新的优化器会初始化动量等状态信息，这对于
使用动量的优化器（momentum参数中的sgd）可能会造成收敛中的震荡
len(optimizer.param_groups:长度为1的list，optimizer.param_groups[0]:长度为6的字典，包括权重参数，lr，momentum等参数
"""

# 优化器比较
import torch
import torch.utils.data as Data
import torch.functional as F
import matplotlib.pyplot as plt


# 超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# 生成数据
x = torch.unsqueeze(torch.linspace(-1,1,1000),dim=1)  # 将数据扩充维度
# print(x)
# 0.1 * torch.normal(x.size())增加噪点
y = x.pow(2) + 0.1 * torch.norm(torch.zeros(*x.size()))

torch_dataset = Data.TensorDataset(x,y)
# 得到一个批量的迭代器
loader = Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True)

# 构建神经网络
class Net(nn.Module):
    # 初始化
    def __init__(self):
        super(Net,self).__init__()
        self.hidden = nn.Linear(1,20)
        self.predict = nn.Linear(20,1)

    # 前向传递
    def forward(self,x):
        x = torch.relu(self.hidden(x))  # 这里新版的将relu，sigmoid和tanh等函数放在torch里面，不在functional里面
        x = self.predict(x)
        return x

# 使用多种优化器
net_SGD = Net()
net_Momentum = Net()
net_PMSprop = Net()
net_Adam = Net()

nets = [net_SGD,net_Momentum,net_PMSprop,net_Adam]

opt_SGD = torch.optim.SGD(net_SGD.parameters(),lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(),lr=LR,momentum=0.9)
opt_RMSprop = torch.optim.RMSprop(net_PMSprop.parameters(),lr=LR,alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(),lr=LR,betas=(0.9,0.99))

optimizers = [opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]

# 训练模型
loss_func = torch.nn.MSELoss()
loss_his = [[],[],[],[]]  # 记录损失
for epoch in range(EPOCH):
    for step,(batch_x,batch_y) in enumerate(loader):
        for net,opt,l_his in zip(nets,optimizers,loss_his):
            output = net(batch_x)
            loss = loss_func(output,batch_y)  # 对每个net计算损失
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data.numpy())
labels = ['SGD','Momentum','RMSProp','Adam']

# 可视化结果
for i,l_his in enumerate(loss_his):
    plt.plot(l_his,label=labels[i])
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0,0.2))
plt.show()