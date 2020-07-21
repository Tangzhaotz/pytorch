# pytorch
# -*- coding: utf-8 -*-
# @Author  : Tangzhao
# @Blog:https://blog.csdn.net/tangzhaotz

"""
pytorch数据处理工具箱概述
torch.utils.data工具包主要包括以下四个类：
    Dataset：是一个抽象类，其他数据集需要继承这个类，并且覆写其中的两个方法（_getitem__、__len__)
    DataLoader：定义一个新的迭代器，实现批量读取、打乱数据，并提供并行加速等功能
    random_split:把数据集随即拆分为给定长度的非重叠的新数据集
    *sample：多重采样函数
torchvision包括四个类：
    datasets:提供常用的数据集加载，设计上都是继承torch.utils.data.Dataset,主要包括MNIST、CIFAR10/100，ImageNet和COCO
    models：提供深度学习中各种经典的网络结构以及训练好的模型（如果选择pretained=True）,包括AlexNet,VGG系列，ResNet，Inception系列等
    transforms:常用的数据预处理操作，主要包括对Tensor及PIL Image对象的操作
    utils：含两个函数，一个是make_grid,它能将多张图片拼接在一个网格中；另一个是save_img，它能将Tensor保存成图片
"""

"""
utils.data简介：
    utils.data包括Dataset和DataLoader。torch.utils.data.Dataset为一个抽象类，自定义数据集需要继承这个类，并实现两个函数，一个是__len__
    ，另一个是__getitem__，前者提供数据的大小，后者通过索引获取数据和标签，__getitem__一次只能获取一个数据，所以需要通过
    torch.utils.data.DataLoader来定义一个新的迭代器，实现batch读取
"""
# import torch
# from torch.utils import data
# import numpy as np
#
# # 定义获取数据集的类，该类继承Dataset，自定义数据集及对应标签
# class TestDataset(data.Dataset):  # 继承Dataset
#     def __init__(self):
#         self.Data = np.asarray([[1.2],[3,4],[2,1],[3,4],[4,5]])  # 一些由2维向量表示的数据集
#         self.Label = np.asarray([0,1,0,1,2])  # 数据集对应的标签
#
#     def __getitem__(self, index):
#         # 把numpy转换为tensor
#         txt =torch.tensor(self.Data[index])
#         label = torch.tensor(self.Label[index])
#         return txt,label
#     def __len__(self):
#         return len(self.Data)
#
# # 获取数据集中的数据
# Test = TestDataset()
# print(Test[2])  # 相当于调用__getitem__(2)
# print(Test.__len__())
"""
(tensor([2, 1]), tensor(0, dtype=torch.int32))
5
"""

"""
以上数据以tuple的形式返回，每次只返回一个样本，实际上Dataset只负责数据的抽取，调用一次__getitem__只返回一个样本，如果希望批处理，还用同时进行
shuffle，并行加速等操作，可选择Dataloader，DataLoader格式为:
data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=<function default_collate at 0x7f108ee01620>,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
)
主要参数说明
dataset：加载的数据集。
·batch_size：批大小。
·shuffle：是否将数据打乱。
·sampler：样本抽样。
·num_workers：使用多进程加载的进程数，0代表不使用多进程。
·collate_fn：如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可。
·pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快
一些。
·drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多
出来不足一个batch的数据丢弃。
"""

# test_loader = data.DataLoader(Test,batch_size=2,shuffle=False,num_workers=0)  # 在windows上，num_workers一般设为0，不然会报错
# for i ,traindata in enumerate(test_loader):
#     print('i:',i)
#     Data,Label = traindata
#     print('Data:',Data)
#     print('Label:',Label)
"""
i: 0
data: tensor([[1, 2],
[3, 4]])
Label: tensor([0, 1])
i: 1
data: tensor([[2, 1],
[3, 4]])
Label: tensor([0, 1])
i: 2
data: tensor([[4, 5]])
Label: tensor([2])

我们可以像使用迭代器一样使用它，比如对它
进行循环操作。不过由于它不是迭代器，我们可以通过iter命令将其转换为迭代器。
"""
# dataiter=iter(test_loader)
# imgs,labels=next(dataiter)


# torchvision介绍
"""
torchvision有4个功能模块：model、datasets、transforms和utils

transforms:
    1）对PIL Image的常见操作如下。
    ·Scale/Resize：调整尺寸，长宽比保持不变。
    ·CenterCrop、RandomCrop、RandomSizedCrop：裁剪图片，CenterCrop和
    RandomCrop在crop时是固定size，RandomResizedCrop则是random size的crop。
    ·Pad：填充。
    ·ToTensor：把一个取值范围是[0,255]的PIL.Image转换成Tensor。形状为（H,W,C）的
    Numpy.ndarray转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloatTensor。
    ·RandomHorizontalFlip：图像随机水平翻转，翻转概率为0.5。
    ·RandomVerticalFlip：图像随机垂直翻转。
    ·ColorJitter：修改亮度、对比度和饱和度。
    2）对Tensor的常见操作如下。
    ·Normalize：标准化，即，减均值，除以标准差。
    ·ToPILImage：将Tensor转为PIL Image
如果要对数据集进行多个操作，可通过Compose将这些操作像管道一样拼接起来，类
似于nn.Sequential

transforms.Compose([
    #将给定的 PIL.Image 进行中心切割，得到给定的 size，
    #size 可以是 tuple，(target_height, target_width)。
    #size 也可以是一个 Integer，在这种情况下，切出来的图片形状是正方形。
    transforms.CenterCrop(10),
    #切割中心点的位置随机选取
    transforms.RandomCrop(20, padding=0),
    #把一个取值范围是 [0, 255] 的 PIL.Image 或者 shape 为 (H, W, C) 的 numpy.ndarray，
    #转换为形状为 (C, H, W)，取值范围是 [0, 1] 的 torch.FloatTensor
    transforms.ToTensor(),
    #规范化到[-1,1]
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])
"""

# ImageFolder
"""
当文件依据标签处于不同文件下时:
─── data
├── zhangliu
│ ├── 001.jpg
│ └── 002.jpg
├── wuhua
│ ├── 001.jpg
│ └── 002.jpg
.................
我们可以利用torchvision.datasets.ImageFolder来直接构造出dataset
loader = datasets.ImageFolder(path)
loader = data.DataLoader(dataset)

ImageFolder会将目录中的文件夹名自动转化成序列，当DataLoader载入时，标签自动
就是整数序列了

    from torchvision import transforms, utils
    from torchvision import datasets
    import torch
    import matplotlib.pyplot as plt
    %matplotlib inline
    my_trans=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder('./data/torchvision_data', transform=my_trans)
    train_loader = data.DataLoader(train_data,batch_size=8,shuffle=True,)
    for i_batch, img in enumerate(train_loader):
    if i_batch == 0:
    print(img[1])
    fig = plt.figure()
    grid = utils.make_grid(img[0])
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()
    utils.save_image(grid,'test01.png')
    break
    
"""

# 可视化工具
"""
Tensorboard是Google TensorFlow的可视化工具，它可以记录训练数据、评估数据、
网络结构、图像等，并且可以在web上展示，对于观察神经网络训练的过程非常有帮助。
PyTorch可以采用tensorboard_logger、visdom等可视化工具，但这些方法比较复杂或不够
友好。为解决这一问题，人们推出了可用于PyTorch可视化的新的更强大的工具——
tensorboardX。

tensorboardX功能很强大，支持scalar、image、figure、histogram、audio、text、
graph、onnx_graph、embedding、pr_curve and videosummaries等可视化方式。
"""

# 导入TensorboardX的步骤
# from tensorboardX import SummaryWriter
# # 实例化summaryWriter，并指明日志存放路径。在当前目录没有logs目录将自动创建
# writer = SummaryWriter(log_dir='logs')
# # 调用实例
# writer.add_xxx()
# # 关闭writer
# writer.close()

# TensorboardX可视化神经网络
# 导入需要的模块
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision
# from tensorboardX import SummaryWriter
#
# # 搭建神经网络
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.conv1 = nn.Conv2d(1,10,kernel_size=5)
#         self.conv2 = nn.Conv2d(10,20,kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320,50)
#         self.fc2 = nn.Linear(50,10)
#         self.bn = nn.BatchNorm2d(20)
#
#     def forward(self,x):
#         x = F.max_pool2d(self.conv1(x),2)
#         x = F.relu(x) + F.relu(-x)
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
#         x = self.bn(x)
#         x = x.view(-1,320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x,training=self.training)
#         x = self.fc2(x)
#         x = F.softmax(x,dim=1)
#         return x
#
# # 包模型保存为graph
# inout = torch.rand(32,1,28,28)
# # 实例化神经网络
# model = Net()
# # 将model保存为graph
# with SummaryWriter(log_dir='logs',comment='Net') as w:
#     w.add_graph(model,(input,))









