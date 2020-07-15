#实现LeNet5
import torch
import torch.nn as nn
import torch.nn.functional as F
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
class Net(nn.Module):
    def __init__(self):
        # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        super(Net, self).__init__()
        # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即RGB图）,输出为 6张特征图, 卷积核为5x5正方形
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上
        self.fc2 = nn.Linear(120, 84)
        # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上
        self.fc3 = nn.Linear(84, 10)
    # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        # 输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
        x = x.view(-1, 16 * 5 * 5)
        print(x.shape,'=======')
        # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc1(x))
        # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc2(x))
        # 输入x经过全连接3，然后更新x
        x = self.fc3(x)
        return x

net = Net()
# 以下代码是为了看一下我们需要训练的参数的数量
print(net)
#打印网格参数 即多少个tensor输出
params = list(net.parameters())
# print(params)
print(len(params))
#打印某一层参数的形状，这里是通过次卷积和池化的结果
print(params[0].size())
#随机输入一个向量，查看前向传播输出
input = torch.randn(1, 3, 32, 32)
out = net(input)
print(out)
#将梯度初始化
net.zero_grad()
#随机一个梯度进行反向传播
out.backward(torch.randn(1, 10))
#用自带的MSELoss()定义损失函数
criterion = nn.MSELoss()
#随机一个真值，并用随机的输入计算损失
target = torch.randn(10)  # 随机真值
target = target.view(1, -1)  # 变成行向量
output = net(input)  # 用随机输入计算输出
loss = criterion(output, target)  # 计算损失
print(loss)
#将梯度初始化，计算上一步中loss的反向传播
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
#计算loss的反向传播
loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
#更新权重 定义SGD优化器算法，学习率设置为0.01
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01)
#使用优化器更新权重
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
# 更新权重
optimizer.step()