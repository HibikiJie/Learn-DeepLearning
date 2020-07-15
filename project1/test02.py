import torch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''定义conv1函数的图像是卷积函数：输入为图形（3个频道，即RBG图），输出为6张特征图，卷积核的大小为5X5正方形'''
        self.conv1 = torch.nn.Conv2d(3, 6, 5)

        '''定义conv2函数的是图像卷积函数：输入为六张特征图，输出为16张特征图，卷积核大小为5X5'''
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        '''定义f1 全连接函数1为线性函数，并将16*5*5个节点连接到120个节点上'''
        self.fc1 = torch.nn.Linear(16 * 5 * 5+10, 120)

        self.fc2 = torch.nn.Linear(120, 84)

        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x,x2):
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), (2, 2))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.cat((x,x2),1)
        print(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

print(net)

'''打印网络参数'''
params = list(net.parameters())
# print(params)

print(len(params))

input1 = torch.randn(1, 3, 32, 32)
print(input1.dtype)
input2 = torch.randn(1,10)
print(input2)
out = net.forward(input1,input2)
print(out)
