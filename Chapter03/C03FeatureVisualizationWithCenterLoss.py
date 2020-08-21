from Chapter03.C03CenterLoss import CenterLoss
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import numpy
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.convolution1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, bias=False),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 3, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.full_connect1 = nn.Sequential(
            nn.Linear(128, 2, bias=False),
        )
        '''输出二维的特征向量，方便绘制于图上'''
        self.full_connect2 = nn.Sequential(
            nn.Linear(2, 10, bias=False)
        )

    def forward(self, enter):
        xy = self.full_connect1(self.convolution1(enter).reshape(-1, 128))

        '''将中间层的特征同时返回'''
        return self.full_connect2(xy), xy


def visualize(features, labels, epoch, point):
    '''
    将特征进行图形展示。
    :param features: 网络中间输出的特征向量
    :param labels: 数据标签
    :param epoch: 批次
    :return: None
    '''

    '''开启绘画'''
    plt.ion()
    color = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    print('1')
    '''清空滑板'''
    plt.clf()

    '''循环开始，依次绘制类别0的点、类别1的点……绘制类别9的点'''
    for i in range(10):
        index = labels == i
        plt.plot(features[index, 0], features[index, 1], '.', c=color[i])
    print('1')
    '''绘制图形的标记'''
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.title('epoch=%d' % epoch)  # 绘制抬头
    plt.plot(point[:, 0], point[:, 1], 'd', c='black')
    plt.savefig('D:/data/chapter3/image/epoch%d' % epoch)  # 保存图片
    plt.pause(1)  # 暂停


if __name__ == '__main__':
    '''加载手写数字(MNIST)的数据集，使用pytorch自带的'''
    dataset = MNIST(root='D:/data/chapter3', transform=ToTensor())
    data_loader1 = DataLoader(dataset, 51200, True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''实例化模型，优化器，损失函数'''
    net = Net().to(device)
    # net.load_state_dict(torch.load('D:/data/chapter3/net5.pt'))
    center_loss = CenterLoss(2, 10).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)

    optimizer1 = torch.optim.Adam([{'params': net.parameters()}, {'params': center_loss.parameters()}])
    for epoch in range(100000000):
        features = []
        labels = []
        # print(center_loss.center_point)
        # if alpha<1:
        #     alpha *=10
        '''循环batch训练网络'''
        for image, target in data_loader1:
            image = image.to(device)
            target = target.to(device)

            '''输入网络，计算损失'''
            out, feature = net(image)
            loss1 = loss_function(out, target)
            loss2 = center_loss.forward(feature, target)
            loss = loss1 + loss2
            '''清空梯度、方向求导、优化参数'''

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            print(loss1.item(), loss2.item())
            features.append(feature.detach().cpu().numpy())
            labels.append(target.detach().cpu())
        print('1')
        # torch.save(net.state_dict(), 'D:/data/chapter3/net5.pt')
        features = numpy.vstack(features)
        labels = torch.cat(labels, 0).numpy()
        print('1')
        point = center_loss.center_point.detach().cpu().numpy()
        visualize(features, labels, epoch + 1000, point)
