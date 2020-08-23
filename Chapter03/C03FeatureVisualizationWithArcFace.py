from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from Chapter03.C03ArcFaceLoss import ArcFace
import torch
import numpy
import matplotlib.pyplot as plt


'''构建卷积神经网络'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.convolution1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,3,2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1),
            nn.ReLU(),
        )
        self.full_connect1 = nn.Sequential(
            nn.Linear(256, 2),
        )

    def forward(self, enter):
        return self.full_connect1(self.convolution1(enter).reshape(-1, 256))


def visualize(features, labels, epoch):
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

    '''清空滑板'''
    plt.clf()

    '''循环开始，依次绘制类别0的点、类别1的点……绘制类别9的点'''
    for i in range(10):
        index = labels == i
        plt.plot(features[index, 0], features[index, 1], '.', c=color[i])

    '''绘制图形的标记'''
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.title('epoch=%d' % epoch)  # 绘制抬头
    plt.savefig('D:/data/chapter3/image6/epoch%d' % epoch)  # 保存图片
    plt.pause(1)  # 暂停


if __name__ == '__main__':

    '''加载手写数字(MNIST)的数据集，使用pytorch自带的'''
    dataset = MNIST(root='D:/data/chapter3', transform=ToTensor())
    data_loader1 = DataLoader(dataset, 512, True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    '''实例化模型，优化器，损失函数'''
    net = Net().to(device)
    arc_face = ArcFace(2, 10, 1).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam([{'params':net.parameters()}, {'params': arc_face.parameters()}])
    i = 0

    for epoch in range(1000):
        features = []
        labels = []

        '''循环batch训练网络'''
        for image, target in data_loader1:
            image = image.to(device)
            target = target.to(device)
            '''输入网络，计算损失'''
            feature = net(image)
            out = arc_face(feature)
            loss = loss_function(out,target)

            '''清空梯度、方向求导、优化参数'''
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            features.append(nn.functional.normalize(feature.detach().cpu(),dim=1).numpy())
            labels.append(target.detach().cpu())
            txt = loss.item()
            print(loss.item())
            i += 1

        torch.save(net.state_dict(), 'D:/data/chapter3/net6.pt')
        features = numpy.vstack(features)
        labels = torch.cat(labels, 0).numpy()

        visualize(features, labels, epoch)