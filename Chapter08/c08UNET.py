from torch import nn
import torch


class CNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        卷积层
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        """
        super(CNNLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, input_):
        return self.layer(input_)


class DownSample(nn.Module):

    def __init__(self, in_channels):
        """
        步长为2的卷积，下采样
        :param in_channels: 输入通道数
        """
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, input_):
        return self.layer(input_)


class UpSample(nn.Module):

    def __init__(self, in_channels):
        """
        反卷积，上采样
        :param in_channels: 输入通道数
        """
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 3, 2, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, input_, concat):
        """
        前向运算过程
        :param input_: 输入
        :param concat: 浅层特征
        :return: 完成上采样后和浅层特征concat的结果
        """
        return torch.cat((concat, self.layer(input_)), dim=1)


class UNet(nn.Module):

    def __init__(self):
        """
        UNet网络，结构类似编解码，
        """
        super(UNet, self).__init__()
        self.c1 = CNNLayer(3, 64)
        self.d1 = DownSample(64)
        self.c2 = CNNLayer(64, 128)
        self.d2 = DownSample(128)
        self.c3 = CNNLayer(128, 256)
        self.d3 = DownSample(256)
        self.c4 = CNNLayer(256, 512)
        self.d4 = DownSample(512)
        self.c5 = CNNLayer(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = CNNLayer(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = CNNLayer(512, 256)
        self.u3 = UpSample(256)
        self.c8 = CNNLayer(256, 128)
        self.u4 = UpSample(128)
        self.c9 = CNNLayer(128, 64)
        self.predict1 = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, image_tensor):
        c1 = self.c1(image_tensor)
        c2 = self.c2(self.d1(c1))
        c3 = self.c3(self.d2(c2))
        c4 = self.c4(self.d3(c3))
        c5 = self.c5(self.d4(c4))
        c6 = self.c6(self.u1(c5, c4))
        c7 = self.c7(self.u2(c6, c3))
        c8 = self.c8(self.u3(c7, c2))
        c9 = self.c9(self.u4(c8, c1))
        return self.predict1(c9)


if __name__ == '__main__':
    net = UNet()
    x = torch.randn(2,3,64,64)
    print(net(x).shape)
