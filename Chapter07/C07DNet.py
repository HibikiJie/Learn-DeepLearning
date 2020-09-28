from torch import nn
import torch


class DBlock(nn.Module):

    def __init__(self, c):
        super(DBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(),
            nn.Conv2d(c, c, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(),
        )

    def forward(self, input_):
        return self.layer(input_) + input_


class DownSample(nn.Module):

    def __init__(self, in_channels,out_channels):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

    def forward(self, input_):
        return self.layer(input_)


class DNet(nn.Module):

    def __init__(self):
        super(DNet, self).__init__()
        self.layer = nn.Sequential(
            DownSample(3, 32),
            DBlock(32),

            DownSample(32, 64),
            DBlock(64),

            DownSample(64, 128),
            DBlock(128),
            DownSample(128, 256),
            DBlock(256),

            DownSample(256, 512),
            DBlock(512),

            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
        )

    def forward(self,input_):
        return self.layer(input_).reshape(-1,1)


if __name__ == '__main__':
    m = DNet()

    x = torch.randn(2,3,128,128)
    print(m(x).shape)