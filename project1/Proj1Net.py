from torch import nn
import torch


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.PReLU(),

            nn.Conv2d(10, 16, 3, 1),
            nn.PReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.PReLU()
        )
        self.classification = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Conv2d(32, 4, 1)

    def forward(self, enter):
        enter = self.conv1(enter)
        return self.classification(enter), self.regression(enter)


class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.PReLU(),

            nn.Conv2d(28, 48, 3, 1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.PReLU(),

            nn.Conv2d(48, 64, 2, 1),
            nn.PReLU()
        )
        self.full_connect = nn.Sequential(
            nn.Linear(3 * 3 * 64, 128),
            nn.PReLU()
        )
        self.classification = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Linear(128, 4)

    def forward(self, enter):
        enter = self.full_connect(self.convolution(enter).reshape(-1, 576))
        return self.classification(enter), self.regression(enter)


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.PReLU(),

            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(3, 2),
            nn.PReLU(),

            nn.Conv2d(64, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.PReLU(),

            nn.Conv2d(64, 128, 2, 1),
            nn.PReLU()
        )
        self.fully_connect = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU()
        )
        self.classification = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.regression = nn.Linear(256, 4)

    def forward(self, enter):
        enter = self.fully_connect(self.convolution(enter).reshape(-1, 1152))
        return self.classification(enter), self.regression(enter)


if __name__ == '__main__':
    x = torch.randn(1, 3, 12, 12)

    net = PNet()
    print(net(x)[0].shape,net(x)[1].shape)
