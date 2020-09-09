from torch import nn
import torch
import thop


class NetV1(nn.Module):
    def __init__(self):
        super(NetV1, self).__init__()
        self.conv1 = nn.Conv2d(6, 30, 3, 1)

    def forward(self, input_):
        return self.conv1(input_)


class NetV2(nn.Module):
    def __init__(self):
        super(NetV2, self).__init__()
        self.conv1 = nn.Conv2d(6,30,3,1,groups=2)

    def forward(self, input_):
        return self.conv1(input_)


class NetV3(nn.Module):
    def __init__(self):
        super(NetV3, self).__init__()
        self.conv1 = nn.Conv2d(6,30,3,1,groups=3)

    def forward(self, input_):
        return self.conv1(input_)


class NetV4(nn.Module):
    def __init__(self):
        super(NetV4, self).__init__()
        self.conv1 = nn.Conv2d(6,30,3,1,groups=6)

    def forward(self, input_):
        return self.conv1(input_)


net1 = NetV1()
net2 = NetV2()
net3 = NetV3()
net4 = NetV4()
x = torch.randn(1, 6, 112, 112)
print(thop.clever_format(thop.profile(model=net1, inputs=(x,))))
print(thop.clever_format(thop.profile(net2, (x,))))
print(thop.clever_format(thop.profile(net3, (x,))))
print(thop.clever_format(thop.profile(net4, (x,))))

