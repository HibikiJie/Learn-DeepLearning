from torch import nn
import torch


class BottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, t, stride=1):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.t = t
        self.stride = stride
        self.is_res = self.in_channels == self.out_channels
        self.bottle_neck = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels * self.t, 1, 1, bias=False),
            nn.BatchNorm2d(self.in_channels * self.t),
            nn.ReLU6(),
            nn.Conv2d(
                self.in_channels * self.t,
                self.in_channels * self.t,
                3,
                self.stride,
                1,
                groups=self.in_channels * self.t,
                bias=False
            ),
            nn.BatchNorm2d(self.in_channels * self.t),
            nn.ReLU6(),
            nn.Conv2d(self.in_channels * self.t, self.out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, input_):
        if self.is_res:
            return self.bottle_neck(input_) + input_
        else:
            return self.bottle_neck(input_)


class ConvolutionSet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvolutionSet, self).__init__()
        self.convolution_set = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
            nn.Conv2d(out_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(),
            nn.Conv2d(in_channels, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, input_):
        return self.convolution_set(input_)


class UpSimpling(nn.Module):

    def __init__(self):
        super(UpSimpling, self).__init__()

    def forward(self, input_):
        return nn.functional.interpolate(input_, scale_factor=2, mode='nearest')


class MobileNetV2(nn.Module):

    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.features52 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),

            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),

            BottleNeck(16, 24, 6, 2),
            BottleNeck(24, 24, 6, 1),

            BottleNeck(24, 32, 6, 2),
            BottleNeck(32, 32, 6),
            BottleNeck(32, 32, 6),
        )
        self.features26 = nn.Sequential(
            BottleNeck(32, 64, 6, 2),
            BottleNeck(64, 64, 6),
            BottleNeck(64, 64, 6),
            BottleNeck(64, 64, 6),
        )
        self.features13 = nn.Sequential(
            BottleNeck(64, 96, 6),
            BottleNeck(96, 96, 6),
            BottleNeck(96, 96, 6),

            BottleNeck(96, 160, 6, 2),
            BottleNeck(160, 160, 6, 1),
            BottleNeck(160, 160, 6, 1),
            BottleNeck(160, 320, 6, 1),
            nn.Conv2d(320, 1280, 1, 1)
        )
        self.con_set13 = ConvolutionSet(1280, 640)
        self.predict1 = nn.Sequential(
            nn.Conv2d(640, 320, 3, 1, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(),
            nn.Conv2d(320, 75, 1, 1, bias=False)
        )
        self.up_to26 = nn.Sequential(
            nn.Conv2d(640, 32, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            UpSimpling()
        )
        self.con_set26 = ConvolutionSet(96, 160)
        self.predict2 = nn.Sequential(
            nn.Conv2d(160, 320, 3, 1, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(),
            nn.Conv2d(320, 75, 1, 1, bias=False)
        )
        self.con_set52 = ConvolutionSet(48, 96)
        self.up_to52 = nn.Sequential(
            nn.Conv2d(160, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(),
            UpSimpling()
        )
        self.predict3 = nn.Sequential(
            nn.Conv2d(96, 160, 3, 1, 1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU6(),
            nn.Conv2d(160, 75, 1, 1, bias=False)
        )

    def forward(self, input_):
        features52 = self.features52(input_)
        features26 = self.features26(features52)
        features13 = self.features13(features26)
        con_set13 = self.con_set13(features13)
        predict1 = self.predict1(con_set13)
        up_to26 = self.up_to26(con_set13)
        con_set26 = self.con_set26(torch.cat((up_to26, features26), dim=1))
        predict2 = self.predict2(con_set26)
        up_to52 = self.up_to52(con_set26)
        con_set52 = self.con_set52(torch.cat((up_to52, features52), dim=1))
        predict3 = self.predict3(con_set52)
        return (
            predict1.permute(0, 2, 3, 1).reshape(-1, 13, 13, 3, 25),
            predict2.permute(0, 2, 3, 1).reshape(-1, 26, 26, 3, 25),
            predict3.permute(0, 2, 3, 1).reshape(-1, 52, 52, 3, 25)
        )


if __name__ == '__main__':
    m = MobileNetV2()
    print(m)
    # torch.save(m.state_dict(),'a.pth')