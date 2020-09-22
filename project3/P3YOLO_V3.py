from torch import nn
import torch
import thop


class ConvolutionLayer(nn.Module):
    """
    卷积层，包含了’卷积‘、‘BatchNorm批归一化’、‘PReLU激活函数’

    参数：
        in_channels：输入通道数；
        out_channels：输出通道数；
        kernel_size：刻的大小；
        stride：步长
        padding：填充像素的多少
        bias：是否添加偏移
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1):
        super(ConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.convolution = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
                groups=self.groups
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, input_):
        return self.convolution(input_)


class DownSampling(nn.Module):
    """
    通过卷积的形式完成下采样，以2的步长快速减小特征图的尺寸。

    参数：
        in_channels：输入通道数；
        out_channels：输出通道数。
    """

    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.convolution = ConvolutionLayer(self.in_channels, self.out_channels, 3, 2, 1)

    def forward(self, input_):
        return self.convolution(input_)


class ResidualBlock(nn.Module):
    """
    残差块，使用1x1的卷积融合压缩通道，再使用3x3的卷积拓展回通道，并做残差。

    参数：
        in_channels：输入通道数，输出通道数与输入通道数保持一致。
    """

    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = self.in_channels // 2
        self.residual_block = nn.Sequential(
            ConvolutionLayer(
                self.in_channels,
                self.mid_channels,
                kernel_size=1
            ),
            ConvolutionLayer(
                self.mid_channels,
                self.in_channels,
                kernel_size=3,
                padding=1,
                groups=self.mid_channels
            )
        )

    def forward(self, input_):
        return self.residual_block(input_) + input_


class ConvolutionSet(nn.Module):
    """
    卷积集合，使用1x1和3x3大小的卷积核，以瓶颈解构构成了五层卷积网络
    要求：输入通道数大于输入通道数

    参数：
        in_channels：输入通道数；
        out_channels：输出通道数。
    """

    def __init__(self, in_channels, out_channels):
        super(ConvolutionSet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.con_set = nn.Sequential(
            ConvolutionLayer(in_channels, out_channels, 1),
            ConvolutionLayer(out_channels, in_channels, 3, padding=1, groups=self.out_channels),
            ConvolutionLayer(in_channels, out_channels, 1),
            ConvolutionLayer(out_channels, in_channels, 3, padding=1, groups=self.out_channels),
            ConvolutionLayer(in_channels, out_channels, 1)
        )

    def forward(self, input_):
        return self.con_set(input_)


class UpSimpling(nn.Module):
    """
    上采样，这里使用机器学习的临近值插值法，进行上采样。
    因为深层的特征信息量不足，同时需要与更大尺寸的特征图做cat，故上采样。
    """

    def __init__(self):
        super(UpSimpling, self).__init__()

    def forward(self, input_):
        return nn.functional.interpolate(input_, scale_factor=2, mode='nearest')


class YOLOVision3Net(nn.Module):
    """
    YOlO v3的网络。使用的为暗黑网络53层（DarkNet53）的网络。

    参数：
        out_channels：输出通道数。此数字为(4+1+c)*k；
        k为一个尺度有多少的检测框，c为检测的类别数，4为边框回归（中心点，长宽），1为是否有目标的置信度。
    """

    def __init__(self, out_channels=84):
        super(YOLOVision3Net, self).__init__()
        self.out_channels = out_channels

        '''实例化，暗黑网络的1~26层'''
        self.feature_map52x52 = nn.Sequential(
            ConvolutionLayer(3, 32, 3, padding=1),
            DownSampling(32, 64),
            ResidualBlock(64),
            DownSampling(64, 128),

            ResidualBlock(128),
            ResidualBlock(128),

            DownSampling(128, 256),

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        '''实例化，暗黑网络的27~43层'''
        self.feature_map26x26 = nn.Sequential(
            DownSampling(256, 512),

            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )

        '''实例化，暗黑网络的44~52层'''
        self.feature_map13x13 = nn.Sequential(
            DownSampling(512, 1024),
            ResidualBlock(1024),
            ResidualBlock(1024),
            ResidualBlock(1024),
            ResidualBlock(1024),
        )

        '''实例化，检测13x13尺度的卷积集（ConvolutionSet）'''
        self.con_set13x13 = ConvolutionSet(1024, 512)

        '''实例化，大尺度目标的输出网络层'''
        self.predict1 = nn.Sequential(
            ConvolutionLayer(512, 1024, 3, padding=1, groups=512),
            nn.Conv2d(1024, self.out_channels, 1, 1, 0)
        )

        '''实例化，13x13变换至26x26的上采样，网络层'''
        self.up_to_26x26 = nn.Sequential(
            ConvolutionLayer(512, 256, 1),
            UpSimpling()
        )

        '''实例化，检测26x26尺度的卷积集（ConvolutionSet）'''
        self.con_set26x26 = ConvolutionSet(768, 256)

        '''实例化，中尺度目标的输出网络层'''
        self.predict2 = nn.Sequential(
            ConvolutionLayer(256, 512, 3, padding=1, groups=256),
            nn.Conv2d(512, self.out_channels, 1)
        )

        '''实例化，26x26变换至52x52的上采样，网络层'''
        self.up_to_52x52 = nn.Sequential(
            ConvolutionLayer(256, 128, 1),
            UpSimpling()
        )

        '''实例化，检测52x52尺度的卷积集（ConvolutionSet）'''
        self.con_set52x52 = ConvolutionSet(384, 128)

        '''实例化，小尺度目标的输出网络层'''
        self.predict3 = nn.Sequential(
            ConvolutionLayer(128, 256, 3, padding=1, groups=128),
            nn.Conv2d(256, self.out_channels, 1)
        )

    def forward(self, input_):
        """获得52x52的特征图"""
        feature_map52x52 = self.feature_map52x52(input_)

        '''获得26x26的特征图'''
        feature_map26x26 = self.feature_map26x26(feature_map52x52)

        '''获得13x13的特征图'''
        feature_map13x13 = self.feature_map13x13(feature_map26x26)

        '''侦测13x13的特征图，并输出结果'''
        con_set_13 = self.con_set13x13(feature_map13x13)
        predict1 = self.predict1(con_set_13)

        '''上采样至26x26，并与暗黑网络输出的26x26的特征图做concatenate'''
        up_26 = self.up_to_26x26(con_set_13)
        concatenated26x26 = torch.cat((up_26, feature_map26x26), dim=1)

        '''侦测26x26的特征图，并通过输出层输出结果'''
        con_set_26 = self.con_set26x26(concatenated26x26)
        predict2 = self.predict2(con_set_26)

        '''上采样至52x52，并与暗黑网络输出的52x52的特征图做concatenate'''
        up_52 = self.up_to_52x52(con_set_26)
        concatenated52x52 = torch.cat((up_52, feature_map52x52), dim=1)

        '''侦测52x52的特征图，并通过输出层输出结果'''
        con_set_52 = self.con_set52x52(concatenated52x52)
        predict3 = self.predict3(con_set_52)
        return (predict1.permute(0, 2, 3, 1).reshape(-1, 13, 13, 3, self.out_channels // 3),
                predict2.permute(0, 2, 3, 1).reshape(-1, 26, 26, 3, self.out_channels // 3),
                predict3.permute(0, 2, 3, 1).reshape(-1, 52, 52, 3,self.out_channels // 3)
                )


if __name__ == '__main__':
    yolo3 = YOLOVision3Net(45)
    torch.save(yolo3.state_dict(), 'yolo.pth')
    print(yolo3)
    x = torch.randn(1, 3, 416, 416)
    # a = thop.profile(yolo3,(x,))
    # print(thop.clever_format(a))
    y1, y2, y3 = yolo3(x)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
