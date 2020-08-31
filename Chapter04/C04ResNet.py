from torch import nn
import torch


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, 1, 1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(16)
        )

    def forward(self, input_):
        return self.res_conv1(input_) + input_
