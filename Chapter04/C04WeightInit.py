from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter

def weight_init(m):
    if isinstance(m, nn.Conv2d):  # 判断该对象是否是一个已知的类型
        nn.init.kaiming_normal_(m.weight)  # 进行凯明初始化
        if m.bias is not False:
            nn.init.zeros_(m.bias)  # 零初始化
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight)  # 正态分布初始化
        if m.bias is not False:
            nn.init.zeros_(m.bias)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
        )
        self.layer1 = nn.Linear(128, 10)
        self.apply(weight_init)

    def forward(self, input_):
        input_ = self.conv1(input_).reshape(-1, 128)
        return self.layer1(input_)


if __name__ == '__main__':
    net = Net()
    print(net.conv1[1].bias)
    # print(net.conv1[0].weight.mean())
    # print(net.conv1[0].weight.var()**2)
    # summary_writer = SummaryWriter('D:/data/chapter4/logs')
    # layer1 = net.conv1[0].weight
    #
    #
    # summary_writer.add_histogram('layer1',layer1)
    # while True:
    #     print(1)
