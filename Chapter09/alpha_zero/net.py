from torch import nn
import torch


class AlphaZero(nn.Module):

    def __init__(self, board_size):
        super(AlphaZero, self).__init__()
        """棋盘尺寸"""
        self.board_size = board_size
        """特征提取主干网络"""
        self.main_net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.Hardswish(),
            nn.Conv2d(32,64, kernel_size=3, padding=1),
            nn.Hardswish(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Hardswish(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Hardswish(),
        )

        """策略网络，输出概率"""
        self.policy_net_layer1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Hardswish(),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.Hardswish(),
        )
        self.policy_net_layer2 = nn.Sequential(
            nn.Linear(32*self.board_size*self.board_size, 8*self.board_size*self.board_size),
            nn.Hardswish(),
            nn.Linear(8 * self.board_size * self.board_size, self.board_size * self.board_size),
            nn.Hardswish(),
            nn.Softmax(dim=1)
        )

        """估值网络，输出当前局面的胜率V"""
        self.val_net_layer1 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,padding=1),
            nn.Hardswish(),
            nn.Conv2d(128, 16, kernel_size=3, padding=1),
            nn.Hardswish(),
        )
        self.val_net_layer2 = nn.Sequential(
            nn.Linear(16 * self.board_size * self.board_size, 4 * self.board_size * self.board_size),
            nn.Hardswish(),
            nn.Linear(4 * self.board_size * self.board_size, 64),
            nn.Hardswish(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.main_net(x)
        p = self.policy_net_layer2(self.policy_net_layer1(x).reshape(-1, 32 * self.board_size * self.board_size))
        v = self.val_net_layer2(self.val_net_layer1(x).reshape(-1, 16 * self.board_size * self.board_size))
        return p.reshape(-1, self.board_size, self.board_size), v


if __name__ == '__main__':
    m = AlphaZero(9)
    x = torch.randn(1,2,9,9)
    print(m(x)[0],m(x)[1].shape)
