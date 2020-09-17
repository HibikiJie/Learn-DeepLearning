from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch


class RNNNet(nn.Module):

    def __init__(self):
        super(RNNNet, self).__init__()
        self.rnn_layer = nn.RNN(28, 64, 7, batch_first=True)
        self.classification = nn.Linear(64, 10)

    def forward(self, input_):
        """(N, C, H, W) ===> (N, S, V):(N, 28, 28)"""
        input_ = input_.reshape(-1, 28, 28)

        """output of shape is (N, S, V):(N, 28, 28)"""
        output, hn = self.rnn_layer(input_)

        """取最后一步的输出"""
        out = output[:, -1, :]  # 形状变为了（N, V）
        return self.classification(out)


if __name__ == '__main__':
    rnn_net = RNNNet()  # 实例化网络模型
    train_data_set = MNIST('D:/data', train=True, transform=ToTensor())  # 加载数据集
    test_data_set = MNIST('D:/data', train=False, transform=ToTensor())
    train_data_loader = DataLoader(train_data_set, 128, True)  # 实例化数据集加载器
    test_data_loader = DataLoader(test_data_set, 128, True)
    optimizer = torch.optim.Adam(rnn_net.parameters())  # 实例化优化器
    loss_func = nn.CrossEntropyLoss()  # 实例化损失函数
    while True:
        loss_sum = 0
        for image, target in train_data_loader:
            output = rnn_net(image)  # 预测

            loss = loss_func(output,target)  # 做损失

            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新梯度
            loss_sum += loss.item()
        correct = 0
        for image, target in test_data_loader:
            output = rnn_net(image)
            out = torch.argmax(output, dim=1)
            acc = out == target
            correct += acc.sum().item()
        print('loss:{},acc:{}'.format(loss_sum/len(train_data_loader),correct/len(test_data_set)))

