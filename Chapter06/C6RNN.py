from torch import nn
import torch
"""RNN，循环神经网络"""
rnn = nn.RNN(
    input_size=10,  # 输入的特征大小
    hidden_size=5,  # 隐藏层的尺寸，即输出的特征尺寸
    num_layers=3,  # 层数
    bias=False,  # 是否使用偏置
    batch_first=True,  # 是否使用批准在前的输入形状
    dropout=0,  # 随机失活比例
    bidirectional=False,  # 是否使用双向循环神经网络
    nonlinearity='tanh'  # 激活函数
)
input_ = torch.randn(128, 20, 10)  # 输入形状（N, S, V）

"""输出的形状（N, S, V），隐藏层状态形状（num_layers, N, 输出特征数）"""
output, hn = rnn(input_)
print(output.shape)
print(hn.shape)
