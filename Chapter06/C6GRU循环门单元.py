from torch import nn
import torch

gru = nn.GRU(
    input_size=16,
    hidden_size=64,
    num_layers=7
)
"""输入数据形状，（S, N, V_input_size）:(32, 128, 16)"""
input_ = torch.randn(32, 128, 16)
output, h_n = gru(input_)
"""输出数据形状，（S, N, V_hidden_size）:(32, 128, 64)"""
print(output.shape)
print(h_n.shape)