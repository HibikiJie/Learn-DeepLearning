from torch import nn
import torch

lstm = nn.LSTM(
    input_size=64,
    hidden_size=128,
    num_layers=7,
    bias=True,
    batch_first=True,
    dropout=0,
    bidirectional=False)

"""输入的形状为（N,S,V）:(128,30,64)"""
input_ = torch.randn(128,30,64)
c_0 = torch.randn(7,128,128)
h_0 = torch.randn(7,128,128)
output_, (h_n, c_n) = lstm(input_,(h_0,c_0))
print(output_.shape)
print(h_n.shape)
print(c_n.shape)
