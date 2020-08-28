import torch
from torch import nn
from torchvision.models import densenet121

net = densenet121()
print(net)
# m = nn.Dropout(p=1)
# input = torch.randn(2, 2)
# print(input)
# output = m(input)
# print(output)
conv = nn.Conv2d(3,16,3,1)
print(conv.weight.mean())
print(conv.weight.var()**2)
nn.init.kaiming_normal_(conv.weight,nonlinearity='relu')
print(conv.weight.mean())
print(conv.weight.var()**2)
torch.nn.init.normal_()
torch.nn.init.zeros_()
torch.nn.init.xavier_normal_()