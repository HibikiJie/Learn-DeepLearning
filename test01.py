import torch
from torch import nn
import numpy
rnn = nn.RNN(10,5,6,batch_first=True)#NSV

input = torch.randn(50,30,10)
h0 = torch.zeros(6,50,5)

output,hn = rnn(input,h0)
print(output.shape)#50,30,5
print(hn.shape)
numpy.all()
nn.GRU