from torch import nn
import torch


class Mish(nn.Module):

    def forward(self, x):
        return x*torch.tanh(nn.functional.softplus(x))


class Swish(nn.Module):

    def forward(self, x):
        return x * x.sigmoid()
