import torch
x = torch.randn(1,3)
print(x)
tmp  = torch.mul(x, x) # or x ** 2
tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True))
print(tmp1)