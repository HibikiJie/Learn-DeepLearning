import torch

a = torch.randn(100,5).cuda()
b = torch.tensor([0.5],dtype=torch.float32).cuda()

print(torch.max(a,b))
