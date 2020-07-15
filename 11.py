import torch

a = torch.tensor([[1,3]])
b = torch.tensor([[3,3]])

print(torch.eq(a,b).float())
print(torch.argmax(a,dim=1))