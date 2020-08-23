import torch
#
a = torch.tensor([1],dtype=torch.float32,requires_grad=True)

y = torch.acos(a)

print(y)
print(y.grad_fn)
y.backward()
print(a.grad)
# a = torch.tensor([[1,2,3],[2,4,1]]).float()
# # c = torch.nn.functional.normalize(a,dim=1)
# c = torch.sum(a**2,dim=0,keepdim=True)**0.5
# print(c)
# print(torch.cos(torch.tensor(3.14)))
# b = torch.randn(15,8)
# a = torch.tensor([1,2,3,4,5,6,7,6,4,3,4,5,6,4,1]).unsqueeze(dim=1)
# print(a)
# # print(len(a))
# print(b)
# print(a.shape)
# print(b.gather(1,a))

# a = torch.randn(512,10)
# b = torch.randn(10,2)
# print(torch.matmul(a,b))