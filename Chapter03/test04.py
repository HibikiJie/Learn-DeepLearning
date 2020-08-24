import torch
#
a = torch.tensor([0.99],dtype=torch.float32,requires_grad=True)
print(-(1-a**2)**(-0.5))
y = torch.acos(a)
y.backward()
print(a.grad)
# c = torch.tensor([-1.])
# d =  torch.tensor([-2.])
# a = 1/(9*torch.exp(c)+1)
# b = torch.exp(d)/(8*torch.exp(c)+torch.exp(d)+torch.exp(torch.tensor([1.])))
# aaaa = torch.tensor([a,b,b,b,b,b,b,b,b,b])
# print(aaaa)
# print(-torch.log(a))
#
# bbbb= torch.exp(aaaa)/torch.exp(aaaa).sum()
# print(bbbb)
#
# print(y)
# print(y.grad_fn)
# y.backward()
#
# print(a.grad)
# a = torch.tensor([[0.1,0.2,0.99],[0.2,0.99999,-0.5],[-0.1,1,-1]])
# c = a.clone()
# c= -(1-c**2)**(-0.5)
# print(c)
# b = torch.acos(a)
# print(b)
# print(-(1-a**2)**(-0.5))
# b[-0.9999<a<0.9999] = c[0.9999<a<0.9999]
# b[a>0.9999] = -70.7048
# b[a < -0.9999] = -70.7048
# print(b)
# print(-(1-0.99999**2)**(-0.5))
# a = torch.randn(3,3,3)
# b = torch.randn(3,3,3)
# print(b)
# print(a>0)
# b[a>0] = 0
# print(b)

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