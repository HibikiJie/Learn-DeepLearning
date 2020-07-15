import torch
import numpy

'''新建一个张量，并设置requires_grad = True,表示该tensor需要求导'''
x = torch.ones(2,2,requires_grad=True)
print(x)

'''对张量进行任意操作（y = x +2）'''

y = x + 2
print(y)
print(y.grad_fn)

'''对y进行任意操作'''
z = y*y*3
out = z.mean()
print(z)
print(out)

'''对out进行反向传播'''
out.backward()

'''打印梯度d(out)/dx'''
print(x.grad)

'''创建一个结果为矢量的计算过程'''
x = torch.randn(3,requires_grad=True)
y = x*2
while y.data.norm()<1000:
    y = y*2
print(y)
