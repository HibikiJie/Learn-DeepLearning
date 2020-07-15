import torch
import numpy

a = '''创建一个空的5X3的tensor矩阵'''
print(a.center(60,'='))

x = torch.empty(5,3)
print(x)

# =================================================
a = '''创建一个随机初始化的5X3的tensor矩阵'''
print(a.center(60,'='))

x = torch.rand(5,3)
print(x)


# =================================================
a = '''创建一个5X3的零的tensor矩阵，类型为long'''
print(a.center(60,'='))

x = torch.zeros(5,3,dtype=torch.long)
print(x)


# =================================================
a = '''将列表转换为tensor类型'''
print(a.center(60,'='))

x = torch.tensor([5,3])
print(x)


# =================================================
a = '''创建一个5X3的单位矩阵，类型为double'''
print(a.center(60,'='))

x = torch.ones(5,3,dtype=torch.double)
print(x)


# =================================================
a = '''从已有的tensor矩阵创建相同维度的新的tensor矩阵，并且重新定义类型为float'''
print(a.center(60,'='))

x = torch.randn_like(x, dtype=torch.float)
print(x)


# =================================================
a = '''打印一个tensor矩阵的维度'''
print(a.center(60,'='))

print(x.size())


# =================================================
a = '''将两个矩阵相加'''
print(a.center(60,'='))

y = torch.rand(5,3)
print(x+y)
print(torch.add(x,y))


# =================================================
a = '''取tensor矩阵的第一列'''
print(a.center(60,'='))

print('='*50)
print(x[:,0])


# =================================================
a = '''将一个4X4的tensor矩阵resize成一个一维tensor列表'''
print(a.center(60,'='))

x = torch.randn(4,4)
y = x.view(16)
print(x.size(),y.size())


# =================================================
a = '将一个4X4的tensor矩阵，resize成一个2X8的tensor矩阵'
print(a.center(60,'='))

y = x.view(2,8)
print(x.size(),y.size())


# =================================================
a = '''从tensor矩阵中取出数字'''
print(a.center(60,'='))

x = torch.randn(1)
print(x)
print(x.item())


# =================================================
a = '''将tensor数组转换成numpy数组'''
print(a.center(60,'='))

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)


# =================================================
a = '''从numpy数组创建tensor数组'''
print(a.center(60,'='))


a = numpy.ones(5)
b = torch.from_numpy(a)