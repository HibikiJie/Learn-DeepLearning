import torch

'''通道混洗(Channel shuffle)'''
'''目的：在不增加计算量的情况下，使通道充分融合'''
x = torch.randn(4, 6, 3, 3)
y = x.reshape(4, 2, 3, 3, 3)
print(y.shape)
z = y.permute(0, 2, 1, 3, 4)
print(z.shape)
v = z.reshape(4, 6, 3, 3)
print(v.shape)
