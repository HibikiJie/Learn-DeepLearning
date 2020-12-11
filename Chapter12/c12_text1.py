import torch
import torchtext
from torchtext import vocab

"""实例化词向量对象"""
gv = torchtext.vocab.GloVe(name='6B')

"""打印词的长度"""
print(len(gv.vectors))

"""打印词的形状"""
print(gv.vectors.shape)

"""查询词索引位置"""
index = gv.stoi['tokyo']
print(index)

"""打印词向量"""
vector_tokyo = gv.vectors[index]
print(vector_tokyo)

"""由索引，查找词"""
print(gv.itos[index])



