import torch
import torchtext
from torchtext import vocab

"""实例化词向量对象"""
gv = vocab.GloVe(name='6B')


def get_word_vector(word):
    """
    取出词的向量
    :param word: 词
    :return: 词向量
    """
    index = gv.stoi[word]
    return gv.vectors[index]


def sim_10(word, n=10, way='cosine_similarity'):
    """
    取出这个词向量，拿这个向量去遍历所有的向量，求距离，拉出10个最近的词
    :param way: 衡量方法
    :param word: 词向量
    :param n: 范围
    :return: 最相似的词向量
    """
    if way == 'cosine_similarity':
        func = torch.cosine_similarity
        reverse = True
    else:
        func = torch.dist
        reverse = False
    all_cosine_similarity = []
    for i, w in enumerate(gv.vectors):
        if reverse:
            temp = func(word, w, dim=0)
        else:
            temp = func(word, w)
        all_cosine_similarity.append((gv.itos[i],temp))
    return sorted(all_cosine_similarity,key=lambda t:t[1],reverse=reverse)[:n]


"""打印和'tokyo'最相似的词向量"""
vector_tokyo = get_word_vector('tokyo')
a = sim_10(vector_tokyo)
print(a)
