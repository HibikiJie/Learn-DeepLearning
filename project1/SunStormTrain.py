from SunStormDataset import SunStormDataset
from SunStormNet import Net
import torch
from torch.utils.data import DataLoader
print('模块导入成功')


class Train():

    def __init__(self):
        '''初始化数据加载器'''
        self.dataset = SunStormDataset()
        self.data_loader = DataLoader(self.dataset,10,sampler=self.dataset.simpler)
        '''初始化网络'''
        self.net = Net()
        self.opttim = torch.optim.Adam(self.net.parameters)


