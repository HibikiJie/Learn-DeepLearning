from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import numpy
import torch
import os


class SunStormDataset(Dataset):

    def __init__(self,root='D:/data/object1/train'):
        self.root = root
        self.dataset = []
        self.para = torch.load('D:/data/object1/para')
        self.target = torch.load('D:/data/object1/target')
        print("加载数据中")
        for i,file_name in enumerate(os.listdir(root)):
            file_path = f'{root}/{file_name}'
            self.dataset.append((file_path,self.para[i],self.target[i]))
        print('加载完成，初始化权重中')
        self.weights = [20 if int(target.item()) == 1 else 1 for img,para,target in self.dataset]
        print('权重初始化完成')
        self.simpler = WeightedRandomSampler(self.weights,len(self.dataset),True)
        # print(self.weights)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        file_path,para_data,target_data = self.dataset[item]
        img = torch.load(file_path)
        return img[0],para_data[0],target_data


if __name__ == '__main__':
    pass
    dataset = SunStormDataset()
    data_loader = DataLoader(dataset,10,sampler=dataset.simpler)
    for img,para,target in data_loader:
        print(img.shape)
        print(para.shape)
        print(target.shape)
        break
