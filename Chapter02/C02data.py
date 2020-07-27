from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os
import torch,numpy


class DataYellow(Dataset):

    def __init__(self,root='D:/data/chapter2/shuju'):
        self.root = root
        super(DataYellow, self).__init__()
        self.dataset = os.listdir(self.root)
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,item):
        file_name = self.dataset[item]
        img = Image.open(f"{self.root}/{file_name}")
        img = self.totensor(img)
        target = file_name.split(".")[1:5]
        target = torch.tensor(numpy.array(target,dtype=numpy.float32))/300
        return img,target

if __name__ == '__main__':
    dataset = DataYellow()
    print(dataset[1][0].shape)