from torch.utils.data import Dataset,DataLoader
from torchvision.models import densenet121
from torchvision.transforms import ToTensor
from PIL import Image
import torch
import os


class FCDataSet(Dataset):

    def __init__(self, root='D:/data/object2/casiafaceV5'):
        super(FCDataSet, self).__init__()
        self.data_set = []
        self.to_tensor = ToTensor()
        for target in os.listdir(root):
            for file_name in os.listdir(f'{root}/{target}'):
                file_path = f'{root}/{target}/{file_name}'
                self.data_set.append((file_path, target))
        print('数据初始化完成')

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        image_path, target = self.data_set[item]
        image = Image.open(image_path)
        image = image.resize((112,112),Image.ANTIALIAS)
        image_tensor = self.to_tensor(image)-0.5
        target = torch.tensor(int(target))
        return image_tensor, target
