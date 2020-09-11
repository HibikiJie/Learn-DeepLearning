from torch.utils.data import Dataset,DataLoader
from torchvision.models import densenet121
from torchvision.transforms import ToTensor
from PIL import Image
import cv2
import numpy
import zipfile
import torch
import os


class FCDataSet(Dataset):

    def __init__(self, root='D:/data/object2/dataset'):
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
        image = image.convert('RGB')
        image = image.resize((112,112),Image.ANTIALIAS)
        image_tensor = self.to_tensor(image)
        target = torch.tensor(int(target))
        return image_tensor, target


class FCDataSet2(Dataset):
    def __init__(self):
        super(FCDataSet2, self).__init__()
        self.zip_files = zipfile.ZipFile('D:/data/object2/dataset.zip')
        self.data_set = []
        files = self.zip_files.namelist()
        for file in files:
            if file.endswith('.jpg'):
                file_name_list = file.split('/')
                target = file_name_list[-2]
                self.data_set.append((file,int(target)))
        print('数据初始化完成')

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_name, target = self.data_set[item]
        target = torch.tensor(target)
        image = self.zip_files.read(file_name)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        image = image/255
        return image, target


if __name__ == '__main__':
    data = FCDataSet2()
