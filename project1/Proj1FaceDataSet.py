from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from time import time
import torch


class FaceDataSet(Dataset):

    def __init__(self, path='D:/data/object1/train',image_size='48'):
        super(FaceDataSet, self).__init__()
        self.path = path
        self.size = image_size
        self.data_set = []
        self.to_tensor = ToTensor()
        print('正在初始化数据集...')
        start_time = time()
        with open(f'{self.path}/{self.size}/positive.txt', 'r') as file:
            for line in file:
                line_data = line.split()
                file_name = 'positive/'+line_data[0]
                coordinate = (float(line_data[1]),float(line_data[2]),float(line_data[3]),float(line_data[4]))
                self.data_set.append((file_name,1,coordinate))
        with open(f'{self.path}/{self.size}/part.txt', 'r') as file:
            for line in file:
                line_data = line.split()
                file_name = 'part/'+line_data[0]
                coordinate = (float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4]))
                self.data_set.append((file_name, 2, coordinate))
        with open(f'{self.path}/{self.size}/negative.txt', 'r') as file:
            for line in file:
                line_data = line.split()
                file_name = 'negative/'+line_data[0]
                coordinate = (float(line_data[1]), float(line_data[2]), float(line_data[3]), float(line_data[4]))
                self.data_set.append((file_name, 0, coordinate))
        end_time = time()
        cost_time = end_time-start_time
        print('数据集加载成功，共耗时 %d s'%cost_time)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        image_data = self.data_set[item]
        image_name = image_data[0]
        confidence = image_data[1]
        coordinate = image_data[2]
        image_path = f'{self.path}/{self.size}/{image_name}'
        image = Image.open(image_path)
        image_tensor = self.to_tensor(image)-0.5
        return image_tensor,torch.tensor([confidence],dtype=torch.float32),torch.tensor(coordinate,dtype=torch.float32)


if __name__ == "__main__":
    face_data_set = FaceDataSet(image_size='24')
    image,c,coordinate = face_data_set[100]
    print(face_data_set[100])
    print(image.max())
    print(image.shape)
    print(c.shape)
    print(coordinate.shape)
