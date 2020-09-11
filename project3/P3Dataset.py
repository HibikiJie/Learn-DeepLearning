from torch.utils.data import Dataset
import cv2
import torch
from project3.P3Set import Set
import numpy as np


class Voc2012DataSet(Dataset):

    def __init__(self, out_channels, image_path='D:/data/object3/dataset', path='D:/data/object3/train.txt'):
        super(Voc2012DataSet, self).__init__()
        self.path = path
        self.set = Set()
        self.image_path = image_path
        self.out_channels = out_channels
        self.dataset = []
        with open(self.path) as file:
            for line in file.readlines():
                line = line.split()
                image_name = line[0]
                path = f'{self.image_path}/{image_name}'
                image_information = []
                boxes = line[1:]
                for i in range(len(boxes)//5):
                    box = boxes[5*i:5*i+5]
                    target = int(box[0])
                    x1 = int(box[1])
                    y1 = int(box[2])
                    x2 = int(box[3])
                    y2 = int(box[4])
                    image_information.append((target,x1,y1,x2,y2))
                self.dataset.append([path,image_information])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, boxes = self.dataset[item]
        image = cv2.imread(image_path)
        image_tensor = torch.from_numpy(image).float()/255
        image_tensor = image_tensor.permute(2, 0, 1)
        target_13 = np.zeros((13, 13, 3, 5), dtype=np.float32)
        target_26 = np.zeros((26, 26, 3, 5), dtype=np.float32)
        target_52 = np.zeros((52, 52, 3, 5), dtype=np.float32)
        for box in boxes:


        return image_tensor,target_13,target_26,target_52


if __name__ == '__main__':
    voc = Voc2012DataSet()
    voc[1]
    torch.nn.NLLLoss
