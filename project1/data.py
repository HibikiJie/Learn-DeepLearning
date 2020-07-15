from torch.utils.data import Dataset
import os
import numpy
import cv2
import torch
import matplotlib.pyplot as plt

class SunDataset(Dataset):

    def __init__(self,root="D:/sun/",is_train=True):
        self.dataset = []
        img_dir = "train" if is_train else "test"
        file1 = open("D:/sun/train.txt")
        file2 = open('D:/sun/train_output.txt')
        para_in = file1.read().split()
        target = file2.read().split()
        a = len(target)//2
        for i in range(a):
            image_path = "C:/sun/train/"+target[2*i]
            self.dataset.append((image_path,para_in[11*i+1:11*i+11],target[2*i+1]))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img_data = cv2.imread(data[0],cv2.IMREAD_GRAYSCALE)
        img_data = img_data.reshape(224,224,1)
        img_data = img_data/255
        target = int(data[2])
        para = data[1]
        para = numpy.array(para,numpy.float64)
        img_data = torch.from_numpy(img_data).permute(2,0,1)
        para = torch.from_numpy(para)
        target = torch.tensor([target])

        return img_data,para,target

if __name__ == '__main__':
    pass
    sun_dataset = SunDataset()
    print(sun_dataset[1])
