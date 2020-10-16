from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from project3.P3Set import Set
import numpy as np
from math import log


class Voc2012DataSet(Dataset):

    def __init__(self, image_path='D:/data/object3/dataset', path='D:/data/object3/train.txt'):
        super(Voc2012DataSet, self).__init__()
        print('正在初始化数据集')
        self.path = path
        self.set = Set()
        self.image_path = image_path
        self.dataset = []
        with open(self.path) as file:
            for line in file.readlines():
                line = line.split()
                image_name = line[0]
                path = f'{self.image_path}/{image_name}'
                image_information = []
                boxes = line[1:]
                for i in range(len(boxes) // 5):
                    box = boxes[5 * i:5 * i + 5]
                    target = int(box[0])
                    x1 = int(box[1])
                    y1 = int(box[2])
                    x2 = int(box[3])
                    y2 = int(box[4])
                    image_information.append((target, x1, y1, x2, y2))
                self.dataset.append([path, image_information])
        print('数据集初始化完成')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, boxes = self.dataset[item]
        image = cv2.imread(image_path)
        image_tensor = torch.from_numpy(image).float() / 255
        image_tensor = image_tensor.permute(2, 0, 1)
        targets_13 = np.zeros((13, 13, 3, 6), dtype=np.float32)
        targets_26 = np.zeros((26, 26, 3, 6), dtype=np.float32)
        targets_52 = np.zeros((52, 52, 3, 6), dtype=np.float32)

        '''循环每一个标签框，放入'''
        for box in boxes:
            target = box[0]
            x1, y1, x2, y2 = box[1:]
            w = x2 - x1
            h = y2 - y1
            c_x = x1 + w / 2
            c_y = y1 + h / 2
            i = 0
            iou = 0
            trunk = []

            '''循环测试该目标框和哪一个目标更加的匹配，iou的值最大'''
            for size in self.set.boxes_base:
                stride = 416 // size
                index_h = c_y // stride
                index_w = c_x // stride
                offset_x = (c_x % stride) / stride
                offset_y = (c_y % stride) / stride
                for box2 in self.set.boxes_base[size]:
                    ratio_w = w / box2[0]
                    ratio_h = h / box2[1]
                    if i == 0:
                        trunk = [int(index_h), int(index_w), 1., offset_x, offset_y, log(ratio_w), log(ratio_h), target, 0, 0]
                        iou = self.calculate_iou((w, h), box2)
                    else:
                        next_iou = self.calculate_iou((w, h), box2)
                        if next_iou > iou:
                            iou = next_iou
                            trunk = [int(index_h), int(index_w), 1., offset_x, offset_y, log(ratio_w), log(ratio_h), target, i // 3,
                                     i % 3]
                    i += 1

            '''写入标签中'''
            # print(w,h)
            if trunk[8] == 0:
                targets_52[trunk[0], trunk[1], trunk[-1]] = torch.tensor(trunk[2:8])
            elif trunk[8] == 1:
                targets_26[trunk[0], trunk[1], trunk[-1]] = torch.tensor(trunk[2:8])
            elif trunk[8] == 2:
                targets_13[trunk[0], trunk[1], trunk[-1]] = torch.tensor(trunk[2:8])
        return image_tensor, targets_13, targets_26, targets_52

    @staticmethod
    def calculate_iou(box1, box2):
        min_w = min(box1[0], box2[0])
        min_h = min(box1[1], box2[1])
        intersection = min_w * min_h
        area1 = box1[0] * box2[0]
        area2 = box1[1] * box2[1]
        return intersection / (area1 + area2 - intersection)


if __name__ == '__main__':
    voc = Voc2012DataSet()
    dataload = DataLoader(voc,1,False)
    for i in dataload:
        pass
