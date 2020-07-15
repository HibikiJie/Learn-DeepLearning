'''使用pytorch处理自己的数据集'''
from torch.utils.data import Dataset
import os
import cv2
import numpy
'''创建自己的数据集'''

"""
Data 模块
"""
class MNISTDataset(Dataset):
    '''只重载  __init__函数
              __len__ 函数
              __getitem__函数'''

    def __init__(self,root,is_train=True):
        '''初始化数据集(将数据集读取进来)'''
        '''一般由于数据过大，只将数据存储的路径读取进来'''

        self.root = root#文件的父路径
        self.dataset = []#记录所有数据

        '''根据is_train选择加载的数据集'''
        sub_dir = "TRAIN" if is_train else "TEST"
        for tag in os.listdir(f'{root}/{sub_dir}'):
            img_dir = f"{self.root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"

                """封装成数据集,img_path为数据的路径，而后一个为标签"""
                self.dataset.append((img_path,int(tag)))
        pass

    def __len__(self):
        '''统计数据的个数'''

        pass
        '''返回数据集的长度'''
        return len(self.dataset)

    def __getitem__(self, index):
        '''每条数据的处理方式'''

        pass

        data = self.dataset[index]
        image_data = cv2.imread(data[0],cv2.IMREAD_GRAYSCALE)

        '''调整数据形状'''
        image_data = image_data.reshape(-1)

        '''数据的归一化'''
        image_data = image_data/255

        """one-hot编码"""
        target = numpy.zeros(10)
        target[data[1]] = 1

        return image_data,target


if __name__ == '__main__':
    dataset = MNISTDataset('D:/data/MNIST_IMG')
    print(dataset[1])
