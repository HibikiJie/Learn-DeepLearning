'''使用pytorch处理自己的数据集'''
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy
import torch
import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
from torch import jit
'''创建自己的数据集'''

"""
Data 模块
"""


class MNISTDataset(Dataset):
    '''只重载  __init__函数
              __len__ 函数
              __getitem__函数'''

    def __init__(self, root, is_train=True):
        '''初始化数据集(将数据集读取进来)'''
        '''一般由于数据过大，只将数据存储的路径读取进来'''

        self.root = root  # 文件的父路径
        self.dataset = []  # 记录所有数据

        '''根据is_train选择加载的数据集'''
        sub_dir = "TRAIN" if is_train else "TEST"
        for tag in os.listdir(f'{root}/{sub_dir}'):
            img_dir = f"{self.root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"

                """封装成数据集,img_path为数据的路径，而后一个为标签"""
                self.dataset.append((img_path, int(tag)))
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
        image_data = cv2.imread(data[0], cv2.IMREAD_GRAYSCALE)

        '''调整数据形状'''
        image_data = image_data.reshape(-1)

        '''数据的归一化'''
        image_data = image_data / 255

        """one-hot编码"""
        target = numpy.zeros(10)
        target[data[1]] = 1

        return torch.tensor(image_data, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class Net_v1(torch.nn.Module):

    def __init__(self):
        super(Net_v1, self).__init__()
        self.w1 = torch.nn.Parameter(torch.randn(784, 1024))
        self.b1 = torch.nn.Parameter(torch.randn(1024))
        self.w2 = torch.nn.Parameter(torch.randn(1024, 512))
        self.b2 = torch.nn.Parameter(torch.randn(512))
        self.w3 = torch.nn.Parameter(torch.randn(512, 10))
        self.b3 = torch.nn.Parameter(torch.randn(10))

    def forward(self, x):
        '''矩阵乘法，即XW+B'''
        x = torch.sigmoid(x @ self.w1 + self.b1)
        x = torch.sigmoid(x @ self.w2 + self.b2)
        x = x @ self.w3 + self.b3

        '''Softmax'''
        h = torch.exp(x)

        '''对每一个批次的求和，故dim=1，同时需要保持维度不变，keepdim传入参数True'''
        z = torch.sum(h, dim=1, keepdim=True)
        return h / z


class Train:

    def __init__(self, root):

        '''实例化SummaryWriter对象'''
        self.summar_writer = SummaryWriter('D:/data/chapter1/logs')

        '''训练集对象'''
        self.train_dataset = MNISTDataset(root)
        self.test_dataset = MNISTDataset(root, False)

        '''实例化数据加载器'''
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True, num_workers=4)
        self.test_dataloader = DataLoader(self.test_dataset, 100, False, num_workers=4)
        '''创建模型'''
        self.net = Net_v1()

        '''创建Adam优化器'''
        self.optim = torch.optim.Adam(self.net.parameters())

    def __call__(self):

        for epoch in range(10000000):
            loss_sum = 0

            for img, taget in tqdm.tqdm(self.train_dataloader):
                out = self.net(img)

                '''定义一个损失函数，此为均方差'''
                loss = torch.mean((taget - out) ** 2)

                '''训练过程'''
                self.optim.zero_grad()  # 梯度归零
                loss.backward()  # 自动求导
                self.optim.step()  # 梯度更新

                loss_sum += loss.detach().item()
            ave_loss = loss_sum / len(self.train_dataloader)

            '''验证'''
            score = 0
            test_loss = 0
            for img, taget in tqdm.tqdm(self.test_dataloader):
                out = self.net(img)
                loss = torch.mean((taget - out) ** 2)
                pre_tage = torch.argmax(out,1)
                taget_tage = torch.argmax(taget,1)
                score += torch.sum(torch.eq(pre_tage,taget_tage).float())
                test_loss += loss.item()
            test_avg_loss = test_loss / len(self.test_dataloader)
            score = score/len(self.test_dataloader)
            self.summar_writer.add_scalars("loss",{'train_loss':ave_loss,'test_loss':test_avg_loss},epoch)
            self.summar_writer.add_scalar("score",score,epoch)
            # print('轮次:', epoch, '训练损失:', ave_loss, '验证损失:', test_avg_loss,'准确度得分:',score)


if __name__ == '__main__':
    # dataset = MNISTDataset('D:/data/MNIST_IMG')
    # print(dataset[1])

    # net = Net_v1()# 实例化网络模型
    # x = torch.randn(3,784)# 假定一个输入数据，该数据有三个批次，每个批次，具有784个维度的数值
    # print(net(x)) # 打印网络的输出
    # print(net(x).shape) # 打印网络模型输出的形状

    # train = Train('D:/data/MNIST_IMG')
    # train()

    model = Net_v1()

    '''虚拟一个输入，占位用的'''
    x = torch.randn(1,784)
    torch_model = jit.trace(model,x)
    torch_model.save("minist_net.pt")
