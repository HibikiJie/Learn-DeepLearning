from torch import nn
import torch
import random
import os
import numpy
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class DriveDataset(Dataset):

    def __init__(self,root='D:/data/chapter8/DRIVE/training',is_train=True):
        super(DriveDataset, self).__init__()
        self.dataset = []
        if is_train:
            start = 20
        else:
            start = 0
            root = 'D:/data/chapter8/DRIVE/test'
        for i in range(1, 21):
            path1 = f'{root}/images/{i+start}_training.tif'
            path2 = f'{root}/1st_manual/{i + start}_manual1.gif'
            self.dataset.append((path1, path2))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        path1, path2 = self.dataset[item]
        image1 = cv2.imread(path1)
        video = cv2.VideoCapture(path2)
        _, image2 = video.read()
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        h, w = image2.shape
        w = random.randint(0, w-256)
        h = random.randint(0, h-256)
        image1 = image1[h:h+256, w:w+256]
        image2 = image2[h:h + 256, w:w + 256]
        image1 = torch.from_numpy(image1).float().permute(2, 0, 1)/255
        image2 = torch.from_numpy(image2).unsqueeze(0).float()/255
        return image1,image2


class CNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        卷积层
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        """
        super(CNNLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, input_):
        return self.layer(input_)


class DownSample(nn.Module):

    def __init__(self, in_channels):
        """
        步长为2的卷积，下采样
        :param in_channels: 输入通道数
        """
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, input_):
        return self.layer(input_)


class UpSample(nn.Module):

    def __init__(self, in_channels):
        """
        反卷积，上采样
        :param in_channels: 输入通道数
        """
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 3, 2, 1, 1),
            nn.LeakyReLU(),
        )

    def forward(self, input_, concat):
        """
        前向运算过程
        :param input_: 输入
        :param concat: 浅层特征
        :return: 完成上采样后和浅层特征concat的结果
        """
        return torch.cat((concat, self.layer(input_)), dim=1)


class UNet(nn.Module):

    def __init__(self):
        """
        UNet网络，结构类似编解码，
        """
        super(UNet, self).__init__()
        self.c1 = CNNLayer(3, 64)
        self.d1 = DownSample(64)
        self.c2 = CNNLayer(64, 128)
        self.d2 = DownSample(128)
        self.c3 = CNNLayer(128, 256)
        self.d3 = DownSample(256)
        self.c4 = CNNLayer(256, 512)
        self.d4 = DownSample(512)
        self.c5 = CNNLayer(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = CNNLayer(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = CNNLayer(512, 256)
        self.u3 = UpSample(256)
        self.c8 = CNNLayer(256, 128)
        self.u4 = UpSample(128)
        self.c9 = CNNLayer(128, 64)
        self.predict1 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, image_tensor):
        c1 = self.c1(image_tensor)
        c2 = self.c2(self.d1(c1))
        c3 = self.c3(self.d2(c2))
        c4 = self.c4(self.d3(c3))
        c5 = self.c5(self.d4(c4))
        c6 = self.c6(self.u1(c5, c4))
        c7 = self.c7(self.u2(c6, c3))
        c8 = self.c8(self.u3(c7, c2))
        c9 = self.c9(self.u4(c8, c1))
        return self.predict1(c9)


class Trainer:

    def __init__(self):
        self.net = UNet().cuda()
        self.net.load_state_dict(torch.load('D:/data/chapter8/unet.pth'))
        self.dataset = DriveDataset()
        self.data_loader = DataLoader(self.dataset, 3,True,drop_last=True)
        self.loss_func = nn.MSELoss().cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters())

    def train(self):
        image = None
        target = None
        out = None
        for epoch in range(100000):
            for i,(image, target) in enumerate(self.data_loader):
                image = image.cuda()
                target = target.cuda()

                out = self.net(image)
                loss = self.loss_func(out,target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(epoch,loss.item())
            if epoch%100==0:
                torch.save(self.net.state_dict(),'D:/data/chapter8/unet.pth')
                save_image([image[0], target[0].expand(3,256,256), out[0].expand(3,256,256)],f'D:/data/chapter8/{epoch}.jpg',normalize=True,range=(0,1))


class Explorer:

    def __init__(self):
        self.net = UNet()
        self.net.load_state_dict(torch.load('D:/data/chapter8/unet.pth'))

    def explore(self, image):
        image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255
        out = self.net(image).squeeze(0).permute(1,2,0)*255
        out = out.detach().numpy()
        return out


if __name__ == '__main__':
    # trainer = Trainer()
    # trainer.train()

    explorer = Explorer()
    image = cv2.imread('D:/data/chapter8/DRIVE/test/images/01_test.tif')
    h,w,c = image.shape
    print(image.shape)
    image = cv2.resize(image, (512, 512),interpolation=cv2.INTER_AREA)
    image = explorer.explore(image)
    image = cv2.resize(image, (w, h),interpolation=cv2.INTER_AREA)
    video = cv2.VideoCapture('D:/data/chapter8/DRIVE/test/mask/01_test_mask.gif')
    _,mask = video.read()
    image = numpy.where(mask[:, :, 0] > 125, image, 0)
    print(image.shape)
    cv2.imshow('JK', image.astype(numpy.uint8))
    cv2.waitKey()
