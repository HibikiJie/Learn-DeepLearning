# Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from zipfile import ZipFile
import numpy
import cv2
import torch


class DANBOORU2018DataSet(Dataset):
    def __init__(self):
        super(DANBOORU2018DataSet, self).__init__()
        self.zip_files = ZipFile('D:/data/chapter7/seeprettyface_anime_face.zip')
        self.data_set = []
        for file_name in self.zip_files.namelist():
            if file_name.endswith('.jpg'):
                self.data_set.append(file_name)

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        file_path = self.data_set[item]
        image = self.zip_files.read(file_path)
        image = numpy.asarray(bytearray(image), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 128 - 1
        image = image.permute(2, 0, 1)
        return image


class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 5, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, padding=0, bias=False),
            # nn.Sigmoid()去掉，训练更快
        )

    def forward(self, input_):
        return self.layers(input_).reshape(-1,1)


class GNet(nn.Module):

    def __init__(self):
        super(GNet, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128,512,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_):
        return self.layers(input_)


class DCGan(nn.Module):
    def __init__(self):
        super(DCGan, self).__init__()
        self.d_net = DNet()
        self.g_net = GNet()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_):
        return self.g_net(input_)

    def get_DNet_loss(self, noise, real_images):
        predict_real = self.d_net(real_images)
        g_images = self.g_net(noise)
        predict_fake = self.d_net(g_images)

        targets_real = torch.ones(real_images.shape[0], 1).cuda()
        targets_fake = torch.zeros(noise.shape[0], 1).cuda()

        loss1 = self.loss_func(predict_real, targets_real)
        loss2 = self.loss_func(predict_fake, targets_fake)
        return loss1 + loss2

    def get_GNet_loss(self, noise):
        image_g = self.g_net(noise)
        predict_fake = self.d_net(image_g)
        target_fake = torch.ones(noise.shape[0], 1).cuda()
        return self.loss_func(predict_fake, target_fake)


class Trainer:

    def __init__(self):
        self.data_set = DANBOORU2018DataSet()
        self.batch_size = 100
        self.data_loader = DataLoader(self.data_set, self.batch_size, True)
        self.net = DCGan().cuda()
        self.d_optimizer = torch.optim.Adam(self.net.d_net.parameters(), 0.0002,betas=(0.5,0.999))
        self.g_optimizer = torch.optim.Adam(self.net.g_net.parameters(), 0.0002,betas=(0.5,0.999))
        self.net.load_state_dict(torch.load(f'D:/data/chapter7/DCGannet/23.pth'))

    def train(self):
        for epoch in range(100000):
            for i, images in enumerate(self.data_loader):
                images = images.cuda()
                noise_d = torch.normal(0, 0.02, (self.batch_size, 128, 1, 1)).cuda()
                loss_d = self.net.get_DNet_loss(noise_d, images)
                self.d_optimizer.zero_grad()
                loss_d.backward()
                self.d_optimizer.step()

                noise_g = torch.normal(0, 0.02, (self.batch_size, 128, 1, 1)).cuda()
                loss_g = self.net.get_GNet_loss(noise_g)

                self.g_optimizer.zero_grad()
                loss_g.backward()
                self.g_optimizer.step()
                print(epoch, i, loss_d.item(), loss_g.item())
                if i % 10 == 0:
                    noise = torch.normal(0, 0.02, (self.batch_size, 128, 1, 1)).cuda()
                    image = self.net.forward(noise)
                    save_image(image, f'D:/data/chapter7/Gimage/{epoch}v2.jpg', 10, normalize=True, range=(-1, 1))
                    torch.save(self.net.state_dict(), f'D:/data/chapter7/DCGannet/{epoch}v2.pth')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
