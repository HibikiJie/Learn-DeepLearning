# Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from zipfile import ZipFile
import numpy
import cv2
import torch


class DNet(nn.Module):

    def __init__(self):
        super(DNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(512, 512, 2, 2),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(512, 512, 3, padding=1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(512, 512, 2, 2),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(8192, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input_):
        n, c, h, w = input_.shape
        input_ = self.layer1(input_).reshape(n, -1)
        return self.layer2(input_)
        # return self.layer1(input_)


class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_):
        temp1 = input_ ** 2
        temp2 = torch.rsqrt(torch.mean(temp1, dim=1, keepdim=True) + self.epsilon)
        return input_ * temp2


class GMapping(nn.Module):

    def __init__(self,
                 num_features=512,
                 ):
        super(GMapping, self).__init__()
        self.layer = nn.Sequential(
            PixelNorm(),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, input_):
        return self.layer(input_)


class ApplyNoise(nn.Module):
    def __init__(self, num_channels, size):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_channels))
        self.size = size

    def forward(self, x):
        noise = torch.randn(x.shape[0], 1, self.size, self.size)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class ApplyStyle(nn.Module):
    def __init__(self, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(512, channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class AdaIN(nn.Module):
    def __init__(self, num_channels, size):
        super(AdaIN, self).__init__()
        self.noise = ApplyNoise(num_channels, size)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.instance_norm = nn.InstanceNorm2d(num_channels)
        self.apply_style = ApplyStyle(num_channels)

    def forward(self, input_, latent):
        input_ = self.noise(input_)
        input_ = self.act(input_)
        input_ = self.instance_norm(input_)
        return self.apply_style(input_, latent)


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, size):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.a1 = AdaIN(out_channels, size)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.a2 = AdaIN(out_channels, size)

    def forward(self, input_, latent):
        input_ = self.layer(input_)
        input_ = self.a1(input_, latent)
        input_ = self.conv1(input_)
        return self.a2(input_, latent)


class SynthesisNetWork(nn.Module):
    def __init__(self):
        super(SynthesisNetWork, self).__init__()
        self.const_input = nn.Parameter(torch.ones(1, 512, 4, 4))
        self.bias = nn.Parameter(torch.ones(512))
        self.adain1 = AdaIN(512, 4)
        self.adain2 = AdaIN(512, 4)
        self.conv1 = nn.Conv2d(512, 512, 3, padding=1)

        self.block1 = Block(512, 512, 8)
        self.block2 = Block(512, 512, 16)
        # self.block3 = Block(512, 512, 32)
        # self.block4 = Block(512, 512, 64)
        self.block5 = Block(512, 256, 32)
        self.block6 = Block(256, 128, 64)
        self.block7 = Block(128, 64, 128)
        self.block8 = Block(64, 32, 256)
        self.layer = nn.Sequential(
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.Conv2d(8, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, latent):
        n, v = latent.shape
        x = self.const_input.expand(n, -1, -1, -1)
        x = x + self.bias.reshape(1, -1, 1, 1)
        x = self.adain1(x, latent)
        x = self.conv1(x)
        x = self.adain2(x, latent)
        x = self.block1(x, latent)
        x = self.block2(x, latent)
        # x = self.block3(x, latent)
        # x = self.block4(x, latent)
        x = self.block5(x, latent)
        x = self.block6(x, latent)
        x = self.block7(x, latent)
        x = self.block8(x, latent)

        return self.layer(x)


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        self.g_mapping = GMapping()
        self.synthesis_network = SynthesisNetWork()

    def forward(self, latent):
        latent = self.g_mapping(latent)
        return self.synthesis_network(latent)


class DANBOORU2018DataSet(Dataset):
    def __init__(self):
        super(DANBOORU2018DataSet, self).__init__()
        self.zip_files = ZipFile('D:/data/chapter7/seeprettyface_chs_wanghong.zip')
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
        h, w, c = image.shape
        if not (h == w and h == 256):
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 128 - 1
        image = image.permute(2, 0, 1)
        return image


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
        self.batch_size = 8
        self.data_loader = DataLoader(self.data_set, self.batch_size, True)
        self.net = DCGan().cuda()
        self.d_optimizer = torch.optim.Adam(self.net.d_net.parameters(), lr=0.00001, betas=(0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(self.net.g_net.parameters(), lr=0.00001, betas=(0.5, 0.999))
        # self.net.load_state_dict(torch.load(f'D:/data/chapter7/DCGannet/3v2.pth'))

    def train(self):
        for epoch in range(100000):
            for i, images in enumerate(self.data_loader):
                images = images.cuda()
                noise_d = torch.randn(self.batch_size, 512).cuda()
                loss_d = self.net.get_DNet_loss(noise_d, images)
                self.d_optimizer.zero_grad()
                loss_d.backward()
                self.d_optimizer.step()

                noise_g = torch.randn(self.batch_size, 512).cuda()
                loss_g = self.net.get_GNet_loss(noise_g)

                self.g_optimizer.zero_grad()
                loss_g.backward()
                self.g_optimizer.step()
                print(epoch, i, loss_d.item(), loss_g.item())
                if i % 100 == 0:
                    noise = torch.randn(1, 512).cuda()
                    image = self.net.forward(noise)
                    save_image(image, f'D:/data/chapter7/Gimage/{epoch}v2.jpg', 1, normalize=True, range=(-1, 1))
                    torch.save(self.net.state_dict(), f'D:/data/chapter7/DCGannet/{epoch}s.pth')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
