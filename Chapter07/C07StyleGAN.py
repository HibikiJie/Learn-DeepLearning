# Thanks to dataset provider:Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the dataset.
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from Chapter07.C07DNet import DNet
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
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = 2*(torch.from_numpy(image).float() / 255 - 0.5)
        image = image.permute(2, 0, 1)
        return image


class MappingNetwork(nn.Module):

    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.PReLU(),
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.PReLU(),
        )

    def forward(self, input_):
        return self.layer(input_)


class AdaIN(nn.Module):

    def __init__(self, c, size):
        super(AdaIN, self).__init__()
        self.c = c
        self.size = size
        self.normalize_channel = nn.LayerNorm([size, size], elementwise_affine=False)
        self.transformation = nn.Sequential(
            nn.Linear(128, 2 * c, bias=False),
            nn.LayerNorm(2 * c),
            nn.PReLU(),
        )

    def forward(self, input_, w_):
        wide = input_.shape[-1]
        w_ = self.transformation(w_).reshape(-1, 2, self.c)
        input_ = self.normalize_channel(input_)
        w = w_[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, self.c, wide, wide)
        b = w_[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, self.c, wide, wide)
        return input_ * w + b


class StohasticVariation(nn.Module):

    def __init__(self, c):
        super(StohasticVariation, self).__init__()
        self.c = c
        self.learned_per_channel_scale = nn.Conv2d(c, c, 1, 1, groups=c, bias=False)

    def forward(self, input_):
        wide = input_.shape[-1]
        batch = input_.shape[0]
        noise = torch.normal(0, 0.02, (batch, self.c, 1, 1)).cuda().expand(batch, self.c, wide, wide)
        return self.learned_per_channel_scale(noise) + input_


class SyntheisBlock(nn.Module):
    def __init__(self, channels, size):
        super(SyntheisBlock, self).__init__()
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1, bias=False),
            nn.LayerNorm([channels, size, size]),
            nn.PReLU(),
        )
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, padding=1,bias=False),
            nn.LayerNorm([channels, size, size]),
            nn.PReLU(),
            StohasticVariation(channels)
        )
        self.adain_1 = AdaIN(channels, size)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, padding=1,bias=False),
            nn.LayerNorm([channels, size, size]),
            nn.PReLU(),
            StohasticVariation(channels)
        )
        self.adain_2 = AdaIN(channels, size)

    def forward(self, input_, w_):
        out = self.up_sample(input_)
        out = self.conv_layer1(out)
        out = self.adain_1(out, w_)
        out = self.conv_layer2(out)
        return self.adain_2(out, w_)


class SyntheisBlock4X4(nn.Module):
    def __init__(self, channels, size):
        super(SyntheisBlock4X4, self).__init__()
        self.conv_layer1 = nn.Sequential(
            StohasticVariation(channels)
        )
        self.adain_1 = AdaIN(channels, size)
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, padding=1,bias=False),
            nn.LayerNorm([channels, size, size]),
            nn.PReLU(),
            StohasticVariation(channels)
        )
        self.adain_2 = AdaIN(channels, size)

    def forward(self, input_, w_):
        out = self.conv_layer1(input_)
        out = self.adain_1(out, w_)
        out = self.conv_layer2(out)
        return self.adain_2(out, w_)


class GNet(nn.Module):

    def __init__(self):
        super(GNet, self).__init__()
        self.mapping_net = MappingNetwork()
        self.block1 = SyntheisBlock4X4(512,4)
        self.block2 = SyntheisBlock(256,8)
        self.block3 = SyntheisBlock(128, 16)
        self.block4 = SyntheisBlock(64, 32)
        self.block5 = SyntheisBlock(32, 64)
        self.block6 = SyntheisBlock(16, 128)
        self.linear = nn.Sequential(
            nn.Conv2d(16, 3, 3, 1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input_):
        batch_size = input_.shape[0]
        mid_w = self.mapping_net(input_)
        features_map = torch.ones(batch_size, 512, 4, 4).cuda()
        out = self.block1(features_map,mid_w)
        out = self.block2(out, mid_w)
        out = self.block3(out, mid_w)
        out = self.block4(out, mid_w)
        out = self.block5(out, mid_w)
        out = self.block6(out, mid_w)
        return self.linear(out)


class StyleGan(nn.Module):
    def __init__(self):
        super(StyleGan, self).__init__()
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
        self.batch_size = 32
        self.data_loader = DataLoader(self.data_set, self.batch_size, True)
        self.net = StyleGan().cuda()
        self.d_optimizer = torch.optim.Adam(self.net.d_net.parameters(), 0.00003, betas=(0.5, 0.9))
        self.g_optimizer = torch.optim.Adam(self.net.g_net.parameters(), 0.00003, betas=(0.5, 0.9))
        self.net.load_state_dict(torch.load(f'D:/data/chapter7/DCGannet/2v2.pth'))

    def train(self):
        for epoch in range(100000):
            for i, images in enumerate(self.data_loader):
                images = images.cuda()

                noise_d = torch.normal(0, 0.02, (self.batch_size, 128)).cuda()
                loss_d = self.net.get_DNet_loss(noise_d, images)
                self.d_optimizer.zero_grad()
                loss_d.backward()
                self.d_optimizer.step()

                noise_g = torch.normal(0, 0.02, (self.batch_size, 128)).cuda()
                loss_g = self.net.get_GNet_loss(noise_g)

                self.g_optimizer.zero_grad()
                loss_g.backward()
                self.g_optimizer.step()
                print(epoch, i, loss_d.item(), loss_g.item())
                if i % 10 == 0:
                    noise = torch.normal(0, 0.02, (self.batch_size, 128)).cuda()
                    image = self.net.forward(noise)
                    save_image(image[:25], f'D:/data/chapter7/Gimage/{epoch}v2.jpg', 5, normalize=True, range=(-1, 1))
                    torch.save(self.net.state_dict(), f'D:/data/chapter7/DCGannet/{epoch}v2.pth')


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
