from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch import nn
import torch
from tqdm import tqdm

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1,bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 64, 3, 2,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 128, 3, 1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, 3, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, 3, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 256, 3, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 512, 3, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        self.features_layer = nn.Sequential(
            nn.Linear(4*4*512, 512, False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU()
        )

    def forward(self, input_):
        output = self.layer(input_)

        output = output.reshape(-1,8192)

        output = self.features_layer(output)
        return output


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(512,8192,bias=False),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(),
        )
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(512,512,3,2,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(512,512,3,2,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 3, 2,output_padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 3, 1,bias=False),

            nn.Conv2d(128, 128, 3, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 64, 3, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 32, 3, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 16, 3, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 3, 3, 1, bias=False),
        )

    def forward(self, input_):
        output = self.layer(input_).reshape(-1,512,4,4)
        output = self.conv_trans(output)
        return torch.min(torch.max(output, torch.tensor([0.]).cuda()), torch.tensor([1.]).cuda())


class MainNet(nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_):
        input_ = self.encoder(input_)
        output = self.decoder(input_)
        return output


if __name__ == '__main__':
    net = MainNet().cuda()
    mini_set = CIFAR10('D:\data', True, ToTensor())

    data_loader = DataLoader(mini_set, 128, True)
    loss_func = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters())
    i = 0
    k = 0
    images = None
    out = None
    while True:
        loss_sum = 0
        for images, _ in tqdm(data_loader):
            images = images.cuda()
            out = net(images)

            loss = loss_func(out, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        save_image(images.detach().cpu()[:16], f'D:/data/chapter6/auto_encoder/{i}real_image.jpg', nrow=4)
        save_image(out.detach().cpu()[:16], f'D:/data/chapter6/auto_encoder/{i}generate_image.jpg', nrow=4)
        i += 1
        print(f'epoch{i}:{loss_sum / len(data_loader)}')
        torch.save(net.decoder.state_dict(),'net.pth')