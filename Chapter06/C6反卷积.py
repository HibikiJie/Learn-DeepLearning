from torch.utils.data import DataLoader
from torchvision.datasets import MNIST,CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torch import nn
import torch


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d
        )

    def forward(self, input_):
        return self.layer(input_)


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.layer = nn.Sequential(

        )

    def forward(self, input_):
        return self.layer(input_)


class MainNet(nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input_):
        input_ = input_.reshape(-1, 784)
        input_ = self.encoder(input_)
        output = self.decoder(input_)
        return output.reshape(-1, 1, 28, 28)


if __name__ == '__main__':
    net = MainNet().cuda()
    mini_set = CIFAR10('D:\data', True, ToTensor(),download=True)
    print(mini_set[1][0].shape)
    exit()
    data_loader = DataLoader(mini_set, 512, True)
    loss_func = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters())
    i = 0
    k = 0
    images = None
    out = None
    while True:
        loss_sum = 0
        for images, _ in data_loader:
            images = images.cuda()
            out = net(images)

            loss = loss_func(out, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            save_image(images.detach().cpu()[:100], f'D:/data/chapter6/auto_encoder/{i}real_image.jpg', nrow=10)
            save_image(out.detach().cpu()[:100], f'D:/data/chapter6/auto_encoder/{i}generate_image.jpg', nrow=10)
            exit()
        i += 1
        print(f'epoch{i}:{loss_sum / len(data_loader)}')