from torch import nn
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader


class DNet(nn.Module):

    def __init__(self):
        super(DNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self,input_):
        return self.layer(input_)


class GNet(nn.Module):

    def __init__(self):
        super(GNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 784),
        )

    def forward(self,input_):
        return self.layer(input_)


if __name__ == '__main__':
    data_set = MNIST('D:/data',True,transform=ToTensor())
    data_loader = DataLoader(data_set,100,True)
    d_net = DNet().cuda()
    g_net = GNet().cuda()
    loss_func = nn.BCELoss()

    d_optimizer = torch.optim.Adam(d_net.parameters(),lr=0.0002,betas=(0.5,0.999))
    g_optimizer = torch.optim.Adam(g_net.parameters(),lr=0.0002,betas=(0.5,0.999))
    fack_images2 = None
    images = None
    loss = None
    loss3=None
    for epoch in range(1000):
        for i, (images, labels) in enumerate(data_loader):
            images = images.reshape(-1,784).cuda()
            realy_labels = torch.ones(images.shape[0],1).cuda()
            realy_outs = d_net(images)
            loss1 = loss_func(realy_outs,realy_labels)

            fack_images = g_net(torch.randn(images.shape[0],128).cuda())
            fack_labels = torch.zeros(images.shape[0],1).cuda()
            fack_out = d_net(fack_images)
            loss2 = loss_func(fack_out,fack_labels)

            loss = loss1+loss2
            d_optimizer.zero_grad()
            loss.backward()
            d_optimizer.step()

            fack_images2 = g_net(torch.randn(images.shape[0], 128).cuda())
            fack_labels2 = torch.ones(images.shape[0],1).cuda()
            fack_out2 = d_net(fack_images2)
            loss3 = loss_func(fack_out2,fack_labels2)

            g_optimizer.zero_grad()
            loss3.backward()
            g_optimizer.step()
        print(epoch,loss.item(),loss3.item())
        save_image(fack_images2.reshape(-1,1,28,28)[:100],f'D:/data/chapter7/image/F_{epoch}.jpg',nrow=10,normalize=True,scale_each=True)
        save_image(images.reshape(-1,1,28,28)[:100],f'D:/data/chapter7/image/T_{epoch}.jpg',nrow=10,normalize=True,scale_each=True)
        torch.save(g_net.state_dict(), f'D:/data/chapter7/g_net.pth')
        torch.save(d_net.state_dict(), f'D:/data/chapter7/d_net.pth')

