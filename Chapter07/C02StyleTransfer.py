from torchvision.models import vgg16
from torch import nn
from torchvision.utils import save_image
import torch
import cv2


class Block(nn.Module):

    def __init__(self, c, wide):
        super(Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.LayerNorm([c, wide, wide]),
            nn.PReLU(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.LayerNorm([c, wide, wide]),
            nn.PReLU(),
        )

    def forward(self, input_):
        return self.layer(input_)


class UpSample(nn.Module):

    def __init__(self, c, wide):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1, bias=False),
            nn.LayerNorm([c // 2, wide, wide]),
            nn.PReLU(),
        )

    def forward(self, input_):
        return self.layer(input_)


class GNet(nn.Module):

    def __init__(self):
        super(GNet, self).__init__()
        self.layer = nn.Sequential(
            UpSample(1024, 14),
            Block(512, 14),
            Block(512, 14),
            UpSample(512, 28),
            Block(256, 28),
            Block(256, 28),
            UpSample(256, 56),
            Block(128, 56),
            Block(128, 56),
            UpSample(128, 112),
            Block(64, 112),
            Block(64, 112),
            UpSample(64, 224),
            Block(32, 224),
            nn.Conv2d(32,3,3,1,1,bias=False)
        )

    def forward(self):
        initial_value = torch.ones(1, 1024, 7, 7)
        return self.layer(initial_value)

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        a = vgg16(True)
        a = a.features
        self.layer1 = a[:21]
        self.layer2 = a[21:30]

    def forward(self, input_):
        style = self.layer1(input_)
        return style, self.layer2(style)


class Trainer(nn.Module):

    def __init__(self):
        super(Trainer, self).__init__()
        self.g_net = GNet()
        self.d_net = VGG16()
        self.optimizer = torch.optim.Adam(self.g_net.parameters())
        self.loss_func = nn.MSELoss()

    def __call__(self):
        image_c = cv2.imread('c.jpg')
        image_c = cv2.cvtColor(image_c,cv2.COLOR_BGR2RGB)
        image_c = (torch.from_numpy(image_c).permute(2,0,1).float().unsqueeze(0)/255-1)*2
        image_s = cv2.imread('s.jpg')
        image_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
        image_s = (torch.from_numpy(image_s).permute(2,0,1).float().unsqueeze(0) / 255 - 1) * 2
        loss = 1
        i=0
        while loss>0.01:
            image_g = self.g_net()
            _, content_c = self.d_net(image_c)
            style, _ = self.d_net(image_s)
            style_g, content_g = self.d_net(image_g)
            loss_c = self.count_c_loss(content_c, content_g)
            loss_s = self.count_s_loss(style_g, style)
            loss = loss_c+loss_s

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(loss.item())
            if i % 100== 0:
                save_image(image_g, f'{i}.jpg', 1, normalize=True, range=(-1, 1))
            i+=1

    def count_c_loss(self, content_c, content_g):
        return self.loss_func(content_g,content_c)

    def count_s_loss(self,style_g,style):
        style_g = style_g.reshape(512,784)
        style = style.reshape(512,784)
        g_g = torch.mm(style_g,torch.t(style_g))
        g_s = torch.mm(style,torch.t(style))
        return self.loss_func(g_g,g_s)


if __name__ == '__main__':
    trainer = Trainer()
    trainer()
