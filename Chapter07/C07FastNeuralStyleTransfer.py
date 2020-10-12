from torchvision.models import vgg19
from torch import nn
from torchvision.utils import save_image
import torch
import cv2


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        a = vgg19(True)
        a = a.features
        self.layer1 = a[:4]
        self.layer2 = a[4:9]
        self.layer3 = a[9:18]
        self.layer4 = a[18:27]
        self.layer5 = a[27:36]

    def forward(self, input_):
        out1 = self.layer1(input_)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        return out1, out2, out3, out4, out5


class GNet(nn.Module):
    def __init__(self):
        super(GNet, self).__init__()
        # image = torch.log(image / ((1 - image) + a))
        self.image_g = nn.Parameter(image)

    def forward(self):
        # return torch.sigmoid(self.image_g)
        return self.image_g.clamp(0, 1)


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def get_gram_matrix(f_map):
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        raise ValueError('批次应该为1,但是传入的并非为1')


if __name__ == '__main__':
    image_content = load_image('c1.jpg').cuda()
    image_style = load_image('s1.jpg').cuda()
    net = VGG19().cuda()
    g_net = GNet(image_content).cuda()
    optimizer = torch.optim.Adam(g_net.parameters())
    loss_func = nn.MSELoss().cuda()

    """计算内容"""
    c1, c2, c3, c4, c5 = net(image_content)
    c1 = c1.detach().clone()
    c2 = c2.detach().clone()
    c3 = c3.detach().clone()
    c4 = c4.detach().clone()
    c5 = c5.detach().clone()
    """计算风格,并计算gram矩阵"""
    out_s = net(image_style)[2]
    style = get_gram_matrix(out_s).detach().clone()
    i = 0
    while True:
        image = g_net()

        out1, out2, out3, out4, out5 = net(image)
        loss_c1 = loss_func(out1, c1)
        loss_c2 = loss_func(out2, c2)
        loss_c3 = loss_func(out3, c3)
        loss_c4 = loss_func(out4, c4)
        loss_c5 = loss_func(out5, c5)
        loss_c = 0.1*loss_c1 + 0.1*loss_c2 + 0.1*loss_c3 + 0.2*loss_c4 + 0.5*loss_c5
        loss_s = loss_func(get_gram_matrix(out3), style)
        loss = 0.4*loss_c + 0.6*loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(i, loss.item(), loss_c.item(), loss_s.item())
        if i % 100 == 0:
            save_image(image, f'{i}.jpg', padding=0, normalize=True, range=(0, 1))
        i += 1