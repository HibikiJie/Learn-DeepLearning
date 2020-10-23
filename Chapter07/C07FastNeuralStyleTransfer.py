from torchvision.models import vgg19
from torch import nn
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
import torch
import cv2
import numpy


class COCODataSet(Dataset):

    def __init__(self):
        super(COCODataSet, self).__init__()
        self.zip_files = ZipFile('D:/data/chapter7/train2014.zip')
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
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).float() / 255
        image = image.permute(2, 0, 1)
        return image


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


class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, input_):
        return torch.sin(input_)


class Swish(nn.Module):

    def forward(self, x):
        return x * x.sigmoid()


class ResBlock(nn.Module):

    def __init__(self,c):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c,c,3,1,1, bias=False),
            nn.BatchNorm2d(c),
            Swish(),
            nn.Conv2d(c, c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),

        )
        self.swish = Swish()

    def forward(self, x):
        return self.swish(self.layer(x)+x)

class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, 9, 1, 4, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.Conv2d(32,64,3,2,1, bias=False),
            nn.InstanceNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            Swish(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128,64,3,1,1, bias=False),
            nn.InstanceNorm2d(64),
            Swish(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            Swish(),
            nn.Conv2d(32,3,9,1,4, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = nn.functional.pad(x, [10, 10, 10, 10])
        # return self.layer(x)[:,:,10:-10,10:-10]
        return self.layer(x)


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def get_gram_matrix(f_map):
    """
    获取格拉姆矩阵
    :param f_map:特征图
    :return:格拉姆矩阵，形状（通道数,通道数）
    """
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix.div(c*h*w)
    else:
        raise ValueError('批次应该为1,但是传入的不为1')


if __name__ == '__main__':
    image_style = load_image('2.jpg').cuda()
    net = VGG19().cuda()
    g_net = TransNet().cuda()
    # summary_writer = SummaryWriter('D:/data/chapter7/logs')
    # g_net.load_state_dict(torch.load('fst.pth'))
    optimizer = torch.optim.Adam(g_net.parameters())
    loss_func = nn.MSELoss().cuda()
    data_set = COCODataSet()
    batch_size = 1
    data_loader = DataLoader(data_set, batch_size, True, drop_last=True)
    """计算分格,并计算gram矩阵"""
    s1, s2, s3, s4, s5 = net(image_style)
    s1 = get_gram_matrix(s1).detach()
    s2 = get_gram_matrix(s2).detach()
    s3 = get_gram_matrix(s3).detach()
    s4 = get_gram_matrix(s4).detach()
    s5 = get_gram_matrix(s5).detach()
    j = 0
    count = 0
    while True:
        for i, image in enumerate(data_loader):
            """生成图片，计算损失"""
            image_c = image.cuda()
            image_g = g_net(image_c)
            out1, out2, out3, out4, out5 = net(image_g)
            # loss = loss_func(image_g, image_c)
            """计算风格损失"""
            loss_s1 = loss_func(get_gram_matrix(out1), s1)
            loss_s2 = loss_func(get_gram_matrix(out2), s2)
            loss_s3 = loss_func(get_gram_matrix(out3), s3)
            loss_s4 = loss_func(get_gram_matrix(out4), s4)
            loss_s5 = loss_func(get_gram_matrix(out5), s5)
            loss_s = loss_s1+loss_s2+loss_s3+loss_s4+loss_s5

            """计算内容损失"""
            c1, c2, c3, c4, c5 = net(image_c)

            # loss_c1 = loss_func(out1, c1.detach())
            # loss_c2 = loss_func(out2, c2.detach())
            loss_c3 = loss_func(out3, c3.detach())
            # loss_c4 = loss_func(out4, c4.detach())
            # loss_c5 = loss_func(out5, c5.detach())
            loss_c = loss_c3

            """总损失"""
            loss = loss_c + 22000000 * loss_s

            """清空梯度、计算梯度、更新参数"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(j, i, loss.item())
            print(j, i, loss.item(), loss_c.item(), loss_s.item())
            count += 1
            if i % 100 == 0:
                # print(j,i, loss.item(), loss_c.item(), loss_s.item())
                torch.save(g_net.state_dict(), 'fst.pth')
                save_image([image_g[0], image_c[0]], f'D:/data/chapter7/{i}.jpg', padding=0, normalize=True,
                           range=(0, 1))
        j += 1
