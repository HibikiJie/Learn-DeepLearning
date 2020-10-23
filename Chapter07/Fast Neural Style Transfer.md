# Fast Neural Style Transfer

# 1、简介

在风格迁移中，是以一张图片作为参数来训练它。

生成一张图片则需要数分钟不等的时间。

如果以网络来转换图片，我们训练这个网络，那么则能够快速的将图片进行风格转换，而无需迭代一张图片数百至千次。

![img](D:%5CLearn-DeepLearning%5Cimage%5Cv2-3c7dc2d2a7c408f11d25eff0cd191f7f_720w.jpg)

于是我们所需要做的则是定义这个风格转换网络。



# 2、网络定义

这里采用UNet来作为风格转换的网络。

![image-20201009141032604](D:%5CLearn-DeepLearning%5Cimage%5Cimage-20201009141032604.png)

网络结构如上。

于是构建出网络：

```python
class CNNLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CNNLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input_):
        return self.layer(input_)


class DownSample(nn.Module):

    def __init__(self, in_channels):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
            nn.ReLU()
        )

    def forward(self, input_):
        return self.layer(input_)


class UpSample(nn.Module):

    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 3, 2, 1, 1),
            nn.ReLU(),
        )

    def forward(self, input_, concat):
        return torch.cat((concat, self.layer(input_)), dim=1)


class UNet(nn.Module):

    def __init__(self):
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
            nn.Conv2d(64, 3, 3, 1, 1),
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
```

数据集使用COCO数据集。



相较于一张[图片迭代](https://blog.csdn.net/weixin_48866452/article/details/109045157)的，风格损失改为带批次的：

```python
def get_gram_matrix(f_map):
    n, c, h, w = f_map.shape
    f_map = f_map.reshape(n, c, h * w)
    gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
    return gram_matrix
```



# 3、训练网络

```python
image_style = load_image('2.jpg').cuda()
net = VGG19().cuda()
g_net = UNet().cuda()
optimizer = torch.optim.Adam(g_net.parameters())
loss_func = nn.MSELoss().cuda()
data_set = COCODataSet()
batch_size = 2
data_loader = DataLoader(data_set, batch_size, True, drop_last=True)
"""计算分格,并计算gram矩阵"""
s1, s2, s3, s4, s5 = net(image_style)
s1 = get_gram_matrix(s1).detach().expand(batch_size, s1.shape[1], s1.shape[1])
s2 = get_gram_matrix(s2).detach().expand(batch_size, s2.shape[1], s2.shape[1])
s3 = get_gram_matrix(s3).detach().expand(batch_size, s3.shape[1], s3.shape[1])
s4 = get_gram_matrix(s4).detach().expand(batch_size, s4.shape[1], s4.shape[1])
s5 = get_gram_matrix(s5).detach().expand(batch_size, s5.shape[1], s5.shape[1])
j = 0
while True:
    for i, image in enumerate(data_loader):
        """生成图片，计算损失"""
        image_c = image.cuda()
        image_g = g_net(image_c)
        out1, out2, out3, out4, out5 = net(image_g)
        # loss = loss_func(image_g, image_c)
        """计算分格损失"""
        loss_s1 = loss_func(get_gram_matrix(out1), s1)
        loss_s2 = loss_func(get_gram_matrix(out2), s2)
        loss_s3 = loss_func(get_gram_matrix(out3), s3)
        loss_s4 = loss_func(get_gram_matrix(out4), s4)
        loss_s5 = loss_func(get_gram_matrix(out5), s5)
        loss_s = loss_s2 + loss_s3 + loss_s4

        """计算内容损失"""
        c1, c2, c3, c4, c5 = net(image_c)

        loss_c1 = loss_func(out1, c1.detach())
        # loss_c2 = loss_func(out2, c2.detach())
        loss_c3 = loss_func(out3, c3.detach())
        # loss_c4 = loss_func(out4, c4.detach())
        # loss_c5 = loss_func(out5, c5.detach())
        loss_c = loss_c3

        """总损失"""
        loss = loss_c + 0.00000001*loss_s

        """清空梯度、计算梯度、更新参数"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(j, i, loss.item(), loss_c.item(), loss_s.item())
        if i % 100 == 0:
            torch.save(g_net.state_dict(), 'fst.pth')
            save_image([image_g[0], image_c[0]], f'D:/data/chapter7/{i}.jpg', padding=0, normalize=True, range=(0, 1))
    j += 1
```



# 4、结果