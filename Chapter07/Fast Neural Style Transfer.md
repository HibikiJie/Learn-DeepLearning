# Fast Neural Style Transfer

# 1、简介

在风格迁移中，是以一张图片作为参数来训练它。

生成一张图片则需要数分钟不等的时间。

如果以网络来转换图片，我们训练这个网络，那么则能够快速的将图片进行风格转换，而无需迭代一张图片数百至千次。

![img](D:%5CLearn-DeepLearning%5Cimage%5Cv2-3c7dc2d2a7c408f11d25eff0cd191f7f_720w.jpg)

于是我们所需要做的则是定义这个风格转换网络。



# 2、网络定义

![image-20201023180657640](../image/image-20201023180657640.png)

网络结构如上。

于是构建出网络：

```python
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
        return self.layer(x)
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
image_style = load_image('xinchuan.jpg').cuda()
    net = VGG19().cuda()
    g_net = TransNet().cuda()
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
            print(j, i, loss.item(), loss_c.item(), loss_s.item())
            count += 1
            if i % 100 == 0:
                torch.save(g_net.state_dict(), 'fst.pth')
                save_image([image_g[0], image_c[0]], f'D:/data/chapter7/{i}.jpg', padding=0, normalize=True,
                           range=(0, 1))
        j += 1
```



# 4、结果