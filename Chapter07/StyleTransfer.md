# 风格迁移及Pytorch实现

风格迁移，就是利用算法学习一幅画的风格，然后再把这种风格应用到另外一张图片上。

本篇文章会介绍其原理，并使用Pytorch实现。



![image-20201012160519549](C:/Users/lieweiai/AppData/Roaming/Typora/typora-user-images/image-20201012160519549.png)

在卷积中，**浅层特征越具体，深层特征则越抽象）；从风格角度来说，浅层特征则记录着颜色纹理等信息，而深层特征则会记录更高级的信息。**



主要方式则是，随机一张图片，通过优化内容损失和风格损失，改变该图，使其内容接近内容图片，风格上接近风格图片。

内容损失：直接计算特征图的欧式距离；

风格损失：计算特征图的格拉姆矩阵的欧式距离





格拉姆矩阵的计算方式：

```python
def get_gram_matrix(f_map):
    n, c, h, w = f_map.shape
    if n == 1:
        f_map = f_map.reshape(c, h * w)
        gram_matrix = torch.mm(f_map, f_map.t())
        return gram_matrix
    else:
        raise ValueError('批次应该为1,但是传入的不为1')
```

将特征图reshape，将宽高的维度合在一起，然后计算其与自身装置的矩阵乘法即可。



迁移出预先训练好的VGG19的模型。并输出五个不同维度的特征图。

```python
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
```



将图片直接定义为网络参数，来训练它。这里直接从原始内容图训练，也可以使用白噪声。

```python
class GNet(nn.Module):
    def __init__(self, image):
        super(GNet, self).__init__()
        self.image_g = nn.Parameter(image.detach().clone())
        # self.image_g = nn.Parameter(torch.rand(image.shape))  # 也可以初始化一张白噪声训练 

    def forward(self):
        return self.image_g.clamp(0, 1)  # 为了限定数值范围。
```



定义加载图片函数：

```python
def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).float() / 255
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image
```



需要使用图片需要保持形状一致

首先加载**内容图片**和**风格图片**，再实例化**VGG19网络**和**图片**，图片直接从原内容图开始训练。

实例化**优化器**和**损失函数**。

```python
image_content = load_image('c.jpg').cuda()
image_style = load_image('s.jpg').cuda()
net = VGG19().cuda()
g_net = GNet(image_content).cuda()
optimizer = torch.optim.Adam(g_net.parameters())
loss_func = nn.MSELoss().cuda()
```



计算风格图片的输入VGG19的输出，并得到其**格拉姆矩阵**。

```python
s1, s2, s3, s4, s5 = net(image_style)
s1 = get_gram_matrix(s1).detach().clone()
s2 = get_gram_matrix(s2).detach().clone()
s3 = get_gram_matrix(s3).detach().clone()
s4 = get_gram_matrix(s4).detach().clone()
s5 = get_gram_matrix(s5).detach().clone()
```



计算内容图片输入VGG19的输出

```python
c1, c2, c3, c4, c5 = net(image_content)
c1 = c1.detach().clone()
c2 = c2.detach().clone()
c3 = c3.detach().clone()
c4 = c4.detach().clone()
c5 = c5.detach().clone()
```



训练该图片。

```python
i = 0
while True:
    """生成图片，计算损失"""
    image = g_net()
    out1, out2, out3, out4, out5 = net(image)

    """计算分格损失"""
    loss_s1 = loss_func(get_gram_matrix(out1), s1)
    loss_s2 = loss_func(get_gram_matrix(out2), s2)
    loss_s3 = loss_func(get_gram_matrix(out3), s3)
    loss_s4 = loss_func(get_gram_matrix(out4), s4)
    loss_s5 = loss_func(get_gram_matrix(out5), s5)
    loss_s = 0.1*loss_s1 + 0.1*loss_s2 + 0.6*loss_s3 + 0.1*loss_s4 + 0.1*loss_s5

    """计算内容损失"""
    loss_c1 = loss_func(out1, c1)
    loss_c2 = loss_func(out2, c2)
    loss_c3 = loss_func(out3, c3)
    loss_c4 = loss_func(out4, c4)
    loss_c5 = loss_func(out5, c5)
    loss_c = 0.05 * loss_c1 + 0.05 * loss_c2 + 0.15 * loss_c3 + 0.3 * loss_c4 + 0.45 * loss_c5

    """总损失"""
    loss = 0.5*loss_c + 0.5*loss_s

    """清空梯度、计算梯度、更新参数"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(i, loss.item(), loss_c.item(), loss_s.item())
    if i % 1000 == 0:
        save_image(image, f'{i}.jpg', padding=0, normalize=True, range=(0, 1))
    i += 1
```

分别计算风格损失和内容损失，然后求得总损失，优化该损失。



基本迭代一千次即可出效果。

内容图片为：

![c1](c1.jpg)

几个图片的效果展示：

|                           风格图片                           |                           生成图片                           |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="https://img-blog.csdnimg.cn/20201013094015480.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center"  /> | <img src="https://img-blog.csdnimg.cn/20201013094015429.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述"  /> |
|                              /                               | <img src="https://img-blog.csdnimg.cn/20201013094015436.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述"  /> |
|                              /                               | <img src="https://img-blog.csdnimg.cn/20201013094015431.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述"  /> |
|                              /                               | <img src="https://img-blog.csdnimg.cn/20201013094426880.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述"  /> |
| <img src="https://img-blog.csdnimg.cn/20201013094015548.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述"  /> | <img src="https://img-blog.csdnimg.cn/20201013094015363.jpg#pic_center" alt="在这里插入图片描述" style="zoom:200%;" /> |
|                              /                               | <img src="https://img-blog.csdnimg.cn/20201013094015510.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center" alt="在这里插入图片描述" style="zoom:50%;" /> |
| ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015469.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) /> | ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015423.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) |
| ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015492.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) | ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015430.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) |
|                              /                               | ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094921493.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) |
| ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015493.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) | ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015435.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) |
| ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015481.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) | ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015446.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) |
| ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015434.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) | ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015420.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) |
| ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015540.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) | ![在这里插入图片描述](https://img-blog.csdnimg.cn/20201013094015447.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODg2NjQ1Mg==,size_16,color_FFFFFF,t_70#pic_center) |

调整各个损失不同的比例系数，能够达到不同的效果。可酌情尝试。
