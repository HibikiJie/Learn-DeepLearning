# Chapter02

[TOC]

# 二、卷积神经网络

![img](D:\Learn-DeepLearning\image\juanji.jpg)

上一章，全连接网络的结构如上图所示。

然而全连接神经网络是有，缺陷的。如果使用全连接来识别一章图片时，首层的神经元个数达到了wXh的个数，假设图片为200X200，首层的神经元个数便达到了，四万个。这极大的增加了内存占用，和计算的次数。因此，在处理图像时，计算速度慢，效果差。

## 1、卷积过程

![img](https://ss2.bdstatic.com/70cFvnSh_Q1YnxGkpoWK1HF6hhy/it/u=2448747360,1280135558&fm=15&gp=0.jpg)

卷积过程是，卷积核在图片上扫描，取出和核大小一样的一块区域，相乘再求和。然后又向左移动一步。一排卷积完成，再向下走一步。直到卷积完成所有的区域。

卷积过程见下动图所示：

![GIF 2020-7-24 10-56-35](D:\Learn-DeepLearning\image\GIF 2020-7-24 10-56-35.gif)

卷积完成后的得到的图片大小为：
$$
L=\left \lfloor \frac{l+2\times Padding-(kernelsize-1)-1}{stride}+1 \right \rfloor
$$
![image-20200717151831115](D:\Learn-DeepLearning\image\image-20200717151831115.png)

一层卷积如上图所以，一张三通道的图片，通过卷积核卷积后生成一张特征图。一共有六个卷积核，共生成了六张特征图，这六张特征图，又输入给下一层。

这类似于全连接神经网络，因为输入要传给所有的卷积核，一个卷积核只输出一张特征图。如此将单个神经元升级为一个卷积核，将输出一个数字升级为输出一张图，也就是单个神经元的计算复杂度提升了。来达到减少参数量，相应的计算力需求更大，特别是并行计算。



## 2、卷积尺寸与计算量

$$
L_{out}=\left \lfloor \frac{L_{in}+2\times padding-dilation\times (kernel_{size}-1)-1}{sride}+1 \right \rfloor
$$

卷积尺寸的计算为公式（2）

其中$L_{out}$为输出特征图的的尺寸大小；

$L_{in}$为输入特征图的尺寸打下；

$padding$为在此次卷积过程中，在边沿部分添加零的层数；

$dilation$为卷积过程中，扩张率；

$kernel_{size}$为卷积核的大小；

$sride$为卷积过程的步长



浮点计算量，指计算量，跟乘加次数有点不一样，若考虑偏置，则：
$$
FLOP_{S}=(C_{in}*2*K^{2})*H_{out}*W_{out}*C_{out}
$$
不考虑偏置的情况下
$$
FLOP_{S}=(C_{in}*2*K^{2}-1)*H_{out}*W_{out}*C_{out}
$$


## 3、池化

![img](D:\Learn-DeepLearning\image\991470-20190208201508704-368644792.png)

池化（Pooling）是卷积神经网络中另一个重要的概念，它实际上是**一种形式的降采样**。有多种不同形式的非线性池化函数，而其中“最大池化（Max pooling）”是最为常见的。它是将输入的图像划分为若干个矩形区域，对每个子区域输出最大值。直觉上，这种**机制能够有效地原因在于，在发现一个特征之后，它的精确位置远不及它和其他特征的相对位置的关系重要**。**池化层会不断地减小数据的空间大小，因此参数的数量和计算量也会下降，这在一定程度上也控制了过拟合**。通常来说，CNN的卷积层之间都会周期性地插入池化层。

池化层通常会**分别作用于每个输入的特征并减小其大小**。当前最常用形式的池化层是每隔2个元素从图像划分出2*2的区块，然后对每个区块中的4个数取最大值。这将会减少75%的数据量。![image-20200724113209533](D:\Learn-DeepLearning\image\image-20200724113209533.png)

池化操作后的结果相比其输入缩小了，池化层的引入是仿照人的视觉系统对视觉输入对象进行降维和抽象。池化一般认为有以下三个功效：

1. **特征不变形**：池化操作使模型更加关注是否存在某些特征而不是特征具体的位置。
2. **特征降维**：池化相当于在空间范围内做了维度约减，从而使模型可以抽取更加广泛的特征。同时减小了下一层的输入大小，进而减少计算量和参数个数。
3. **在一定程度上放置过拟合**，更方便优化

## 4、自适应池化

![image-20200724113921103](D:\Learn-DeepLearning\image\image-20200724113921103.png)

确定输出的数量，根据输入图片的尺寸划分区域，例如，输出为3X3的尺寸，则将图片划分为3X3的不同区域。即根据输入图片尺寸，确定池化窗口的大小。



## 5、空洞卷积

![image-20200724114425828](D:\Learn-DeepLearning\image\image-20200724114425828.png)

Dilated Convolutions，翻译为扩张卷积或空洞卷积。空洞卷积与普通的卷积相比，除了卷积核的大小以外，还有一个扩张率(dilation rate)参数，主要用来表示扩张的大小。默认值为1，表示不扩张。扩张卷积与普通卷积的相同点在于，卷积核的大小是一样的，在神经网络中即参数数量不变，区别在于扩张卷积具有更大的感受野。相应的，对细节的感受会降低，**或许只对一些大物体分割有效果，而对小物体来说可能则有弊无利了**。



## 6、感受野

在卷积神经网络中，**感受野**（Receptive Field）的定义是卷积神经网络每一层输出的特征图（feature map）上的像素点在输入图片上映射的区域大小。再通俗点的解释是，特征图上的一个点对应输入图上的区域，如图所示：

![image-20200724123645768](D:\Learn-DeepLearning\image\image-20200724123645768.png)

**感受视野的计算**

感受野大小的计算方式是采用从最后一层往下计算的方法，即先计算最深层在前一层上的感受野，然后逐层传递到第一层，使用的公式可以表示如下：
$$
RF_{i} =(RF_{i+1}-1)\times stride_{i}+Ksize_{i}
$$
其中

$RF_{i}$是第$i$层卷积层的感受野

$RF_{i+1}$是$i+1$层上的感受野

$stride_{i}$是卷积的步长

$Ksize_{i}$本层卷积核的大小。



## 7、单目标检测

### （1）制作数据

首先获取数据集，使用`background_image_downloads.py`文件爬取各种图片。

将小黄人随机地粘贴在图片上，并做好标记。

### （2）编写网络

```python
import torch
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,1,1),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16,32,3,1,1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,3,1,1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,3,1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128,256,3,1),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,3,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256,512,3,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512,1024,3,1),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 4)
        )
    def forward(self,x):
        return self.classifier(self.features(x).reshape(-1,1024))
```

### （3）加载数据

```python
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import torch,numpy
class DataYellow(Dataset):
    def __init__(self,root='D:/data/chapter2/shuju'):
        self.root = root
        super(DataYellow, self).__init__()
        self.dataset = os.listdir(self.root)
        self.totensor = ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,item):
        file_name = self.dataset[item]
        img = Image.open(f"{self.root}/{file_name}")
        img = self.totensor(img)
        target = file_name.split(".")[1:5]
        target = torch.tensor(numpy.array(target,dtype=numpy.float32))/300
        return img,target
```

### （4）训练

```python
from Chapter02.C02net import Net
from Chapter02.C02data import DataYellow
from torch.utils.data import DataLoader
import torch,os
import tqdm

class Train():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.dataset = DataYellow()
        self.data_loader = DataLoader(self.dataset,100,True)

        self.net = Net().to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(),0.0001)
        self.loss_function = torch.nn.MSELoss().to(self.device)
        if os.path.exists('D:/data/chapter2/chapter02'):
            self.net.load_state_dict(torch.load('D:/data/chapter2/chapter02'))


    def __call__(self):
        print('训练开始')
        for epoch in range(100):
            loss_sum = 0
            for img,target in tqdm.tqdm(self.data_loader):

                out = self.net(img.to(self.device))
                loss = self.loss_function(out,target.to(self.device))

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_sum += loss.cpu().detach().item()
            print(epoch,loss_sum/len(self.data_loader))
            torch.save(self.net.state_dict(),'D:/data/chapter2/chapter02')
```

进行训练：

```python
if __name__ == '__main__':
    train = Train()
    train()
```

![image-20200727105226229](D:\Learn-DeepLearning\image\image-20200727105226229.png)

完成训练。

测试结果。

![GIF 2020-7-27 14-43-19](D:\Learn-DeepLearning\image\c02test.gif)

测试完成。能较好的完成目标检测。

然而也是存在问题的。譬如，目标不在是这样方正的之后：

![GIF 2020-7-27 14-47-57](D:\Learn-DeepLearning\image\c02test2.gif)

可以看到，输出框依旧是正方形，这是由于训练集的所有标签是方形所致。