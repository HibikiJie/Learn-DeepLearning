# Project 1

[TOC]

## MTCNN

## 多任务卷积神经网络

《Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks》

## 1、目标识别的步骤

1. 目标侦测
2. 特征提取
3. 特征对比

## 2、目标侦测

单目标和多目标

![image-20200729103007950](D:\Learn-DeepLearning\image\image-20200729103007950.png)

定位目标，给左上角和右下角的两个坐标点，或是中心点和宽高的标签，进行回归运算。

如果图像目标中个数不定，则之前所用的无效，因为有全连接。

如何实现输出的目标不定，这需要利用到卷积神经网络的特性。

将**图片切分**，依次传入神经网络，让他判断是否有目标，目标所在的位置。

此时应输出5个值，x1、y1、x2、y2、置信度c。

此时，神经网络为**单目标检测的网络**。

然而图像切分时，因为神经网络使用特性，切分的大小应该固定大小。但图片中人脸有大有小，又需要使用不同大小的切分框。

使用不同大小的切分框，显然是不利的。可以通过**缩放图片**的形式，变相地改变切分框在图片上切分的大小。

如此将图像进行不同尺度的变换，**构建图像金字塔**，以适应不同大小的目标的进行检测。

在切分框扫描图像时，如果按照等分切分，则极大可能无法切分出完整人脸。因此**步长应为1**，则可保证所有的人脸可以完整切分出来。

实现切分过程，需要将切分出来的图像依次传入神经网络。切分框定为12*12的大小。

![image-20200729114002279](D:\Learn-DeepLearning\image\image-20200729114002279.png)

控制该神经网络的感受野为12*12，即输出的特征图的单个值，则对应的原图上切分。并输出5个通道，来分别代表坐标点和置信度。

则该单目标侦测网络，输入为（1,3,12,12）的数据，输出为（
1,5,1,1），并且该网络无全连接层。这便是MTCNN的P-Net网络。

## 3、MTCNN网络结构

由三级网络构成：

1. **P-Net**
2. **R-Net**
3. **O-Net**

### P-Net

![image-20200729135355746](D:\Learn-DeepLearning\image\image-20200729135355746.png)

### R-Net

![image-20200729135443930](D:\Learn-DeepLearning\image\image-20200729135443930.png)

### O-Net

![image-20200729135546856](D:\Learn-DeepLearning\image\image-20200729135546856.png)

MTCNN为了兼顾性能和准确率，避免滑动窗口加分类器等传统思路带来的巨大的性能消耗，先使用小模型生成有一定可能性的目标区域候选框，然后在使用更复杂的模型进行细分类和更高精度的区域框回归，并且让这一步递归执行，以此思想构成三层网络，分别为P-Net、R-Net、O-Net，实现快速高效的人脸检测。在输入层使用图像金字塔进行初始图像的尺度变换，并使用P-Net生成大量的候选目标区域框，之后使用R-Net对这些目标区域框进行第一次精选和边框回归，排除大部分的负例，然后再用更复杂的、精度更高的网络O-Net对剩余的目标区域框进行判别和区域边框回归。

## 4、NMS非极大值抑制

因为图像作了图像金字塔，并且按步长为1进行窗口滑动，这件必然会产生非常多的候选矩形框，这些矩形框有很多是指向同一目标，因此就存在大量冗余的候选矩形框。非极大值抑制算法的目的正在于此，它可以消除多余的框，找到最佳的物体检测位置。

非极大值抑制（Non-Maximum Suppression）的**思想**是**搜索局部极大值，抑制非极大值元素**。

NMS严格按照搜索局部极大值，抑制非极大值元素的思想来实现的，具体的实现步骤如下：

1. 设定目标框的置信度阈值
2. 根据置信度降序排列候选框列表
3. 选取置信度最高的框A添加到输出列表，并将其从候选框列表中删除
4. 计算A与候选框列表中的所有框的**IoU**值，删除大于阈值的候选框
5. 重复上述过程，直到候选框列表为空，返回输出列表

### 重叠度IOU

其中**IoU**(Intersection over Union)为交并比，如图所示，IoU相当于两个区域交叉的部分除以两个区域的并集部分得出的结果。

![image-20200731102947464](D:\Learn-DeepLearning\image\image-20200731102947464.png)

交并比计算：

```python
import numpy


def compute_iou(box1, box2):
    """
    计算两个候选框之间的交并比
    :param box1: 候选框的两个坐标值
    :param box2: 候选框的两个坐标值
    :return:
        iou:iou of box1 and box2
    """
    '''分别获取坐标'''
    x1, y1 = box1[0]
    x2, y2 = box1[1]
    x3, y3 = box2[0]
    x4, y4 = box2[1]

    '''计算两候选框之间的矩形相交区域的两角点的坐标'''
    x_1 = numpy.max([x1, x3])
    y_1 = numpy.max([y1, y3])
    x_2 = numpy.min([x2, x4])
    y_2 = numpy.min([y2, y4])

    '''计算相交区域的面积'''
    intersection_area = (numpy.max([0, x_2 - x_1])) * (numpy.max([0, y_2 - y_1]))

    '''计算两候选框各自的面积'''
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    '''返回交并比'''
    return intersection_area / (area1 + area2 - intersection_area)
```

### NMS代码实现

```python
import torch

'''
1. 设定目标框的置信度阈值
2. 根据置信度降序排列候选框列表
3. 选取置信度最高的框A添加到输出列表，并将其从候选框列表中删除
4. 计算A与候选框列表中的所有框的IoU值，删除大于阈值的候选框
5. 重复上述过程，直到候选框列表为空，返回输出列表
'''


def compute_iou(box1, box2):
    '''
    计算两候选框之间的交并比
    :param box1: 第一个候选框
    :param box2: 第二个候选框
    :return: 两候选框之间的交并比
    '''
    area1 = (box1[2] - box1[1]) * (box1[3] - box1[1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    x1 = torch.max(box1[0], box2[:, 0])
    y1 = torch.max(box1[1], box2[:, 1])
    x2 = torch.min(box1[2], box2[:, 2])
    y2 = torch.min(box1[3], box2[:, 3])
    intersection_area = torch.max(torch.tensor(0, dtype=torch.float32), x2 - x1) * torch.max(
        torch.tensor(0, dtype=torch.float32), y2 - y1)
    return intersection_area / (area1 + area2)


def non_maximum_suppression(predict_dict, threshold):
    '''
    非极大值抑制
    :param predict_dict: 输入的候选框
    :param threshold: 交并比的阈值
    :return: 非极大值抑制之后的候选框们
    '''
    '''阈值设定'''
    threshold = threshold

    '''获取得分'''
    score = predict_dict[:, 4]

    '''设定存放输出的数据的索引'''
    picked_boxes = []

    '''获取置信度排序，按照降序排列'''
    order = torch.argsort(score, descending=True)

    while order.size()[0] > 0:
        '''选出当前置信度最大的索引'''
        picked_boxes.append(order[0])

        '''将当前最大置信度的候选框，与剩下的候选框计算交并比'''
        iou = compute_iou(predict_dict[order[0]], predict_dict[order[1:]])

        '''保留下小于阈值的候选框的位置索引'''
        order = order[torch.where(iou < threshold)[0] + 1]

    return predict_dict[[picked_boxes]]


if __name__ == '__main__':
    predict_dict = torch.tensor([[59, 120, 137, 368, 0.124648176],
                                 [221, 89, 369, 367, 0.35818103],
                                 [54, 154, 148, 382, 0.13638769]])
    print(non_maximum_suppression(predict_dict, 0.6))
```

## 5、特征图到原图的映射

![image-20200731160248073](D:\Learn-DeepLearning\image\image-20200731160248073.png)

通过网络求得特征图，此时通过特征图上的位置，反求对应的在原图上的位置。

计算公式为：
$$
(x1,y1)=(x',y')\times Stride
$$

$$
(x2,y2)=(x',y')\times Stride + kernelsize-1
$$

(x1,y1)对应候选框左上角于原图的位置

(x2,y2)对应候选框右下角于原图的位置

(x',y')为特征图上的一个值的位置

kernelsize为该网络感受野的大小。

Stride为扫描时的步长。

## 6、MTCNN训练样本

训练数据集：CelebA

CelebA是CelebFaces Attribute的缩写，意即名人人脸属性数据集，其包含10,177个名人身份的202,599张人脸图片，每张图片都做好了特征标记，包含人脸bbox标注框、5个人脸特征点坐标以及40个属性标记

官方网址：[Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

然而该数据集质量不高，也可以采用自己手动标注的形式完成。

下载软件：[精灵标注助手](http://www.jinglingbiaozhu.com/)

安装完成后，打开软件

新建

![image-20200731162322927](image\image-20200731162322927.png)

![image-20200731162649117](image\image-20200731162649117.png)

选择项目类型，图片所在文件夹，分类的值，这里只需要人脸的位置，则类别值仅为人脸。

![image-20200731162814877](image\image-20200731162814877.png)

给人脸画上矩形框，即可完成标注。

标注完成后，导出即可

训练的有置信度，和人脸位置。

如果训练置信度，需要有正负样本，即有人脸和无人脸。同时，为了训练识别人脸位置的能力，需要人脸随机处于任意位置。

因此需要根据标记好的图片，制造样本。

其中，截取出来的图片与标签人脸框的交并比

0~0.3，为非人脸

0.65~1.00，为人脸

0.4~0.65,为部分人脸

0.3~0.65，部分舍弃不用，地标



训练样本比例，负样本：正样本：部分正样本：地标=3:1:1:2

nn

完成人脸标注后，切记**对数据集副本**进行操作，**不要对原始数据集进行任何操作**。以免丢失数据。

### 制作样本。

训练模型时，使用的是，12x12、24x24、48x48规格的图片，且为单目标检测。所以，需要从原图片上，扣取这样的样本，并自动计算出归一化之后的坐标标签值。

详细代码见**Proj1DataProsscesing.py**





## 6、数据集

制作数据集，同样的需要初始化，在\_\_init\_\_()方法中创建，数据集，加载数据的读取目录以及标签。然后重写\_\_len\_\_()方法，以及\__getitem__()方法。

详细代码见**Proj1FaceDataSet.py**

```python
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
from time import time
import torch


class FaceDataSet(Dataset):

    def __init__(self, path='D:/data/object1/train',image_size='48'):
        super(FaceDataSet, self).__init__()
        ''' '''

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, item):
        ''' '''
```

## 7、构建模型

按照[MTCNN网络](#3、MTCNN网络结构)中[P-Net](#P-Net)、[R-Net](#R-Net)、[O-Net](#O-Net)网络结构编写模型代码

详细代码见**Proj1Net.py**



## 8、编写训练器

在训练损失时，变换输出的形状`out_confidence = out_confidence.reshape(-1, 1)`、`out_coordinate = out_coordinate.reshape(-1, 4)`，如此可以使代码满足三个模型的同时使用。同时，在输出和标签中，分别取出部分样本和正样本训练回归、负样本和正样本训练分类能力。

详细代码见**Proj1Trainer.py**

训练完网络......

![image-20200810170200342](D:\Learn-DeepLearning\image\image-20200810170200342.png)

其中O网络的损失曲线见上图所示。至此网络的参数训练完成。



## 9、正向使用过程

编写一个探索者（Explorer）的类，其中包含初始化、探索、p网络探索、r网络探索、o网络探索、以及扣取图片等方法。

详细代码见**Proj1Explorer.py**、**Proj1GUI.py**、**Proj1Main.py**

测试使用结果：

```python
    def catch_face(path_txt):
        image = Image.open(path_txt)
        image = image.convert('RGB')
        boxes = self.explorer.explore(image)
        if boxes.shape[0] == 0:
            pass
        else:
            cors = boxes[:, 0:4]
            draw = ImageDraw.Draw(image)
            for cor in cors:
                draw.rectangle(tuple(cor), outline='red', width=2)
            image.show()
```

![image-20200810162916621](D:\Learn-DeepLearning\image\image-20200810162916621.png)

结果显示，能完成全部的人脸检测

观察输出结果：

```python
tensor([[234.5096,  31.3873, 273.6535,  86.5899,   1.0000],
        [299.9764, 142.6213, 352.2200, 216.3686,   1.0000],
        [301.3135,  34.9491, 338.5693,  87.4093,   1.0000],
        [191.6967, 132.7279, 234.9627, 194.1653,   1.0000],
        [125.6473,  49.3581, 166.0767, 106.7610,   1.0000]])
```

所有人脸的置信度均为1

```python
P网络监测时间： 0.12192559242248535
torch.Size([3581, 5])
NMS_COST_TIME: 0.354780912399292
p: 0.47870540618896484
torch.Size([677, 5])
torch.Size([292, 5])
NMS_COST_TIME: 0.05196738243103027
r: 0.7785205841064453
torch.Size([8, 5])
NMS_COST_TIME: 0.00299835205078125
o: 0.9244303703308105
```

可见网络的前向过程并不太花时间，计算时间主要消耗在了非极大值抑制上。