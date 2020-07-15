# Chapter 1

## 一、全连接神经网络

### 1、神经元

![img](D:\Learn-DeepLearning\image\b783af0ad19214c94498e94414f5c501.jpg)

<img src="D:\Learn-DeepLearning\image\perceptron.jpg" alt="img"  />

通过对人脑神经细胞的研究后，一个神经元有一条轴突和若干的树突，即多个输入，一个输出。

图中，n维向量：[a1,a2,a3,…,an]的输入，[w1,w2,w3,…,wn]为输入分量连接到感知机的权重(weight)，b为偏置，f(sum)为激活函数，t为该神经元的输出。其数学表达式为：
$$
sum=a_{1}w_{1}+a_{2}w_{2}+a_{3}w_{3}+...+a_{n}w_{n}+b
$$

$$
t = f(x)
$$

写作矩阵的形式为：
$$
sum = \begin{bmatrix}
a_{1} &a_{2}  &a_{3} &...  & a_{n}
\end{bmatrix}*\begin{bmatrix}
w_{1}\\ 
w_{2}\\ 
w_{3}\\ 
...\\ 
w_{n}
\end{bmatrix}+\begin{bmatrix}
b
\end{bmatrix}
$$

$$
t = f(A^{T}W+B)
$$



### 2、激活函数

激活函数可以为神经元提供非线性能力，如果没有非线性的激活函数，只是将众多的线性神经元叠加，只会得到由众多的w参数和b参数控制的一条直线：
$$
y=(w_{n}...(w_{3}(w_{2}(w_{1}x+b_{1})+b_{2})+b_{3})+...+b_{n})
$$


下面是几种常见的激活函数：

#### 1）sigmoid函数

表达式为：
$$
sigmoid(z)=\frac{1}{1+e^{-z}}
$$
![img](D:\Learn-DeepLearning\image\u=1303190971,3760905646&fm=26&gp=0.jpg)

sigmoid函数的值域为(0,1)。从图中可知，当因变量取值为零时，sigmoid曲线的斜率最大，为0.25。当其取值越大或是越小时，则斜率(梯度)就越小。如果z取值过大，那么参数的更新速度就可能会变得非常缓慢。

#### 2）tanh函数

$$
tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}
$$

![img](D:\Learn-DeepLearning\image\u=88831987,3655763565&fm=26&gp=0.jpg)

如果选择 tanh 作为隐藏层的激活函数，效果几乎总是比 sigmoid 函数要好，因为 tanh 函数的输出介于 -1 与 1 之间，激活函数的平均值就更接近于 0，而不是 0.5，这让下一层的学习更加方便一点。

和 sigmoid 函数类似，当因变量 zzz 取值为 0 时，tanh 曲线的斜率（梯度）最大，计算出来为 1。当因变量 zzz 的取值越大或越小时，sigmoid 曲线的斜率（梯度）就越小。所以 tanh 也会面临和 sigmiod 函数同样的问题：当 zzz 很大或很小时，tanh 曲线的斜率（梯度）几乎为零，这对于执行梯度下降算法十分不利。

#### 3）ReLU函数

$$
ReLU(z)=max(0,x)
$$

![img](D:\Learn-DeepLearning\image\relu.jpg)

ReLU (Rectified Linear Unit)，即修正线性单元。现在 ReLU 已经成为了隐层激活函数的默认选择。如果不确定隐层应该使用哪个激活函数，可以先尝试一下 ReLU。

当 z > 0 时，曲线斜率为 1；当 z < 0 时，曲线的斜率为 0；当 z = 0 时，曲线不可导，斜率就失去了意义，但在实际编程的时候可以令其为 1 或 0，尽管在数学上这样不正确，但不必担心，因为出现 z = 0 的概率非常非常小。ReLu函数的好处在于对于大部分的 z，激活函数的斜率（导数）和 0 差很远。所以在隐层中使用 ReLU 激活函数，神经网络的学习速度通常会快很多。ReLU 也有一个缺点，那就是当 z 为负数时，ReLU 曲线的斜率为 0，不过这在实践中不会有什么问题，因为有足够多的隐层节点使 z 大于 0。

#### 4）LeakyReLU函数

$$
ReLU(z)=max(kz,x)，k=0.1
$$

![img](D:\Learn-DeepLearning\image\lrelu.jpg)

#### 5）ELU函数

$$
ELU(z)=\left\{\begin{matrix}
z & x\geq 0\\ 
\sigma (e^{z}-1) & x<0
\end{matrix}\right.
$$

![img](D:\Learn-DeepLearning\image\u=800186656,2093919147&fm=15&gp=0.jpg)



### 3、深度神经网络与深度学习

![img](D:\Learn-DeepLearning\image\v2-e166111e4ee7dafe93ca6e71c28a21a7_b.jpg)

- 多个神经元构成一层神经网络，然后神经网络按一层一层构建，就构成了深度神经网络
- 网络由输入层，隐藏层，输出层构成
- 深度神经网络的学习过程被称为深度学习



### 4、输出函数

激活函数是使用在隐藏层之间的，主要是为了给网络提供非线性能力

输出函数的作用是为了限制输出的值域。

线性输出函数：
$$
linear(x)=x
$$


sigmoid函数将输出限制在（0,1）之间，多用于两分类：

$$
sigmoid(z)=\frac{1}{1+e^{-z}}
$$


softmax函数，用于多分类，也叫归一化指数函数：

$$
softmax(x_{i})=\frac{e^{x_{i}}}{\sum_{j}^{}e^{x_{j}}}
$$


### 5、pytorch中读取转化数据

使用pytorch的torch.utils.data.Dataset处理自己的数据集。

创建一个类，继承于Dataset。

其中，只重载`__init__`函数、`__len__` 函数、`__getitem__`函数

示例：

```python
class MNISTDataset(Dataset):
    '''只重载  __init__函数
              __len__ 函数
              __getitem__函数'''

    def __init__(self,is_train=True):
        '''初始化数据集(将数据集读取进来)'''
        '''一般由于数据过大，只将数据存储的路径读取进来'''	
        
        print("init")
        pass

    def __len__(self):
        '''统计数据的个数'''
        print('len')
        return 2
        pass

    def __getitem__(self, index):
        '''每条数据的处理方式'''
        print(index)
        pass
```

其中`__init__`函数在实例化对象时执行，`__len__` 函数在调用len()时执行，而`__getitem__`则在索引时执行

以下调用函数：

```python
mnist_dataset = MNISTDataset()#实例化对象
print(len(mnist_dataset))
mnist_dataset[1]
```

输出结果：

```python
init
len
2
1
```



读取手写数字的代码

```python
'''使用pytorch处理自己的数据集'''
from torch.utils.data import Dataset
import os
import cv2
import numpy
'''创建自己的数据集'''

"""
Data 模块
"""
class MNISTDataset(Dataset):
    '''只重载  __init__函数
              __len__ 函数
              __getitem__函数'''

    def __init__(self,root,is_train=True):
        '''初始化数据集(将数据集读取进来)'''
        '''一般由于数据过大，只将数据存储的路径读取进来'''

        self.root = root#文件的父路径
        self.dataset = []#记录所有数据

        '''根据is_train选择加载的数据集'''
        sub_dir = "TRAIN" if is_train else "TEST"
        for tag in os.listdir(f'{root}/{sub_dir}'):
            img_dir = f"{self.root}/{sub_dir}/{tag}"
            for img_filename in os.listdir(img_dir):
                img_path = f"{img_dir}/{img_filename}"

                """封装成数据集,img_path为数据的路径，而后一个为标签"""
                self.dataset.append((img_path,int(tag)))
        pass

    def __len__(self):
        '''统计数据的个数'''

        pass
        '''返回数据集的长度'''
        return len(self.dataset)

    def __getitem__(self, index):
        '''每条数据的处理方式'''

        pass

        data = self.dataset[index]
        image_data = cv2.imread(data[0],cv2.IMREAD_GRAYSCALE)

        '''调整数据形状'''
        image_data = image_data.reshape(-1)

        '''数据的归一化'''
        image_data = image_data/255

        """one-hot编码"""
        target = numpy.zeros(10)
        target[data[1]] = 1

        return image_data,target


if __name__ == '__main__':
    dataset = MNISTDataset('D:/data/MNIST_IMG')
    print(dataset[1])
```

调试结果：

```python
(array([0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.00392157,
       0.        , 0.        , 0.        , 0.00784314, 0.        ,
       0.00784314, 0.        , 0.00784314, 0.        , 0.        ,
       0.00784314, 0.        , 0.        , 0.00784314, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.00392157, 0.00784314, 0.        , 0.00784314,
       0.00392157, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.00784314, 0.        ,
       0.00392157, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
		..................
       0.        , 0.        , 0.        , 0.        ]), array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
```

完成数据的读取。







