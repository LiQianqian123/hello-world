# CNN
```python
卷积运算
python:conv_forward
tensorflow:tf.nn.conv2d
keras：Conv_2D
原图为NxN的图片，卷积核（滤波器是FxF）,那么输出的图片就是(N-F+1)x(N-F+1),这样意味着每次做卷积图像都会缩小，也就会丢掉边缘信息。
为了解决上述的两个问题，可以在输入图像的边缘填充一层像素，即长和宽的尺寸各增加了2，此时p(padding) = 1
没有进行了padding的卷积叫做valid convolutions,反之叫做same convolutions,且p = (f-1)/2,大多数情况下滤波器为奇数滤波器，即f的值为奇数。
卷积步长(stride)为s,padding为p,那么卷积后的图的尺寸为：[(n+2p-f)/s]+1      若上式的商不是整数，则向下取整，即对z进行地板除：floor(z)
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/41-1.png)
```python
每一个过滤器的通道数（channel）都必须与原图保持一致,若有多个滤波器，那么每个滤波器都承担不一样的任务，有的可能是垂直边缘检测，有的可能是水平边缘检测，起到的检测效果各不相同。
卷积神经网络的一层由两步组成：
（1）  z[1]=w[1]z[0]+b[1],z[0]为输入图，w[1]为卷积核，b[1]为偏差。   #对通过卷积计算后得到的矩阵加上一个相同的偏差
（2）  a[1]=g(z[1])    #g()表示类似于ReLu的非线性激活函数函数，此式表示对z[1]进行非线性处理.
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/41-2.png)
```python
上图都得到的4x4x2的矩阵就成为神经网络的下一层，也就是激活层
几个参数：
f[l]=filter size 表示l层中过滤器的大小为fxf,即过滤器的尺寸
p[l]=padding     p[l]标记padding的数量
s[l]=stride      s[l]标记步幅
一个典型的卷积网络通常分为三层
卷积层(Convolution)——池化层(pooling)——全连接层(Fully connected)
池化层，大多数池化层采用最大池化：根据输格子的数量对输入进行划分，选取每个区域中的最大值存放在对应的格子中
还有一种池化是平均池化，即对区域中的数取平均值来存放在对应的输出的格子中。
全连接层：每个卷积和池化步骤都是隐藏层。在此之后，我们有一个完全连接的层，然后是输出层。完全连接的层是典型的神经网络（多层感知器）类型的层，与输出层相同。
```
# 程序实现
```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
# cifar10是一个包含60000张图片的数据集。其中每张照片为32*32的彩色照片，每个像素点包括RGB三个数值，数值范围 0 ~ 255。所有照片分属10个不同的类别，
# 分别是 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck',
# 其中五万张图片被划分为训练集，剩下的一万张图片属于测试集。
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ImageDataGenerator是keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。
比如进行旋转，变形，归一化等等。
from tensorflow.keras.models import Sequential
# Sequential序列惯性模型，序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉。
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# Dense是全连接层，
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
# Activation 是激活函数。
# dropout顾名思义就是丢弃，丢弃的是每一层的某些神经元。在DNN深度网络中过拟合问题一直存在，dropout技术可以在一定程度上防止网络的过拟合。

from tensorflow.keras.layers import Conv2D, MaxPooling2D
# Conv2D （卷积层）,MaxPooling2D（池化层）

import pickle

pickle_in = open("E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("E:/pycharm/practice/kagglecatsanddogs_3367a/PetImages/y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。
# 为整数意为各个维度值相同且为该数字。

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
```
