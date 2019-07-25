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
```
