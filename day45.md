# Numpy入门
```python
 Python中专门用来处理这些数值数组的工具： NumPy 包和 Pandas 包
 用 np 作为别名导入 NumPy
 import numpy as np
 ```
# 2.1 理解Python中的数据类型
```python
在 Python中，类型是动态推断的，可以将任何类型的数据指定给任何变量
x = 4
x = "four"
这里已经将 x 变量的内容由整型转变成了字符串，Python 变量不仅是它们的值，还包括了关于值的类型的一些额外信息
```
## 2.1.1 Python整型不仅仅是一个整型
```python
 Python 的整型其实是一个指针，指向包含这个 Python 对象所有信息的某个内存位置，其中包括可以转换成整型的字节
 ```
## 2.1.2 Python列表不仅仅是一个列表
```python
Python 中的标准可变多元素容器是列表。
In[1]: L = list(range(10))
       L
Out[1]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
In[2]: type(L[0])
Out[2]: int  #整型
In[3]: L2 = [str(c) for c in L]  #把L变成字符型
       L2
Out[3]: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
In[4]: type(L2[0])
Out[4]: str
In[5]: L3 = [True, "2", 3.0, 4]
[type(item) for item in L3]
Out[5]: [bool, str, float, int]
如果列表中的所有变量都是同一类型的，那么很多信息都会显得多余——将数据存储在固定类型的数组中应该会更高效
列表的优势是灵活，因为每个列表元素是一个包含数据和类型信息的完整结构体，而且列表可以用任意类型的数据填充。
## 2.1.3 Python中的固定类型数组
```python
内置的数组（array）模块（在 Python 3.3 之后可用）可以用于创建统一类型的密集数组：
In[6]: import array
       L = list(range(10))
       A = array.array('i', L)
       A
       Out[6]: array('i', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])   # 'i' 是一个数据类型码，表示数据为整型
```
## 2.1.4　从Python列表创建数组
```python
可以用 np.array 从 Python 列表创建数组
In[8]: # 整型数组:
       np.array([1, 4, 2, 5, 3])
Out[8]: array([1, 4, 2, 5, 3])
NumPy 要求数组必须包含同一类型的数据。如果类型不匹配， NumPy 将会向上转换（如果可行）
In[9]: np.array([3.14, 4, 2, 3])
Out[9]: array([ 3.14, 4. , 2. , 3. ])
用 dtype 关键字可以明确设置数组的数据类型
In[10]: np.array([1, 2, 3, 4], dtype='float32')
Out[10]: array([ 1., 2., 3., 4.], dtype=float32)
NumPy 数组可以被指定为多维的
In[11]: # 嵌套列表构成的多维数组
        np.array([range(i, i + 3) for i in [2, 4, 6]])  #左闭右开
Out[11]: array([[2, 3, 4],
                [4, 5, 6],
                [6, 7, 8]])
```
## 2.1.5　从头创建数组
```python
面对大型数组的时候，用 NumPy 内置的方法从头创建数组是一种更高效的方法
In[12]: # 创建一个长度为10的数组，数组的值都是0
        np.zeros(10, dtype=int)
Out[12]: array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  
In[13]: # 创建一个3× 5的浮点型数组，数组的值都是1
        np.ones((3, 5), dtype=float)
Out[13]: array([[ 1., 1., 1., 1., 1.],
                [ 1., 1., 1., 1., 1.],
                [ 1., 1., 1., 1., 1.]])
In[14]: # 创建一个3× 5的浮点型数组，数组的值都是3.14
        np.full((3, 5), 3.14)
Out[14]: array([[ 3.14, 3.14, 3.14, 3.14, 3.14],
                [ 3.14, 3.14, 3.14, 3.14, 3.14],
               [ 3.14, 3.14, 3.14, 3.14, 3.14]])
In[15]: # 创建一个3× 5的浮点型数组，数组的值是一个线性序列
        # 从0开始，到20结束，步长为2
        # （它和内置的range()函数类似）
        np.arange(0, 20, 2)
Out[15]: array([ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
In[16]: # 创建一个5个元素的数组，这5个数均匀地分配到0~1
        np.linspace(0, 1, 5)
Out[16]: array([ 0. , 0.25, 0.5 , 0.75, 1. ])
In[17]: # 创建一个3× 3的、在0~1均匀分布的随机数组成的数组
        np.random.random((3, 3))
Out[17]: array([[ 0.99844933, 0.52183819, 0.22421193],
                [ 0.08007488, 0.45429293, 0.20941444],
                 [ 0.14360941, 0.96910973, 0.946117 ]])
In[18]: # 创建一个3× 3的、均值为0、方差为1的
        # 正态分布的随机数数组
        np.random.normal(0, 1, (3, 3))
Out[18]: array([[ 1.51772646, 0.39614948, -0.10634696],
                 [ 0.25671348, 0.00732722, 0.37783601],
                [ 0.68446945, 0.15926039, -0.70744073]])
In[19]: # 创建一个3× 3的、 [0, 10)区间的随机整型数组
        np.random.randint(0, 10, (3, 3))
Out[19]: array([[2, 3, 4],
                [5, 7, 8],
                [0, 5, 0]])
In[20]: # 创建一个3× 3的单位矩阵
        np.eye(3)
Out[20]: array([[ 1., 0., 0.],
                [ 0., 1., 0.],
                [ 0., 0., 1.]])
In[21]: # 创建一个由3个整型数组成的未初始化的数组
        # 数组的值是内存空间中的任意值
        np.empty(3)
Out[21]: array([ 1., 1., 1.])
```
## 2.1.6 NumPy标准数据类型
```python
可以用一个字符串参数来指定数据类型
np.zeros(10, dtype='int16')
或者用相关的 NumPy 对象来指定
np.zeros(10, dtype=np.int16)
```
# 2.2 NumPy数组基础
```python
用 NumPy 数组操作获取数据或子数组，对数组进行分裂、变形和连接
```
## 2.2.1 NumPy数组的属性
```python
将用 NumPy 的随机数生成器设置一组种子值，以确保每次程序执行时都可以生成同样的随机数组
In[1]: import numpy as np
       np.random.seed(0) # 设置随机数种子
       x1 = np.random.randint(10, size=6) # 一维数组
       x2 = np.random.randint(10, size=(3, 4)) # 二维数组
       x3 = np.random.randint(10, size=(3, 4, 5)) # 三维数组
 nidm（数组的维度）、 shape（数组每个维度的大小）和size（数组的总大小）
 In[2]: print("x3 ndim: ", x3.ndim)
        print("x3 shape:", x3.shape)
        print("x3 size: ", x3.size)
x3 ndim: 3
x3 shape: (3, 4, 5)
x3 size: 60
 dtype是数组的数据类型
In[3]: print("dtype:", x3.dtype)
dtype: int64
表示每个数组元素字节大小的 itemsize，以及表示数组总字节大小的属性nbytes
In[4]: print("itemsize:", x3.itemsize, "bytes")
       print("nbytes:", x3.nbytes, "bytes")
itemsize: 8 bytes
nbytes: 480 bytes   #可以认为 nbytes 跟 itemsize 和 size 的乘积大小相等
# 2.2.2　数组索引： 获取单个元素
```python
在一维数组中，你也可以通过中括号指定索引获取第 i 个值（从 0 开始计数）
In[5]: x1
Out[5]: array([5, 0, 3, 3, 7, 9])
In[6]: x1[0]
Out[6]: 5
In[7]: x1[4]
Out[7]: 7
为了获取数组的末尾索引，可以用负值索引：
In[8]: x1[-1]
Out[8]: 9
In[9]: x1[-2]
Out[9]: 7
在多维数组中，可以用逗号分隔的索引元组获取元素：
In[10]: x2
Out[10]: array([[3, 5, 2, 4],
[7, 6, 8, 8],
[1, 6, 7, 7]])
In[11]: x2[0, 0]
Out[11]: 3
In[12]: x2[2, 0]
Out[12]: 1
In[13]: x2[2, -1]
Out[13]: 7
也可以用以上索引方式修改元素值：
In[14]: x2[0, 0] = 12
        x2
Out[14]: array([[12, 5, 2, 4],
                [ 7, 6, 8, 8],
                [ 1, 6, 7, 7]])
NumPy数组是固定类型的,当你试图将一个浮点值插入一个整型数组时，浮点值会被截短成整型。并且这种截短是自动完成的，不会给你提示或警告
In[15]: x1[0] = 3.14159 # 这将被截短
        x1
Out[15]: array([3, 0, 3, 3, 7, 9])
```
## 2.2.3　数组切片： 获取子数组
```python
用中括号获取单个数组元素，用切片（slice）符号获取子数组，切片符号用冒号（:）表示
x[start:stop:step]
默认值 start=0、 stop= 维度的大小（size of dimension） 和 step=1。
In[16]: x = np.arange(10)
        x
Out[16]: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
In[17]: x[:5] # 前五个元素
Out[17]: array([0, 1, 2, 3, 4])
In[18]: x[5:] # 索引五之后的元素
Out[18]: array([5, 6, 7, 8, 9])
In[19]: x[4:7] # 中间的子数组
Out[19]: array([4, 5, 6])
In[20]: x[::2] # 每隔一个元素
Out[20]: array([0, 2, 4, 6, 8])
In[21]: x[1::2] # 每隔一个元素，从索引1开始
Out[21]: array([1, 3, 5, 7, 9])
步长值为负时,逆序数组
In[22]: x[::-1] # 所有元素，逆序的
Out[22]: array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
In[23]: x[5::-2] # 从索引5开始每隔一个元素逆序
Out[23]: array([5, 3, 1])
多维切片用冒号分隔
In[24]: x2
Out[24]: array([[12, 5, 2, 4],
                [ 7, 6, 8, 8],
                [ 1, 6, 7, 7]])
In[25]: x2[:2, :3] # 两行，三列
Out[25]: array([[12, 5, 2],
                [ 7, 6, 8]])
In[26]: x2[:3, ::2] # 所有行，每隔一列
Out[26]: array([[12, 2],
                [ 7, 8],
                [ 1, 7]])
子数组维度也可以同时被逆序：
In[27]: x2[::-1, ::-1]
Out[27]: array([[ 7, 7, 6, 1],
                [ 8, 8, 6, 7],
               [ 4, 2, 5, 12]])
获取数组的行和列
 In[28]: print(x2[:, 0]) # x2的第一列
[12 7 1]
In[29]: print(x2[0, :]) # x2的第一行
[12 5 2 4]
在获取行时，出于语法的简介考虑，可以省略空的切片：
In[30]: print(x2[0]) #等于x2[0, :]
[12 5 2 4]
在获取行时，出于语法的简介考虑，可以省略空的切片：
In[30]: print(x2[0]) #等于x2[0, :]
[12 5 2 4]
在Python 列表中，切片是值的副本
 In[31]: print(x2)
[[12 5 2 4]
[ 7 6 8 8]
[ 1 6 7 7]]
从中抽取一个 2× 2 的子数组：
In[32]: x2_sub = x2[:2, :2]
print(x2_sub)
[[12 5]
[ 7 6]]
现在如果修改这个子数组，将会看到原始数组也被修改了！结果如下所示：
In[33]: x2_sub[0, 0] = 99
print(x2_sub)
[[99 5]
[ 7 6]]
In[34]: print(x2)
[[99 5 2 4]
[ 7 6 8 8]
[ 1 6 7 7]]
复制数组里的数据或子数组通过 copy() 方法实现：
 In[35]: x2_sub_copy = x2[:2, :2].copy()
print(x2_sub_copy)
[[99 5]
[ 7 6]]
如果修改这个子数组，原始的数组不会被改变：#经过复制后得来的数组改变其中的值原数组不会发生变化
In[36]: x2_sub_copy[0, 0] = 42
print(x2_sub_copy)
[[42 5]
[ 7 6]]
 In[37]: print(x2)
[[99 5 2 4]
[ 7 6 8 8]
[ 1 6 7 7]]
```
##2.2.4数组的变形
```python
数组变形最灵活的实现方式是通过 reshape() 函数来实现
将数字 1~9 放入一个3×3的矩阵中
In[38]: grid = np.arange(1, 10).reshape((3, 3))
print(grid)
[[1 2 3]
 [4 5 6]
 [7 8 9]]
另外一个变形模式是将一个一维数组转变为二维的行或列的矩阵,可以通过reshape 方法来实现，或者更简单地在一个切片操作中利用 newaxis 关键字：
 In[39]: x = np.array([1, 2, 3])
# 通过变形获得的行向量
x.reshape((1, 3))
Out[39]: array([[1, 2, 3]])
In[40]: # 通过newaxis获得的行向量
x[np.newaxis, :]
Out[40]: array([[1, 2, 3]])
In[41]: # 通过变形获得的列向量
x.reshape((3, 1))
Out[41]: array([[1],
                [2],
                [3]])
In[42]: # 通过newaxis获得的列向量
x[:, np.newaxis]
Out[42]: array([[1],
                [2],
                [3]])
```
## 2.2.5 数组拼接和分裂
```python
将多个数组合并为一个，或将一个数组分裂成多个。拼接或连接 NumPy 中的两个数组主要由 np.concatenate、 np.vstack 和 np.hstack 例程实现。 np.concatenate 将数组元组或数组列表作为第一个参数。
拼接
In[43]: x = np.array([1, 2, 3])
        y = np.array([3, 2, 1])
        np.concatenate([x, y])
Out[43]: array([1, 2, 3, 3, 2, 1])
可以一次性拼接两个以上数组：
In[44]: z = [99, 99, 99]
print(np.concatenate([x, y, z]))
[ 1 2 3 3 2 1 99 99 99]
np.concatenate 也可以用于二维数组的拼接：
In[45]: grid = np.array([[1, 2, 3],
                         [4, 5, 6]])
In[46]: # 沿着第一个轴拼接
np.concatenate([grid, grid])
Out[46]: array([[1, 2, 3],
                [4, 5, 6],
                [1, 2, 3],
                [4, 5, 6]])
In[47]: # 沿着第二个轴拼接（从0开始索引）
        np.concatenate([grid, grid], axis=1)   #axis 轴
Out[47]: array([[1, 2, 3, 1, 2, 3],
                [4, 5, 6, 4, 5, 6]])
沿着固定维度处理数组时，使用 np.vstack（垂直栈）和 np.hstack（水平栈）函数会更简洁：
In[48]: x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
# 垂直栈数组
np.vstack([x, grid])
Out[48]: array([[1, 2, 3],
                [9, 8, 7],
                [6, 5, 4]])
In[49]: # 水平栈数组
y = np.array([[99],
              [99]])
np.hstack([grid, y])
Out[49]: array([[ 9, 8, 7, 99],
                [ 6, 5, 4, 99]])
分裂
分裂可以通过 np.split、 np.hsplit 和 np.vsplit 函数来实现
可以向以上函数传递一个索引列表作为参数，索引列表记录的是分裂点位置（从1开始计）：
In[50]: x = [1, 2, 3, 99, 99, 3, 2, 1]
        x1, x2, x3 = np.split(x, [3, 5])  #分裂点在第3个和第5个数
        print(x1, x2, x3)
[1 2 3] [99 99] [3 2 1]
N 分裂点会得到 N + 1 个子数组。相关的 np.hsplit 和 np.vsplit 的用法也
类似：
In[51]: grid = np.arange(16).reshape((4, 4))
        grid
Out[51]: array([[ 0, 1, 2, 3],
                [ 4, 5, 6, 7],
                [ 8, 9, 10, 11],
                [12, 13, 14, 15]])
In[52]: upper, lower = np.vsplit(grid, [2])
        print(upper)
        print(lower)
[[0 1 2 3]
 [4 5 6 7]]
[[ 8 9 10 11]
 [12 13 14 15]]
In[53]: left, right = np.hsplit(grid, [2])
print(left)
print(right)
[[ 0 1]
 [ 4 5]
 [ 8 9]
 [12 13]]
[[ 2 3]
 [ 6 7]
 [10 11]
 [14 15]] 
 ```
# 2.3 NumPy数组的计算： 通用函数
```python
 NumPy 通用函数的重要性——它可以提高数组元素的重复计算的效率
 ```
## 2.3.1　缓慢的循环
```python  
Python 的相对缓慢通常出现在很多小操作需要不断重复的时候，比如对数组的每个元素做循环操作时。假设有一个数组，我们想计算每个元素的倒数，一种直接的解决方法是：
In[1]: import numpy as np
       np.random.seed(0)
       def compute_reciprocals(values):
           output = np.empty(len(values))
           for i in range(len(values)):
               output[i] = 1.0 / values[i]
           return output
       values = np.random.randint(1, 10, size=5)
       compute_reciprocals(values)
Out[1]: array([ 0.16666667, 1. , 0.25 , 0.25 , 0.125 ])
这一操作将非常耗时，并且是超出意料的慢！我们将用 IPython 的 %timeit 魔法函数来测量
In[2]: big_array = np.random.randint(1, 100, size=1000000)
       %timeit compute_reciprocals(big_array)
1 loop, best of 3: 2.91 s per loop  
```
## 2.3.2　通用函数介绍  
```python
NumPy 为很多类型的操作提供了非常方便的、静态类型的、可编译程序的接口，也被称作向量操作。  
NumPy 中的向量操作是通过通用函数实现的。通用函数的主要目的是对 NumPy 数组中的值执行更快的重复操作。  
比较以下两个结果：  
In[3]: print(compute_reciprocals(values))
       print(1.0 / values)
[ 0.16666667 1. 0.25 0.25 0.125 ]
[ 0.16666667 1. 0.25 0.25 0.125 ]  
如果计算一个较大数组的运行时间，可以看到它的完成时间比 Python 循环花费的时间更短：  
In[4]: %timeit (1.0 / big_array)  
100 loops, best of 3: 4.6 ms per loop  
```  
## 2.3.3 探索Numpy的通用函数  
```python
1. 数组的运算
NumPy 通用函数的使用方式非常自然，因为它用到了 Python 原生的算术运算符，标准的加、减、乘、除都可以使用：
In[7]: x = np.arange(4)
print("x =", x)
print("x + 5 =", x + 5)
print("x - 5 =", x - 5)
print("x * 2 =", x * 2)
print("x / 2 =", x / 2)
print("x // 2 =", x // 2) #地板除法运算
x = [0 1 2 3]
x + 5 = [5 6 7 8]
x - 5 = [-5 -4 -3 -2]
x * 2 = [0 2 4 6]
x / 2 = [ 0. 0.5 1. 1.5]
x // 2 = [0 0 1 1]
还有逻辑非、 ** 表示的指数运算符和 % 表示的模运算符的一元通用函数：
In[8]: print("-x = ", -x)
print("x ** 2 = ", x ** 2)
print("x % 2 = ", x % 2)
-x = [ 0 -1 -2 -3]
x ** 2 = [0 1 4 9]
x % 2 = [0 1 0 1]
所有这些算术运算符都是 NumPy 内置函数的简单封装器，例如 + 运算符就是一个 add 函数的封装器：
In[10]: np.add(x, 2)
Out[10]: array([2, 3, 4, 5])
绝对值
NumPy 也可以理解 Python 内置的绝对值函数：
In[11]: x = np.array([-2, -1, 0, 1, 2])
        abs(x)
Out[11]: array([2, 1, 0, 1, 2])
对应的 NumPy 通用函数是 np.absolute，该函数也可以用别名 np.abs 来访问：
         In[12]: np.absolute(x)
         Out[12]: array([2, 1, 0, 1, 2])
         In[13]: np.abs(x)
         Out[13]: array([2, 1, 0, 1, 2])
三角函数
首先定义一个角度数组：
         In[15]: theta = np.linspace(0, np.pi, 3)
现在可以对这些值进行一些三角函数计算：
In[16]: print("theta = ", theta)
        print("sin(theta) = ", np.sin(theta))
        print("cos(theta) = ", np.cos(theta))
        print("tan(theta) = ", np.tan(theta))
        theta = [ 0. 1.57079633 3.14159265]
        sin(theta) = [ 0.00000000e+00 1.00000000e+00 1.22464680e-16]
        cos(theta) = [ 1.00000000e+00 6.12323400e-17 -1.00000000e+00]
        tan(theta) = [ 0.00000000e+00 1.63312394e+16 -1.22464680e-16]
逆三角函数同样可以使用：
In[17]: x = [-1, 0, 1]
        print("x = ", x)
        print("arcsin(x) = ", np.arcsin(x))
        print("arccos(x) = ", np.arccos(x))
        print("arctan(x) = ", np.arctan(x))
x = [-1, 0, 1]
arcsin(x) = [-1.57079633 0. 1.57079633]
arccos(x) = [ 3.14159265 1.57079633 0. ]
arctan(x) = [-0.78539816 0. 0.78539816]
指数和对数
NumPy 中另一个常用的运算通用函数是指数运算：
In[18]: x = [1, 2, 3]
        print("x =", x)
        print("e^x =", np.exp(x))
        print("2^x =", np.exp2(x))
        print("3^x =", np.power(3, x))
x = [1, 2, 3]
e^x = [ 2.71828183 7.3890561 20.08553692]
2^x = [ 2. 4. 8.]
3^x = [ 3 9 27]
指数运算的逆运算，即对数运算也是可用的
In[19]: x = [1, 2, 4, 10]
        print("x =", x)
        print("ln(x) =", np.log(x))
        print("log2(x) =", np.log2(x))
        print("log10(x) =", np.log10(x))
x = [1, 2, 4, 10]
ln(x) = [ 0. 0.69314718 1.38629436 2.30258509]
log2(x) = [ 0. 1. 2. 3.32192809]
log10(x) = [ 0. 0.30103 0.60205999 1. ]
还有一些特殊的版本，对于非常小的输入值可以保持较好的精度：
In[20]: x = [0, 0.001, 0.01, 0.1]
print("exp(x) - 1 =", np.expm1(x))
print("log(1 + x) =", np.log1p(x))
exp(x) - 1 = [ 0. 0.0010005 0.01005017 0.10517092]
log(1 + x) = [ 0. 0.0009995 0.00995033 0.09531018]
NumPy 还提供了很多通用函数，包括双曲三角函数、比特位运算、比较运算符、弧度转化为角度的运算、取整和求余运算，等等
还有一个更加专用，也更加晦涩的通用函数优异来源是子模块 scipy.special。如果你希望对你的数据进行一些更晦涩的数学计算， scipy.special 可能包含了你需要的计算函数。
In[21]: from scipy import special
In[22]: # Gamma函数（广义阶乘， generalized factorials）和相关函数
        x = [1, 5, 10]
        print("gamma(x) =", special.gamma(x))
        print("ln|gamma(x)| =", special.gammaln(x))
        print("beta(x, 2) =", special.beta(x, 2))
gamma(x) = [ 1.00000000e+00 2.40000000e+01 3.62880000e+05]
ln|gamma(x)| = [ 0. 3.17805383 12.80182748]
beta(x, 2) = [ 0.5 0.03333333 0.00909091]
In[23]: # 误差函数（高斯积分）
        # 它的实现和它的逆实现
        x = np.array([0, 0.3, 0.7, 1.0])
        print("erf(x) =", special.erf(x))
        print("erfc(x) =", special.erfc(x))
        print("erfinv(x) =", special.erfinv(x))
erf(x) = [ 0. 0.32862676 0.67780119 0.84270079]
erfc(x) = [ 1. 0.67137324 0.32219881 0.15729921]
erfinv(x) = [ 0. 0.27246271 0.73286908 inf]
NumPy 和 scipy.special 中提供了大量的通用函数，这些包的文档在网上就可以查到，搜索“gamma function python”即可
指定输出
在进行大量运算时，指定一个用于存放运算结果的数组是非常有用的。你可以用这个特性将计算结果直接写入到你期望的存储位置。所有的通用函数都可以通过 out 参数来指定计算结果的存放位置：
In[24]: x = np.arange(5)
        y = np.empty(5)
        np.multiply(x, 10, out=y)
        print(y)
[ 0. 10. 20. 30. 40.]
这个特性也可以被用作数组视图，可以将计算结果写入指定数组的每隔一个元素的位置：
In[25]: y = np.zeros(10)
        np.power(2, x, out=y[::2])  #指数运算
        print(y)
[ 1. 0. 2. 0. 4. 0. 8. 0. 16. 0.]
对于上述例子中比较小的计算量来说，这两种方式的差别并不大。但是对于较大的数组，通过慎重使用 out 参数将能够有效节约内存。
聚合
二元通用函数有些非常有趣的聚合功能，这些聚合可以直接在对象上计算。
如果我们希望用一个特定的运算 reduce 一个数组，那么可以用任何通用函数的 reduce 方法，一个 reduce 方法会对给定的元素和操作重复执行，直至得到单个的结果。
对 add 通用函数调用 reduce 方法会返回数组中所有元素的和：
In[26]: x = np.arange(1, 6)
        np.add.reduce(x)
Out[26]: 15
对 multiply 通用函数调用 reduce 方法会返回数组中所有元素的乘积：
In[27]: np.multiply.reduce(x)
Out[27]: 120
如果需要存储每次计算的中间结果，可以使用 accumulate：
In[28]: np.add.accumulate(x)
Out[28]: array([ 1, 3, 6, 10, 15])
In[29]: np.multiply.accumulate(x)
Out[29]: array([ 1, 2, 6, 24, 120])
外积
任何通用函数都可以用 outer 方法获得两个不同输入数组所有元素对的函数运算结果
In[30]: x = np.arange(1, 6)
np.multiply.outer(x, x)
Out[30]: array([[ 1, 2, 3, 4, 5],
               [ 2, 4, 6, 8, 10]
               [ 3, 6, 9, 12, 15],
               [ 4, 8, 12, 16, 20],
               [ 5, 10, 15, 20, 25]])
通用函数另外一个非常有用的特性是它能操作不同大小和形状的数组，一组这样的操作被称为广播（broadcasting）。























