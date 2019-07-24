# 2.7　花哨的索引
```python
花哨的索引和前面那些简单的索引非常类似，但是传递的是索引数组，而不是单个标量。花哨的索引让我们能够快速获得并修改复杂的数组值的子数据集。
```
## 2.7.1　探索花哨的索引
```python
花哨的索引在概念上非常简单，它意味着传递一个索引数组来一次性获得多个数组元素。
例如以下数组：
In[1]: import numpy as np
       rand = np.random.RandomState(42)
       x = rand.randint(100, size=10)
       print(x)
[51 92 14 71 60 20 82 86 74 74]
假设我们希望获得三个不同的元素，可以用以下方式实现：
In[2]: [x[3], x[7], x[2]]
Out[2]: [71, 86, 14]
另外一种方法是通过传递索引的单个列表或数组来获得同样的结果：
In[3]: ind = [3, 7, 4]
       x[ind]
Out[3]: array([71, 86, 60])
利用花哨的索引，结果的形状与索引数组的形状一致，而不是与被索引数组的形状一致：
In[4]: ind = np.array([[3, 7],
                       [4, 5]])
       x[ind]
Out[4]: array([[71, 86],
               [60, 20]])
花哨的索引也对多个维度适用。假设我们有以下数组：
In[5]: X = np.arange(12).reshape((3, 4))
       X
Out[5]: array([[ 0, 1, 2, 3],
               [ 4, 5, 6, 7],
               [ 8, 9, 10, 11]])
和标准的索引方式一样，第一个索引指的是行，第二个索引指的是列：
In[6]: row = np.array([0, 1, 2])
       col = np.array([2, 1, 3])
       X[row, col]
Out[6]: array([ 2, 5, 11])
这里需要注意，结果的第一个值是 X[0, 2]，第二个值是 X[1, 1]，第三个值是 X[2, 3]。
在花哨的索引中，索引值的配对遵循 2.5 节介绍过的广播的规则。因此当我们将一个列向
量和一个行向量组合在一个索引中时，会得到一个二维的结果：
In[7]: X[row[:, np.newaxis], col]
Out[7]: array([[ 2, 1, 3],
               [ 6, 5, 7],
               [10, 9, 11]])
这里，每一行的值都与每一列的向量配对，正如我们看到的广播的算术运算：
In[8]: row[:, np.newaxis] * col
Out[8]: array([[0, 0, 0],
               [2, 1, 3],
               [4, 2, 6]])
这里特别需要记住的是，花哨的索引返回的值反映的是广播后的索引数组的形状，而不是
被索引的数组的形状。
2.7.2　组合索引
```python
花哨的索引可以和其他索引方案结合起来形成更强大的索引操作：
In[9]: print(X)
[[ 0 1 2 3]
[ 4 5 6 7]
[ 8 9 10 11]]
可以将花哨的索引和简单的索引组合使用：
In[10]: X[2, [2, 0, 1]]
Out[10]: array([10, 8, 9])
也可以将花哨的索引和切片组合使用：
In[11]: X[1:, [2, 0, 1]]
Out[11]: array([[ 6, 4, 5],
                [10, 8, 9]])
更可以将花哨的索引和掩码组合使用：
In[12]: mask = np.array([1, 0, 1, 0], dtype=bool)
X[row[:, np.newaxis], mask]
Out[12]: array([[ 0, 2],
                [ 4, 6],
                [ 8, 10]])
索引选项的组合可以实现非常灵活的获取和修改数组元素的操作。
## 2.7.3　示例： 选择随机点
花哨的索引的一个常见用途是从一个矩阵中选择行的子集。例如我们有一个 N× D 的矩阵，表示在 D 个维度的 N 个点。以下是一个二维正态分布的点组成的数组：
In[13]: mean = [0, 0]
cov = [[1, 2],
[2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
X.shape
Out[13]: (100, 2)
利用将在第 4 章介绍的画图工具，可以用散点图将这些点可视化（如图 2-7 所示）：
In[14]: %matplotlib inline
        import matplotlib.pyplot as plt
        import seaborn; seaborn.set() # 设置绘图风格
        plt.scatter(X[:, 0], X[:, 1]);我们将利用花哨的索引随机选取 20 个点——选择 20 个随机的、不重复的索引值，并利用
这些索引值选取到原始数组对应的值：
In[15]: indices = np.random.choice(X.shape[0], 20, replace=False)
        indices
Out[15]: array([93, 45, 73, 81, 50, 10, 98, 94, 4, 64, 65, 89, 47, 84, 82,80, 25, 90, 63, 20])72 
In[16]: selection = X[indices] # 花哨的索引
        selection.shape
Out[16]: (20, 2)
将选中的点在图上用大圆圈标示出来
In[17]: plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
        plt.scatter(selection[:, 0], selection[:, 1],
        facecolor='none', edgecolor='b', s=200);
这种方法通常用于快速分割数据，即需要分割训练 / 测试数据集以验证统计模型（详情请参见 5.3 节）时，以及在解答统计问题时的抽样方法中使用。
```
2.7.4　用花哨的索引修改值
```python
正如花哨的索引可以被用于获取部分数组，它也可以被用于修改部分数组。例如，假设我们有一个索引数组，并且希望设置数组中对应的值：
In[18]: x = np.arange(10)
        i = np.array([2, 1, 8, 4])
        x[i] = 99
print(x)
[ 0 99 99 3 99 5 6 7 99 9]
可以用任何的赋值操作来实现，例如：
In[19]: x[i] -= 10
print(x)
[ 0 89 89 3 89 5 6 7 89 9]
不过需要注意，操作中重复的索引会导致一些出乎意料的结果产生，如以下例子所示：
## 2.7.4　用花哨的索引修改值
正如花哨的索引可以被用于获取部分数组，它也可以被用于修改部分数组。
In[18]: x = np.arange(10)
        i = np.array([2, 1, 8, 4])
        x[i] = 99
        print(x)
[ 0 99 99 3 99 5 6 7 99 9]
可以用任何的赋值操作来实现，例如：
In[19]: x[i] -= 10
        print(x)
[ 0 89 89 3 89 5 6 7 89 9]
In[20]: x = np.zeros(10)
        x[[0, 0]] = [4, 6]
        print(x)
[ 6. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
4 去哪里了呢？这个操作首先赋值 x[0] = 4，然后赋值 x[0] = 6，因此当然 x[0] 的值为 6。
In[21]: i = [2, 3, 3, 4, 4, 4]
        x[i] += 1
        x
Out[21]: array([ 6., 0., 1., 1., 1., 0., 0., 0., 0., 0.])
希望累加，该怎么做呢？你可以借助通用函数中的 at() 方法
In[22]: x = np.zeros(10)
        np.add.at(x, i, 1)
        print(x)
[ 0. 0. 1. 2. 3. 0. 0. 0. 0. 0.]
at() 函数在这里对给定的操作、给定的索引（这里是 i）以及给定的值（这里是 1）执行的是就地操作。另一个可以实现该功能的类似方法是通用函数中的 reduceat() 函数，你可以在 NumPy 文档中找到关于该函数的更多信息。
## 2.7.5　示例： 数据区间划分
用这些方法有效地将数据进行区间划分并手动创建直方图。例如，假定我们有 1000
个值，希望快速统计分布在每个区间中的数据频次，可以用 ufunc.at 来计算：
In[23]: np.random.seed(42)
        x = np.random.randn(100)
# 手动计算直方图
        bins = np.linspace(-5, 5, 20)
        counts = np.zeros_like(bins)
# 为每个x找到合适的区间
        i = np.searchsorted(bins, x)
# 为每个区间加上1
        np.add.at(counts, i, 1)
计数数组 counts 反映的是在每个区间中的点的个数，即直方图分布
In[24]: # 画出结果
plt.plot(bins, counts, linestyle='steps');
 Matplotlib提供了 plt.hist() 方法，该方法仅用一行代码就实现了上述功能：
plt.hist(x, bins, histtype='step')
为了计算区间， Matplotlib 将使用
np.histogram 函数，该函数的计算功能也和上面执行的计算类似
```
# 2.8　数组的排序
```python
一个简单的选择排序重复寻找列表中的最小值，并且不断交换直到列表是有序的。
可以在 Python 中仅用几行代码来实现：
In[1]: import numpy as np
       def selection_sort(x):
       for i in range(len(x)):
       swap = i + np.argmin(x[i:])
       (x[i], x[swap]) = (x[swap], x[i])
       return x
In[2]: x = np.array([2, 1, 4, 3, 5])
       selection_sort(x)
Out[2]: array([1, 2, 3, 4, 5])
```
# 2.8.1 NumPy中的快速排序： np.sort和np.argsort
```python
Python 有内置的 sort 和 sorted 函数可以对列表进行排序，默认情况下， np.sort 的排序算法是快速排序，其算法复杂度为 [N log N ]，另外也可以选择归并排序和堆排序。对于大多数应用场景，默认的快速排序已经足够高效了。
如果想在不修改原始输入数组的基础上返回一个排好序的数组，可以使用 np.sort：
In[5]: x = np.array([2, 1, 4, 3, 5])
       np.sort(x)
Out[5]: array([1, 2, 3, 4, 5])
如果希望用排好序的数组替代原始数组，可以使用数组的 sort 方法：
In[6]: x.sort()
       print(x)
[1 2 3 4 5]
另外一个相关的函数是 argsort，该函数返回的是原始数组排好序的索引值：
In[7]: x = np.array([2, 1, 4, 3, 5])
       i = np.argsort(x)
       print(i)
[1 0 3 2 4]
这些索引值可以被用于（通过花哨的索引）创建有序的数组：
In[8]: x[i]
Out[8]: array([1, 2, 3, 4, 5])
沿着行或列排序
NumPy 排序算法的一个有用的功能是通过 axis 参数，沿着多维数组的行或列进行排序，
例如：
In[9]: rand = np.random.RandomState(42)
       X = rand.randint(0, 10, (4, 6))
       print(X)
[[6 3 7 4 6 9]
[2 6 7 4 3 7]
[7 2 5 4 1 7]
[5 1 4 0 9 5]]
In[10]: # 对X的每一列排序
       np.sort(X, axis=0)
Out[10]: array([[2, 1, 4, 0, 1, 5],
[5, 2, 5, 4, 3, 7],
[6, 3, 7, 4, 6, 7],
[7, 6, 7, 4, 9, 9]])
In[11]: # 对X每一行排序
        np.sort(X, axis=1)
Out[11]: array([[3, 4, 6, 6, 7, 9],
[2, 3, 4, 6, 7, 7],
[1, 2, 4, 5, 7, 7],
[0, 1, 4, 5, 5, 9]])
这种处理方式是将行或列当作独立的数组，任何行或列的值之间的关系将会丢失！
```
## 2.8.2　部分排序： 分隔
```python
希望找到数组中第 K 小的值， NumPy 的np.partition 函数提供了该功能。 np.partition 函数的输入是数组和数字 K，输出结果是一个新数组，最左边是第 K 小的值，往右是任意顺序的其他值：
In[12]: x = np.array([7, 2, 3, 1, 6, 5, 4])
        np.partition(x, 3)
Out[12]: array([2, 1, 3, 4, 6, 5, 7])
请注意，结果数组中前三个值是数组中最小的三个值，剩下的位置是原始数组剩下的值。在这两个分隔区间中，元素都是任意排列的。与排序类似，也可以沿着多维数组任意的轴进行分隔：
In[13]: np.partition(X, 2, axis=1)
Out[13]: array([[3, 4, 6, 7, 6, 9],
                [2, 3, 4, 7, 6, 7],
                [1, 2, 4, 5, 7, 7],
                [0, 1, 4, 5, 9, 5]])
输出结果是一个数组，该数组每一行的前两个元素是该行最小的两个值，每行的其他值分布在剩下的位置。最后，正如 np.argsort 函数计算的是排序的索引值，也有一个 np.argpartition 函数计算的是分隔的索引值，我们将在下一节中举例介绍它。
```
## 2.8.3　示例： K个最近邻
```
利用 argsort 函数沿着多个轴快速找到集合中每个点的最近邻，将这些数据点放在一个10× 2 的数组中：
In[14]: X = rand.rand(10, 2)
为了对这些点有一个直观的印象，来画出它的散点图
In[15]: %matplotlib inline
        import matplotlib.pyplot as plt
        import seaborn; seaborn.set() # 设置画图风格
        plt.scatter(X[:, 0], X[:, 1], s=100);
现在来计算两两数据点对间的距离。两点间距离的平方等于每个维度的距离差的平方的和。利用 NumPy 的广播和聚合功能，可以用一行代码计算矩阵的平方距离：
In[16]: dist_sq = np.sum((X[:,np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis=-1)
In[17]: # 在坐标系中计算每对点的差值
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
differences.shape
Out[17]: (10, 10, 2)
In[18]: # 求出差值的平方
sq_differences = differences ** 2
sq_differences.shape
Out[18]: (10, 10, 2)
In[19]: # 将差值求和获得平方距离
dist_sq = sq_differences.sum(-1)
dist_sq.shape
Out[19]: (10, 10)
请再次确认以上步骤，应该看到该矩阵的对角线（也就是每个点到其自身的距离）的值
都是 0：
In[20]: dist_sq.diagonal()
Out[20]: array([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
结果确实是这样的！当我们有了这样一个转化为两点间的平方距离的矩阵后，就可以使用
np.argsort 函数沿着每行进行排序了。最左边的列给出的索引值就是最近邻：
In[21]: nearest = np.argsort(dist_sq, axis=1)
        print(nearest)
[[0 3 9 7 1 4 2 5 6 8]
[1 4 7 9 3 6 8 5 0 2]
[2 1 4 6 3 0 8 9 7 5]
[3 9 7 0 1 4 5 8 6 2]
[4 1 8 5 6 7 9 3 0 2]
[5 8 6 4 1 7 9 3 2 0]
[6 8 5 4 1 7 9 3 2 0]
[7 9 3 1 4 0 5 8 6 2]
[8 5 6 4 1 7 9 3 2 0]
[9 7 3 0 1 4 5 8 6 2]]
第一列是按 0~9 从小到大排列的。这是因为每个点的最近邻是其自身，所以结果也正如我们所想。
如果我们仅仅关心 k 个最近邻，那么唯一需要做的是分隔每一行，这样最小的 k + 1 的平方距离将排在最前面，其他更长的距离占据矩阵该行的其他位置。可以用 np.argpartition 函数实现：
In[22]: K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)
为了将邻节点网络可视化，我们将每个点和其最近的两个最近邻连接
In[23]: plt.scatter(X[:, 0], X[:, 1], s=100)
# 将每个点与它的两个最近邻连接
        K = 2
        for i in range(X.shape[0]):
        for j in nearest_partition[i, :K+1]:
# 画一条从X[i]到X[j]的线段
# 用zip方法实现：
plt.plot(*zip(X[j], X[i]), color='black')
```
# 2.9　结构化数据： NumPy的结构化数组
```python
NumPy 的结构化数组和记录数组，它们为复合的、异构的数据提供了非常有效的存储。NumPy 的结构化数组和记录数组，它们为复合的、异构的数据提供了非常有效的存储。
因为并没有任何信息告诉我们这三个数组是相关联的。如果可以用一种单一结构来存储所有的数据，那么看起来会更自然。 NumPy 可以用结构化数组实现这种存储，这些结构化数组是复合数据类型的。
通过指定复合数据类型，可以构造一个结构化数组：
In[4]: # 使用复合数据结构的结构化数组
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)
[('name', '<U10'), ('age', '<i4'), ('weight', '<f8')]
这里 U10 表示“长度不超过 10 的 Unicode 字符串”， i4 表示“4 字节（即 32 比特）整型”，f8 表示“8 字节（即 64 比特）浮点型”。
现在生成了一个空的数组容器，可以将列表数据放入数组中：
In[5]: data['name'] = name
       data['age'] = age
       data['weight'] = weight
print(data)
[('Alice', 25, 55.0) ('Bob', 45, 85.5) ('Cathy', 37, 68.0)
('Doug', 19, 61.5)]
现在生成了一个空的数组容器，可以将列表数据放入数组中：
In[5]: data['name'] = name
       data['age'] = age
       data['weight'] = weight
print(data)
[('Alice', 25, 55.0) ('Bob', 45, 85.5) ('Cathy', 37, 68.0)
('Doug', 19, 61.5)]
利用布尔掩码，还可以做一些更复杂的操作，如按照年龄进行筛选：
In[9]: # 获取年龄小于30岁的人的名字
       data[data['age'] < 30]['name']
Out[9]: array(['Alice', 'Doug'],
               dtype='<U10')
```
## 2.9.1　生成结构化数组
```python
结构化数组的数据类型有多种制定方式。此前我们看过了采用字典的方法：
In[10]: np.dtype({'names':('name', 'age', 'weight'),
'formats':('U10', 'i4', 'f8')})
Out[10]: dtype([('name', '<U10'), ('age', '<i4'), ('weight', '<f8')])
为了简明起见，数值数据类型可以用 Python 类型或 NumPy 的 dtype 类型指定：
In[11]: np.dtype({'names':('name', 'age', 'weight'),
'formats':((np.str_, 10), int, np.float32)})
Out[11]: dtype([('name', '<U10'), ('age', '<i8'), ('weight', '<f4')])
复合类型也可以是元组列表：
In[12]: np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')])
Out[12]: dtype([('name', 'S10'), ('age', '<i4'), ('weight', '<f8')])
如果类型的名称对你来说并不重要，那你可以仅仅用一个字符串来指定它。在该字符串中
数据类型用逗号分隔：
In[13]: np.dtype('S10,i4,f8')
Out[13]: dtype([('f0', 'S10'), ('f1', '<i4'), ('f2', '<f8')])
第一个（可选）字符是 < 或者 >，分别表示“低字节序”（little endian）和“高字节序”（bidendian），表示字节（bytes）类型的数据在内存中存放顺序的习惯用法。后一个字符指定的是数据的类型：字符、字节、整型、浮点型，等等（如表 2-4 所示）。最后一个字符表示该对象的字节大小。
```
## 2.9.2　更高级的复合类型
```
NumPy 中也可以定义更高级的复合数据类型。例如，你可以创建一种类型，其中每个元素
都包含一个数组或矩阵。我们会创建一个数据类型，该数据类型用 mat 组件包含一个 3× 3
的浮点矩阵：
In[14]: tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
         X = np.zeros(1, dtype=tp)
         print(X[0])
         print(X['mat'][0])
(0, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
[[ 0. 0. 0.]
[ 0. 0. 0.]
[ 0. 0. 0.]]
现在 X 数组的每个元素都包含一个 id 和一个 3× 3 的矩阵。
```
## 2.9.3　记录数组： 结构化数组的扭转
```python
NumPy 还提供了 np.recarray 类。域可以像属性一样获取，而不是像字典的键那样获取。前面的例子通过以下代
码获取年龄：
In[15]: data['age']
Out[15]: array([25, 45, 37, 19], dtype=int32)
如果将这些数据当作一个记录数组，我们可以用很少的按键来获取这个结果：
In[16]: data_rec = data.view(np.recarray)
data_rec.age
Out[16]: array([25, 45, 37, 19], dtype=int32)
记录数组的不好的地方在于，即使使用同样的语法，在获取域时也会有一些额外的开销，
如以下示例所示：
In[17]: %timeit data['age']
%timeit data_rec['age']
%timeit data_rec.age
1000000 loops, best of 3: 241 ns per loop
100000 loops, best of 3: 4.61 µs per loop
100000 loops, best of 3: 7.27 µs per loop
```

