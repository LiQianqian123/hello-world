# 2.4 聚合： 最小值、 最大值和其他值
```python
最常用的概括统计值可能是均值和标准差，这两个值能让你分别概括出数据集中的“经典”值，但是其他一些形式的聚合也是非常有用的（如求和、乘积、中位数、最小值和最大值、分位数，等等）。NumPy 有非常快速的内置聚合函数可用于数组
```
## 2.4.1 数组值求和
```python
计算一个数组中所有元素的和。 Python 本身可用内置的 sum 函数来实现：
In[1]: import numpy as np
In[2]: L = np.random.random(100)
       sum(L)
Out[2]: 55.61209116604941
它的语法和 NumPy 的 sum 函数非常相似
In[3]: np.sum(L)
Out[3]: 55.612091166049424
因为 NumPy 的 sum 函数在编译码中执行操作，所以 NumPy 的操作计算得更快一些：
In[4]: big_array = np.random.rand(1000000)
       %timeit sum(big_array)
       %timeit np.sum(big_array)
10 loops, best of 3: 104 ms per loop
1000 loops, best of 3: 442 µs per loop
np.sum 函数是知道数组的维度的
```
## 2.4.2 最小值和最大值
```python
In[5]: min(big_array), max(big_array)
Out[5]: (1.1717128136634614e-06, 0.9999976784968716)
NumPy 对应的函数也有类似的语法，并且也执行得更快：
In[6]: np.min(big_array), np.max(big_array)
Out[6]: (1.1717128136634614e-06, 0.9999976784968716)
In[7]: %timeit min(big_array)
       %timeit np.min(big_array)
10 loops, best of 3: 82.3 ms per loop
1000 loops, best of 3: 497 µs per loop
一种更简洁的语法形式是数组对象直接调用这些方法:
In[8]: print(big_array.min(), big_array.max(), big_array.sum())
1.17171281366e-06 0.999997678497 499911.628197
多维度聚合:
一种常用的聚合操作是沿着一行或一列聚合
In[9]: M = np.random.random((3, 4))
       print(M)
[[ 0.8967576 0.03783739 0.75952519 0.06682827]
[ 0.8354065 0.99196818 0.19544769 0.43447084]
[ 0.66859307 0.15038721 0.37911423 0.6687194]]
In[10]: M.sum()
Out[10]: 6.0850555667307118
聚合函数还有一个参数，用于指定沿着哪个轴的方向进行聚合。例如，可以通过指定axis=0 找到每一列的最小值：    #axis参数, axis=0  是列, axis=1  是行
In[11]: M.min(axis=0)
Out[11]: array([ 0.66859307, 0.03783739, 0.19544769, 0.06682827])
可以找到每一行的最大值：
In[12]: M.max(axis=1)
Out[12]: array([ 0.8967576 , 0.99196818, 0.6687194])
axis 关键字指定的是数组将会被折叠的维度，而不是将要返回的维度。因此指定 axis=0 意味着第一个轴将要被折叠——对于二维数组，这意味着每一列的值都将被聚合。
其他聚合函数: NumPy中可用的聚合函数
np.sum np.nansum 计算元素的和
np.prod np.nanprod 计算元素的积
np.mean np.nanmean 计算元素的平均值
np.std np.nanstd 计算元素的标准差
np.var np.nanvar 计算元素的方差
np.min np.nanmin 找出最小值
np.max np.nanmax 找出最大值
np.argmin np.nanargmin 找出最小值的索引
np.argmax np.nanargmax 找出最大值的索引
np.median np.nanmedian 计算元素的中位数
np.percentile np.nanpercentile 计算基于元素排序的统计值
np.any N/A 验证任何一个元素是否为真
np.all N/A 验证所有元素是否为真
```
## 2.4.3　示例： 美国总统的身高是多少
```python
In[13]: head -4 data/president_heights.csv
order,name,height(cm)
1,George Washington,189
2,John Adams,170
3,Thomas Jefferson,189
In[14]: import pandas as pd
data = pd.read_csv('data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)
[189 170 189 163 183 171 185 168 173 183 173 173 175 178 183 193 178 173
174 183 183 168 170 178 182 180 183 178 182 188 175 179 183 193 182 183
177 185 188 188 182 185]
In[15]: print("Mean height: ", heights.mean())
        print("Standard deviation:", heights.std())
        print("Minimum height: ", heights.min())
        print("Maximum height: ", heights.max())
Mean height: 179.738095238
Standard deviation: 6.93184344275
Minimum height: 163
Maximum height: 193
In[16]: print("25th percentile: ", np.percentile(heights, 25))
        print("Median: ", np.median(heights))
        print("75th percentile: ", np.percentile(heights, 75))
 25th percentile: 174.25
 Median: 182.0
 75th percentile: 183.0
进行一个快速的可视化，通过 Matplotlib用以下代码创建
In[17]: %matplotlib inline
        import matplotlib.pyplot as plt
        import seaborn; seaborn.set() # 设置绘图风格
In[18]: plt.hist(heights)
        plt.title('Height Distribution of US Presidents')
        plt.xlabel('height (cm)')
        plt.ylabel('number');
```
# 2.5　数组的计算： 广播
```python
前一节中介绍了 NumPy 如何通过通用函数的向量化操作来减少缓慢的 Python 循环，另外一种向量化操作的方法是利用 NumPy 的广播功能。广播可以简单理解为用于不同大小数组的二进制通用函数（加、减、乘等）的一组规则。
```
## 2.5.1　广播的介绍
```python
对于同样大小的数组，二进制操作是对相应元素逐个计算：
In[1]: import numpy as np
In[2]: a = np.array([0, 1, 2])
       b = np.array([5, 5, 5])
       a + b
Out[2]: array([5, 6, 7])
广播允许这些二进制操作可以用于不同大小的数组。可以简单地将一个标量（可以认为是一个零维的数组）和一个数组相加：
In[3]: a + 5
Out[3]: array([5, 6, 7])
可以认为这个操作是将数值 5 扩展或重复至数组 [5, 5, 5]，然后执行加法。观察以下将一个一维数组和一个二维数组相加的结果： 
In[4]: M = np.ones((3, 3))
       M
Out[4]: array([[ 1., 1., 1.],
               [ 1., 1., 1.],
               [ 1., 1., 1.]])
In[5]: M + a
Out[5]: array([[ 1., 2., 3.],
[ 1., 2., 3.],
[ 1., 2., 3.]])
对两个数组的同时广播
In[6]: a = np.arange(3)
       b = np.arange(3)[:, np.newaxis]
       print(a)
       print(b)
[0 1 2]
[[0]
[1]
[2]]
In[7]: a + b
Out[7]: array([[0, 1, 2],
               [1, 2, 3],
               [2, 3, 4]])
```
## 2.5.2　广播的规则
```python
• 规则 1：如果两个数组的维度数不相同，那么小维度数组的形状将会在最左边补 1。
• 规则 2：如果两个数组的形状在任何一个维度上都不匹配，那么数组的形状会沿着维度
  为 1 的维度扩展以匹配另外一个数组的形状。
• 规则 3：如果两个数组的形状在任何一个维度上都不匹配并且没有任何一个维度等于 1，
  那么会引发异常。
1. 广播示例1
将一个二维数组与一个一维数组相加：
In[8]: M = np.ones((2, 3))
       a = np.arange(3)
两个数组的形状如下：
M.shape = (2, 3)
a.shape = (3,)
根据规则 1，数组 a 的维度数更小，所以在其左边补 1：
M.shape->(2, 3)
a.shape->(1, 3)
根据规则 2，第一个维度不匹配，因此扩展这个维度以匹配数组：
M.shape->(2, 3)
a.shape->(2, 3)
现在两个数组的形状匹配了，可以看到它们的最终形状都为 (2, 3)：
In[9]: M + a
Out[9]: array([[ 1., 2., 3.],
               [ 1., 2., 3.]])
2. 广播示例2
来看两个数组均需要广播的示例：
In[10]: a = np.arange(3).reshape((3, 1))
b = np.arange(3)
首先写出两个数组的形状：
a.shape = (3, 1)
b.shape = (3,)
规则 1 告诉我们，需要用 1 将 b 的形状补全：
a.shape->(3, 1)
b.shape->(1, 3)
规则 2 告诉我们，需要更新这两个数组的维度来相互匹配：
a.shape->(3, 3)
b.shape->(3, 3)
因为结果匹配，所以这两个形状是兼容的，可以看到以下结果：
In[11]: a + b
Out[11]: array([[0, 1, 2],
                [1, 2, 3],
                [2, 3, 4]]
3. 广播示例3
现在来看一个两个数组不兼容的示例：
In[12]: M = np.ones((3, 2))
        a = np.arange(3)
和第一个示例相比，这里有个微小的不同之处：矩阵 M 是转置的。
M.shape = (3, 2)
a.shape = (3,)
同样，规则 1 告诉我们， a 数组的形状必须用 1 进行补全：
M.shape -> (3, 2)
a.shape -> (1, 3)
根据规则 2， a 数组的第一个维度进行扩展以匹配 M 的维度：
M.shape -> (3, 2)
a.shape -> (3, 3)
现在需要用到规则 3——最终的形状还是不匹配，因此这两个数组是不兼容的
这里可能发生的混淆在于：你可能想通过在 a 数组的右边补 1，而不是左边补 1，让 a 和 M 的维度变得兼容。但是这不被广播的规则所允许。
```
## 2.5.3　广播的实际应用
```python
1. 数组的归一化
通用函数让 NumPy 用户免于写很慢的 Python 循环。广播进一步扩展了这个功能，一个常见的例子就是数组数据的归一化。假设你有一个有 10 个观
察值的数组，每个观察值包含 3 个数值。按照惯例（详情请参见 5.2 节），我们将用一个10×3的数组存放该数据：
In[17]: X = np.random.random((10, 3))
可以计算每个特征的均值，计算方法是利用 mean 函数沿着第一个维度聚合：
In[18]: Xmean = X.mean(0)
        Xmean
Out[18]: array([ 0.53514715, 0.66567217, 0.44385899])
现在通过从 X 数组的元素中减去这个均值实现归一化（该操作是一个广播操作）：
In[19]: X_centered = X - Xmean
为了进一步核对我们的处理是否正确，可以查看归一化的数组的均值是否接近 0：
In[20]: X_centered.mean(0)
Out[20]: array([ 2.22044605e-17, -7.77156117e-17, -1.66533454e-17])
在机器精度范围内，该均值为 0。
2. 画一个二维函数
广播另外一个非常有用的地方在于，它能基于二维函数显示图像。我们希望定义一个函数z = f (x, y)，可以用广播沿着数值区间计算该函数：
In[21]: # x和y表示0~5区间50个步长的序列
        x = np.linspace(0, 5, 50)
        y = np.linspace(0, 5, 50)[:, np.newaxis]
        z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
将用 Matplotlib 来画出这个二维数组
In[22]: %matplotlib inline
        import matplotlib.pyplot as plt
In[23]: plt.imshow(z, origin='lower', extent=[0, 5, 0, 5],
                   cmap='viridis')
        plt.colorbar();
```
# 2.6　比较、 掩码和布尔逻辑
```python
介绍如何用布尔掩码来查看和操作 NumPy 数组中的值。当你想基于某些准则来抽取、修改、计数或对一个数组中的值进行其他操作时，掩码就可以派上用场了。例如你可能希望统计数组中有多少值大于某一个给定值，或者删除所有超出某些门限值的异常点。在 NumPy 中，布尔掩码通常是完成这类任务的最高效方式。
```
## 2.6.1　示例： 统计下雨天数
```python
假设你有一系列表示某城市一年内日降水量的数据，这里将用 Pandas（将在第 3 章详细介
绍）加载 2014 年西雅图市的日降水统计数据：
In[1]: import numpy as np
import pandas as pd
# 利用Pandas抽取降雨量，放入一个NumPy数组
rainfall = pd.read_csv('data/Seattle2014.csv')['PRCP'].values
inches = rainfall / 254 # 1/10mm -> inches
inches.shape
Out[1]: (365,)
这个数组包含 365 个值，给出了从 2014 年 1 月 1 日至 2014 年 12 月 31 日每天的降水量。这里降水量的单位是英寸。首先做一个快速的可视化，用 Matplotlib（将在第 4 章详细讨论该工具）生成下雨天数的直方图
In[2]: %matplotlib inline
       import matplotlib.pyplot as plt
       import seaborn; seaborn.set() # 设置绘图风格
In[3]: plt.hist(inches, 40);
但是这样做并没有很好地传递出我们希望看到的某些信息，例如一年中有多少天在下雨，这些下雨天的平均降水量是多少，有多少天的降水量超过了半英寸？
我们从 2.3节中了解到， NumPy 的通用函数可以用来替代循环，以快速实现数组的逐元素（elementwise）运算。同样，我们也可以用其他通用函数实现数组的逐元素比较，然后利用计算结果回答之前提出的问题。先将数据放在一边，来介绍一下 NumPy 中有哪些用掩码来快速回答这类问题的通用工具
NumPy 还实现了如 <（小于）和 >（大于）的逐元素比较的通用函数。这些比较运算的结果是一个布尔数据类型的数组。一共有 6 种标准的比较操作
In[4]: x = np.array([1, 2, 3, 4, 5])
In[5]: x < 3 # 小于
Out[5]: array([ True, True, False, False, False], dtype=bool)
In[6]: x > 3 # 大于
Out[6]: array([False, False, False, True, True], dtype=bool)
In[7]: x <= 3 # 小于等于
Out[7]: array([ True, True, True, False, False], dtype=bool)
In[8]: x >= 3 # 大于等于
Out[8]: array([False, False, True, True, True], dtype=bool)
In[9]: x != 3 # 不等于
Out[9]: array([ True, True, False, True, True], dtype=bool)
In[10]: x == 3 # 等于
Out[10]: array([False, False, True, False, False], dtype=bool)
比较运算操作在 NumPy 中也是借助通用函数来实现的。例如当你写
x < 3 时， NumPy 内部会使用 np.less(x, 3)。这些比较运算符和其对应的通用函数如下表所示
比较运算操作在 NumPy 中也是借助通用函数来实现的。例如当你写
x < 3 时， NumPy 内部会使用 np.less(x, 3)。这些比较运算符和其对应的通用函数如下表所示。
== np.equal
!= np.not_equal
< np.less         
<= np.less_equal
> np.greater
>= np.greater_equal
下
面是一个二维数组的示例：
In[12]: rng = np.random.RandomState(0)
         x = rng.randint(10, size=(3, 4))
         x
Out[12]: array([[5, 0, 3, 3],
                [7, 9, 3, 5],
                [2, 4, 7, 6]])
In[13]: x < 6
Out[13]: array([[ True, True, True, True],
               [False, False, True, True],
               [ True, True, False, False]], dtype=bool)
## 2.6.3　操作布尔数组
给定一个布尔数组，你可以实现很多有用的操作。首先打印出此前生成的二维数组 x：
In[14]: print(x)
[[5 0 3 3]
[7 9 3 5]
[2 4 7 6]]
1. 统计记录的个数
如果需要统计布尔数组中 True 记录的个数，可以使用 np.count_nonzero 函数：
In[15]: # 有多少值小于6？
np.count_nonzero(x < 6)
Out[15]: 8
另外一种实现方式是利用 np.sum。在这个例子中，
False 会被解释成 0， True 会被解释成 1：
In[16]: np.sum(x < 6)
Out[16]: 8
sum() 的好处是，和其他 NumPy 聚合函数一样，这个求和也可以沿着行或列进行：
In[17]: # 每行有多少值小于6？
np.sum(x < 6, axis=1)
Out[17]: array([4, 2, 2])
如要快速检查任意或者所有这些值是否为 True，可以用np.any() 或np.all()：
In[18]: # 有没有值大于8？
        np.any(x > 8)
Out[18]: True
In[19]: # 有没有值小于0？
        np.any(x < 0)
Out[19]: False
In[20]: # 是否所有值都小于10？
        np.all(x < 10)
Out[20]: True
In[21]: # 是否所有值都等于6？
        np.all(x == 6)
Out[21]: False
np.all() 和 np.any() 也可以用于沿着特定的坐标轴，例如：
In[22]: # 是否每行的所有值都小于8？
        np.all(x < 8, axis=1)
Out[22]: array([ True, False, True], dtype=bool)
Python 有内置的 sum()、 any() 和 all() 函数，这些函数在 NumPy 中有不同的语法版本。如果在多维数组上混用这两个版本，会导致失败或产生不可预知的错误结果。因此，确保在以上的示例中用的都是 np.sum()、 np.any()和 np.all() 函数。
2. 布尔运算符
如果我们想统计降水量小于 4 英寸且大于 2 英寸的天数该如何操作呢？这可以通过 Python 的逐位逻辑运算符（bitwise logic operator） &、 |、 ^ 和 ~ 来实现。同标准的算术运算符一样， NumPy用通用函数重载了这些逻辑运算符，这样可以实现数组的逐位运算（通常是布尔运算）
可以写如下的复合表达式：
In[23]: np.sum((inches > 0.5) & (inches < 1))
Out[23]: 29
可以看到，降水量在 0.5 英寸 ~1 英寸间的天数是 29 天
这些括号是非常重要的，因为有运算优先级规则。
利用 A AND B 和 NOT (A OR B) 的等价原理（你应该在基础逻辑课程中学习过），可以用另外一种形式实现同样的结果：
In[24]: np.sum(~( (inches <= 0.5) | (inches >= 1) ))
Out[24]: 29
将比较运算符和布尔运算符合并起来用在数组上，可以实现更多有效的逻辑运算操作
& np.bitwise_and
| np.bitwise_or
^ np.bitwise_xor
~ np.bitwise_not
利用这些工具，就可以回答那些关于天气数据的问题了。以下的示例是结合使用掩码和聚合实现的结果计算：
In[25]: print("Number days without rain: ", np.sum(inches == 0))
        print("Number days with rain: ", np.sum(inches != 0))
        print("Days with more than 0.5 inches:", np.sum(inches > 0.5))
        print("Rainy days with < 0.1 inches :", np.sum((inches > 0) & (inches < 0.2)))
Number days without rain: 215
Number days with rain: 150
Days with more than 0.5 inches: 37
Rainy days with < 0.1 inches : 75
## 2.6.4　将布尔数组作为掩码
一种更强大的模式是使用布尔数组作为掩码，通过该掩码选择数据的子数据集。以前面小节用过的 x 数组为例，假设我们希望抽取出数组中所有小于 5 的元素：
In[26]: x
Out[26]: array([[5, 0, 3, 3],
                [7, 9, 3, 5],
                [2, 4, 7, 6]])
利用比较运算符可以得到一个布尔数组：
In[27]: x < 5
Out[27]: array([[False, True, True, True],
[False, False, True, False],
[ True, True, False, False]], dtype=bool)
现在为了将这些值从数组中选出，可以进行简单的索引，即掩码操作：
In[28]: x[x < 5]
Out[28]: array([0, 3, 3, 3, 2, 4])
现在返回的是一个一维数组，它包含了所有满足条件的值。换句话说，所有的这些值是掩码数组对应位置为 True 的值。
```

