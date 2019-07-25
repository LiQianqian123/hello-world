# Pandas数据处理
Pandas 是在 NumPy 基础上建立的新程序库，提供了一种高效的 DataFrame
数据结构。DataFrame 本质上是一种带行标签和列标签、支持相同类型数据和缺失值的多
维数组。Pandas 不仅为带各种标签的数据提供了便利的存储界面，还实现了许多强大的操
作。建立在 NumPy 数组结构上的
Pandas，尤其是它的 Series 和 DataFrame 对象，为数据科学家们处理那些消耗大量时间的
“数据清理”（data munging）任务提供了捷径。
如果从底层视角观察 Pandas 对象，可以把它们看成增强版的 NumPy 结构化数组，行列都
不再只是简单的整数索引，还可以带上标签。
虽然
Pandas 在基本数据结构上实现了许多便利的工具、方法和功能，但是后面将要介绍的每
一个工具、方法和功能几乎都需要我们理解基本数据结构的内部细节。因此，在深入学习
Pandas 之前，先来看看 Pandas 的三个基本数据结构：Series、DataFrame 和 Index。
## pandas对象简介
### Pandas的Series对象
Pandas 的 Series 对象是一个带索引数据构成的一维数组。可以用一个数组创建 Series 对
象，如下所示：
```python
In[2]: data = pd.Series([0.25, 0.5, 0.75, 1.0])
 data 

Out[2]: 0 0.25
 1 0.50
 2 0.75
 3 1.00
 dtype: float64
  ```
从上面的结果中，你会发现 Series 对象将一组数据和一组索引绑定在一起，我们可以通过
values 属性和 index 属性获取数据。values 属性返回的结果与 NumPy 数组类似：
```python
In[3]: data.values
Out[3]: array([ 0.25, 0.5 , 0.75, 1. ])
 ```
index 属性返回的结果是一个类型为 pd.Index 的类数组对象，我们将在后面的内容里详细
介绍它：
```python
In[4]: data.index
Out[4]: RangeIndex(start=0, stop=4, step=1)
 ```
和 NumPy 数组一样，数据可以通过 Python 的中括号索引标签获取
```python
In[5]: data[1]
Out[5]: 0.5
In[6]: data[1:3]
Out[6]: 1 0.50
 2 0.75
 dtype: float64
  ```
但是我们将会看到，Pandas 的 Series 对象比它模仿的一维 NumPy 数组更加通用、灵活。
#### 1. Serise是通用的NumPy数组
到目前为止，我们可能觉得 Series 对象和一维 NumPy 数组基本可以等价交换，但两者
间的本质差异其实是索引：NumPy 数组通过隐式定义的整数索引获取数值，而 Pandas 的
Series 对象用一种显式定义的索引与数值关联。
显式索引的定义让 Series 对象拥有了更强的能力。例如，索引不再仅仅是整数，还可以是
任意想要的类型。如果需要，完全可以用字符串定义索引：
```python
In[7]: data = pd.Series([0.25, 0.5, 0.75, 1.0],
 index=['a', 'b', 'c', 'd'])
 data
Out[7]: a 0.25
 b 0.50
 c 0.75
 d 1.00
 dtype: float64
  ```
获取数值的方式与之前一样
```python
In[8]: data['b']
Out[8]: 0.5
 ```
也可以使用不连续或不按顺序的索引：
```python
In[9]: data = pd.Series([0.25, 0.5, 0.75, 1.0],
 index=[2, 5, 3, 7])
 data
Out[9]: 2 0.25
 5 0.50
 3 0.75
 7 1.00
 dtype: float64
In[10]: data[5]
Out[10]: 0.5
 ```
#### 2. Series是特殊的字典
你可以把 Pandas 的 Series 对象看成一种特殊的 Python 字典。字典是一种将任意键映射到
一组任意值的数据结构，而 Series 对象其实是一种将类型键映射到一组类型值的数据结
构。类型至关重要：就像 NumPy 数组背后特定类型的经过编译的代码使得它在某些操作
上比普通的 Python 列表更加高效一样，Pandas Series 的类型信息使得它在某些操作上比
Python 的字典更高效。
我们可以直接用 Python 的字典创建一个 Series 对象，让 Series 对象与字典的类比更
加清晰：
```python
In[11]: population_dict = {'California': 38332521,
 'Texas': 26448193,
 'New York': 19651127,
 'Florida': 19552860,
 'Illinois': 12882135}
 population = pd.Series(population_dict)
 population
Out[11]: California 38332521
 Florida 19552860
 Illinois 12882135
 New York 19651127
 Texas 26448193
 dtype: int64
  ```
用字典创建 Series 对象时，其索引默认按照顺序排列。典型的字典数值获取方式仍然
有效：
```python
In[12]: population['California']
Out[12]: 38332521
 ```
和字典不同，Series 对象还支持数组形式的操作，比如切片：
```python
In[13]: population['California':'Illinois']
Out[13]: California 38332521
 Florida 19552860
 Illinois 12882135
 dtype: int64
 ```
我们将在 3.3 节中介绍 Pandas 取值与切片的一些技巧。
#### 3. 创建Series对象
我们已经见过几种创建 Pandas 的 Series 对象的方法，都是像这样的形式：
```python
>>> pd.Series(data, index=index)
 ```
其中，index 是一个可选参数，data 参数支持多种数据类型。
例如，data 可以是列表或 NumPy 数组，这时 index 默认值为整数序列：
```python
In[14]: pd.Series([2, 4, 6])
Out[14]: 0 2
 1 4
 2 6
 dtype: int64
  ```
data 也可以是一个标量，创建 Series 对象时会重复填充到每个索引上：
```python
In[15]: pd.Series(5, index=[100, 200, 300])
Out[15]: 100 5
 200 5
 300 5
 dtype: int64
  ```
data 还可以是一个字典，index 默认是排序的字典键：
```python
In[16]: pd.Series({2:'a', 1:'b', 3:'c'})
Out[16]: 1 b
 2 a
 3 c
 dtype: object
  ```
每一种形式都可以通过显式指定索引筛选需要的结果：
```python
In[17]: pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])
Out[17]: 3 c
 2 a
 dtype: object
 ```
这里需要注意的是，Series 对象只会保留显式定义的键值对。
### Pandas的DataFrame对象
Pandas 的另一个基础数据结构是 DataFrame。和上一节介绍的 Series 对象一样，DataFrame
既可以作为一个通用型 NumPy 数组，也可以看作特殊的 Python 字典。下面来分别看看。
#### 1. DataFrame是通用的NumPy数组
如果将 Series 类比为带灵活索引的一维数组，那么 DataFrame 就可以看作是一种既有灵活
的行索引，又有灵活列名的二维数组。就像你可以把二维数组看成是有序排列的一维数组
一样，你也可以把 DataFrame 看成是有序排列的若干 Series 对象。这里的“排列”指的是
它们拥有共同的索引。
下面用上一节中美国五个州面积的数据创建一个新的 Series 来进行演示：
```python
In[18]:
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
 'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
area
Out[18]: California 423967
 Florida 170312
 Illinois 149995
 New York 141297
 Texas 695662
 dtype: int64
 ```
再结合之前创建的 population 的 Series 对象，用一个字典创建一个包含这些信息的二维
对象：
```python
In[19]: states = pd.DataFrame({'population': population,
 'area': area})
 states
Out[19]: area population
 California 423967 38332521
 Florida 170312 19552860
 Illinois 149995 12882135
 New York 141297 19651127
 Texas 695662 26448193
 ```
和 Series 对象一样，DataFrame 也有一个 index 属性可以获取索引标签：
```python
In[20]: states.index
Out[20]:
Index(['California', 'Florida', 'Illinois', 'New York', 'Texas'], dtype='object')
```
另外，DataFrame 还有一个 columns 属性，是存放列标签的 Index 对象：
```python
In[21]: states.columns
Out[21]: Index(['area', 'population'], dtype='object')
```
因此 DataFrame 可以看作一种通用的 NumPy 二维数组，它的行与列都可以通过索引获取。
#### 2. DataFrame是特殊的字典
与 Series 类似，我们也可以把 DataFrame 看成一种特殊的字典。字典是一个键映射一个
值，而 DataFrame 是一列映射一个 Series 的数据。例如，通过 'area' 的列属性可以返回
包含面积数据的 Series 对象：
```python
In[22]: states['area']
Out[22]: California 423967
 Florida 170312
 Illinois 149995
 New York 141297
 Texas 695662
 Name: area, dtype: int64
 ```
这里需要注意的是，在 NumPy 的二维数组里，data[0] 返回第一行；而在 DataFrame 中，
data['col0'] 返回第一列。因此，最好把 DataFrame 看成一种通用字典，而不是通用数
组，即使这两种看法在不同情况下都是有用的。
#### 3. 创建DataFrame对象
Pandas 的 DataFrame 对象可以通过许多方式创建，这里举几个常用的例子。
(1) 通过单个 Series 对象创建。DataFrame 是一组 Series 对象的集合，可以用单个 Series
创建一个单列的 DataFrame：
```python
In[23]: pd.DataFrame(population, columns=['population'])
Out[23]: population
 California 38332521
 Florida 19552860
 Illinois 12882135
 New York 19651127
 Texas 26448193
 ```
(2) 通过字典列表创建。任何元素是字典的列表都可以变成 DataFrame。用一个简单的列表
综合来创建一些数据：
```python
In[24]: data = [{'a': i, 'b': 2 * i}
 for i in range(3)]
 pd.DataFrame(data)
Out[24]: a b
 0 0 0
 1 1 2
 2 2 4
 ```
即使字典中有些键不存在，Pandas 也会用缺失值 NaN（不是数字，not a number）来表示：
```python
In[25]: pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])
Out[25]: a b c
 0 1.0 2 NaN
 1 NaN 3 4.0
 ```
(3) 通过 Series 对象字典创建。就像之前见过的那样，DataFrame 也可以用一个由 Series
对象构成的字典创建：
```python
In[26]: pd.DataFrame({'population': population,
 'area': area})
Out[26]: area population
 California 423967 38332521
 Florida 170312 19552860
 Illinois 149995 12882135
 New York 141297 19651127
 Texas 695662 26448193
 ```
(4) 通过 NumPy 二维数组创建。假如有一个二维数组，就可以创建一个可以指定行列索引
值的 DataFrame。如果不指定行列索引值，那么行列默认都是整数索引值：
```python
In[27]: pd.DataFrame(np.random.rand(3, 2),
 columns=['foo', 'bar'],
 index=['a', 'b', 'c'])
Out[27]: foo bar
 a 0.865257 0.213169
 b 0.442759 0.108267
 c 0.047110 0.905718
 ```
(5) 通过 NumPy 结构化数组创建。2.9 节曾介绍过结构化数组。由于 Pandas 的 DataFrame
与结构化数组十分相似，因此可以通过结构化数组创建 DataFrame：
```python
In[28]: A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
 A
Out[28]: array([(0, 0.0), (0, 0.0), (0, 0.0)],
 dtype=[('A', '<i8'), ('B', '<f8')])
In[29]: pd.DataFrame(A)
Out[29]: A B
 0 0 0.0
 1 0 0.0
 2 0 0.0
 
### Pandas的Index对象
Series 和 DataFrame 对象都使用便于引用和调整的显式索引。Pandas 的
Index 对象是一个很有趣的数据结构，可以将它看作是一个不可变数组或有序集合（实际
上是一个多集，因为 Index 对象可能会包含重复值）。这两种观点使得 Index 对象能呈现一
些有趣的功能。让我们用一个简单的整数列表来创建一个 Index 对象：
```python
In[30]: ind = pd.Index([2, 3, 5, 7, 11])
 ind
Out[30]: Int64Index([2, 3, 5, 7, 11], dtype='int64')

#### 1. 将Index看作不可变数组
Index 对象的许多操作都像数组。例如，可以通过标准 Python 的取值方法获取数值，也可
以通过切片获取数值：
```python
In[31]: ind[1]
Out[31]: 3
In[32]: ind[::2]
Out[32]: Int64Index([2, 5, 11], dtype='int64')
```
Index 对象还有许多与 NumPy 数组相似的属性：
```python
In[33]: print(ind.size, ind.shape, ind.ndim, ind.dtype)
5 (5,) 1 int64
```
Index 对象与 NumPy 数组之间的不同在于，Index 对象的索引是不可变的，也就是说不能
通过通常的方式进行调整：
```python
In[34]: ind[1] = 0
---------------------------------------------------------------------------
TypeError Traceback (most recent call last)
<ipython-input-34-40e631c82e8a> in <module>()
----> 1 ind[1] = 0
/Users/jakevdp/anaconda/lib/python3.5/site-packages/pandas/indexes/base.py ...
 1243
 1244 def __setitem__(self, key, value):
-> 1245 raise TypeError("Index does not support mutable operations")
 1246
 1247 def __getitem__(self, key):
TypeError: Index does not support mutable operations
```
Index 对象的不可变特征使得多个 DataFrame 和数组之间进行索引共享时更加安全，尤其是
可以避免因修改索引时粗心大意而导致的副作用。
#### 2. 将Index看作有序集合
Pandas 对象被设计用于实现许多操作，如连接（join）数据集，其中会涉及许多集合操作。
Index 对象遵循 Python 标准库的集合（set）数据结构的许多习惯用法，包括并集、交集、
差集等：
```python
In[35]: indA = pd.Index([1, 3, 5, 7, 9])
 indB = pd.Index([2, 3, 5, 7, 11])
In[36]: indA & indB # 交集
Out[36]: Int64Index([3, 5, 7], dtype='int64')
In[37]: indA | indB # 并集


Out[37]: Int64Index([1, 2, 3, 5, 7, 9, 11], dtype='int64')
In[38]: indA ^ indB # 异或
Out[38]: Int64Index([1, 2, 9, 11], dtype='int64')
```
这些操作还可以通过调用对象方法来实现，例如 indA.intersection(indB)。
## 　数据取值与选择
第 2 章具体介绍了获取、设置、调整 NumPy 数组数值的方法与工具，包括取值操作（如
arr[2, 1]）、切片操作（如 arr[:, 1:5]）、掩码操作（如 arr[arr > 0]）、花哨的索引操作
（如 arr[0, [1, 5]]），以及组合操作（如 arr[:, [1, 5]]）。下面介绍 Pandas 的 Series 和
DataFrame 对象相似的数据获取与调整操作。如果你用过 NumPy 操作模式，就会非常熟悉
Pandas 的操作模式，只是有几个细节需要注意一下。
我们将从简单的一维 Series 对象开始，然后再用比较复杂的二维 DataFrame 对象进行
演示。
###  Series数据选择方法
如前所述，Series 对象与一维 NumPy 数组和标准 Python 字典在许多方面都一样。只要牢
牢记住这两个类比，就可以帮助我们更好地理解 Series 对象的数据索引与选择模式。
#### 1. 将Series看作字典
和字典一样，Series 对象提供了键值对的映射：
```python
In[1]: import pandas as pd
 data = pd.Series([0.25, 0.5, 0.75, 1.0],
 index=['a', 'b', 'c', 'd'])
 data
Out[1]: a 0.25
 b 0.50
 c 0.75
 d 1.00
 dtype: float64
In[2]: data['b']
Out[2]: 0.5
```
我们还可以用 Python 字典的表达式和方法来检测键 / 索引和值：
```python
In[3]: 'a' in data
Out[3]: True
In[4]: data.keys()

Out[4]: Index(['a', 'b', 'c', 'd'], dtype='object')
In[5]: list(data.items())
Out[5]: [('a', 0.25), ('b', 0.5), ('c', 0.75), ('d', 1.0)]
```
Series 对象还可以用字典语法调整数据。就像你可以通过增加新的键扩展字典一样，你也
可以通过增加新的索引值扩展 Series：
```python
In[6]: data['e'] = 1.25
 data
Out[6]: a 0.25
 b 0.50
 c 0.75
 d 1.00
 e 1.25
 dtype: float64
 ```
Series 对象的可变性是一个非常方便的特性：Pandas 在底层已经为可能发生的内存布局和
数据复制自动决策，用户不需要担心这些问题。
#### 2. 将Series看作一维数组
Series 不仅有着和字典一样的接口，而且还具备和 NumPy 数组一样的数组数据选择功能，
包括索引、掩码、花哨的索引等操作，具体示例如下所示：
```python
In[7]: # 将显式索引作为切片
 data['a':'c']
Out[7]: a 0.25
 b 0.50
 c 0.75
 dtype: float64
In[8]: # 将隐式整数索引作为切片
 data[0:2]
Out[8]: a 0.25
 b 0.50
 dtype: float64
In[9]: # 掩码
 data[(data > 0.3) & (data < 0.8)]
Out[9]: b 0.50
 c 0.75
 dtype: float64
In[10]: # 花哨的索引
 data[['a', 'e']]
Out[10]: a 0.25
 e 1.25
 dtype: float64
```
在以上示例中，切片是绝大部分混乱之源。需要注意的是，当使用显式索引（即
data['a':'c']）作切片时，结果包含最后一个索引；而当使用隐式索引（即 data[0:2]）
作切片时，结果不包含最后一个索引。
#### 3. 索引器：loc、iloc和ix
这些切片和取值的习惯用法经常会造成混乱。例如，如果你的 Series 是显式整数索引，那
么 data[1] 这样的取值操作会使用显式索引，而 data[1:3] 这样的切片操作却会使用隐式
索引。
```python
In[11]: data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
 data
Out[11]: 1 a
 3 b
 5 c
 dtype: object
In[12]: # 取值操作是显式索引
 data[1]
Out[12]: 'a'
In[13]: # 切片操作是隐式索引
 data[1:3]
Out[13]: 3 b
 5 c
 dtype: object
 ```
由于整数索引很容易造成混淆，所以 Pandas 提供了一些索引器（indexer）属性来作为取值
的方法。它们不是 Series 对象的函数方法，而是暴露切片接口的属性。
第一种索引器是 loc 属性，表示取值和切片都是显式的：
```python
In[14]: data.loc[1]
Out[14]: 'a'
In[15]: data.loc[1:3]
Out[15]: 1 a
 3 b
 dtype: object
 ```
第二种是 iloc 属性，表示取值和切片都是 Python 形式的 1 隐式索引：
```python
In[16]: data.iloc[1]
Out[16]: 'b'
In[17]: data.iloc[1:3]


Out[17]: 3 b
 5 c
 dtype: object
