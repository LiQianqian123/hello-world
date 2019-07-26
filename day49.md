# 3.7　合并数据集： Concat与Append操作
```python
数据源进行合并既包括将两个不同的数据集
非常简单地拼接在一起，也包括用数据库那样的连接（join） 与合并（merge）操作处理有重叠字段的数据集。Series 与 DataFrame 都具备这类操作， Pandas 的函数与方法让数据合
并变得快速简单。
先来用 pd.concat 函数演示一个 Series 与 DataFrame 的简单合并操作.
首先导入 Pandas 和 NumPy：
In[1]: import pandas as pd
       import numpy as np
 定义一个能够创建 DataFrame 某种形式的函数，后面将会用到：
In[2]: def make_df(cols, ind):
          """一个简单的DataFrame"""
          data = {c: [str(c) + str(i) for i in ind]
          for c in cols}
          return pd.DataFrame(data, ind)
       # DataFrame示例
       make_df('ABC', range(3))
Out[2]: A B C
       0 A0 B0 C0
       1 A1 B1 C1
       2 A2 B2 C2
```
## 3.7.1　知识回顾： NumPy数组的合并
```python
合并Series与DataFrame与合并NumPy数组基本相同
In[4]: x = [1, 2, 3]
       y = [4, 5, 6]
       z = [7, 8, 9]
       np.concatenate([x, y, z])
Out[4]: array([1, 2, 3, 4, 5, 6, 7, 8, 9])
第一个参数是需要合并的数组列表或元组。还有一个 axis 参数可以设置合并的坐标轴方向：
In[5]: x = [[1, 2],
           [3, 4]]
np.concatenate([x, x], axis=1)
Out[5]: array([[1, 2, 1, 2],
               [3, 4, 3, 4]])
```
## 3.7.2　通过pd.concat实现简易合并
```python
pd.concat() 可以简单地合并一维的 Series 或 DataFrame 对象，与 np.concatenate() 合并
数组一样：
In[6]: ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
pd.concat([ser1, ser2])
Out[6]: 1 A
        2 B
        3 C
        4 D
        5 E
        6 F
dtype: object
它也可以用来合并高维数据，例如下面的 DataFrame：
In[7]: df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
print(df1); print(df2); print(pd.concat([df1, df2]))
df1 df2 pd.concat([df1, df2])
A B A B A B
1 A1 B1 3 A3 B3 1 A1 B1
2 A2 B2 4 A4 B4 2 A2 B2
3 A3 B3
4 A4 B4
默认情况下， DataFrame 的合并都是逐行进行的（默认设置是 axis=0）。与 np.concatenate()
一样， pd.concat 也可以设置合并坐标轴，例如下面的示例：
In[8]: df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
print(df3); print(df4); print(pd.concat([df3, df4], axis='col'))
df3 df4 pd.concat([df3, df4], axis='col')
A B C D A B C D
0 A0 B0 0 C0 D0 0 A0 B0 C0 D0
1 A1 B1 1 C1 D1 1 A1 B1 C1 D1
这里也可以使用 axis=1，效果是一样的。但是用 axis='col' 会更直观。
1. 索引重复
np.concatenate 与 pd.concat 最主要的差异之一就是 Pandas 在合并时会保留索引，即使索
引是重复的！例如下面的简单示例：
In[9]: x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index # 复制索引
print(x); print(y); print(pd.concat([x, y]))
x y pd.concat([x, y])
A B A B A B
0 A0 B0 0 A2 B2 0 A0 B0
1 A1 B1 1 A3 B3 1 A1 B1
0 A2 B2
1 A3 B3
你会发现结果中的索引是重复的。虽然 DataFrame 允许这么做，但结果并不是我们想要的。
pd.concat() 提供了一些解决这个问题的方法
2. 类似join的合并
前面介绍的简单示例都有一个共同特点，那就是合并的 DataFrame 都是同样的列名。而在
实际工作中，需要合并的数据往往带有不同的列名，而 pd.concat 提供了一些选项来解决
这类合并问题。看下面两个 DataFrame，它们的列名部分相同，却又不完全相同：
In[13]: df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
print(df5); print(df6); print(pd.concat([df5, df6])
df5 df6 pd.concat([df5, df6])
A B C B C D A B C D
1 A1 B1 C1 3 B3 C3 D3 1 A1 B1 C1 NaN
2 A2 B2 C2 4 B4 C4 D4 2 A2 B2 C2 NaN
3 NaN B3 C3 D3
4 NaN B4 C4 D4
3. append()方法
因为直接进行数组合并的需求非常普遍，所以 Series 和 DataFrame 对象都支持 append 方
法，让你通过最少的代码实现合并功能。例如，你可以使用 df1.append(df2)，效果与
pd.concat([df1, df2]) 一样：
In[16]: print(df1); print(df2); print(df1.append(df2))
df1 df2 df1.append(df2)
A B A B A B
1 A1 B1 3 A3 B3 1 A1 B1
2 A2 B2 4 A4 B4 2 A2 B2
3 A3 B3
4 A4 B4
```
# 3.8　合并数据集： 合并与连接
```python
Pandas 的基本特性之一就是高性能的内存式数据连接（join）与合并（merge）操作。如果你有使用数据库的经验，那么对这类操作一定很熟悉。 Pandas 的主接口是 pd.merge 函数，
下面让我们通过一些示例来介绍它的用法。
```
## 3.8.1　关系代数
```python
pd.merge() 实现的功能基于关系代数（relational algebra）的一部分。关系代数是处理关
系型数据的通用理论，绝大部分数据库的可用操作都以此为理论基础。
Pandas 在 pd.merge() 函数与 Series 和 DataFrame 的 join() 方法里实现了这些基本操作规
则。下面来看看如何用这些简单的规则连接不同数据源的数据。
```
## 3.8.2　数据连接的类型
```python
pd.merge() 函数实现了三种数据连接的类型： 一对一、 多对一和多对多。这三种数据连接类型都通过 pd.merge() 接口进行调用，根据不同的数据连接需求进行不同的操作。下面将
通过一些示例来演示这三种类型，并进一步介绍更多的细节。
1. 一对一连接
一对一连接可能是最简单的数据合并类型了，与 3.7 节介绍的按列合并十分相似。如下面
示例所示，有两个包含同一所公司员工不同信息的 DataFrame：
In[2]:
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
'hire_date': [2004, 2008, 2012, 2014]})
print(df1); print(df2)
df1 df2
employee group employee hire_date
0 Bob Accounting 0 Lisa 2004
1 Jake Engineering 1 Bob 2008
2 Lisa Engineering 2 Jake 2012
3 Sue HR 3 Sue 2014
若想将这两个 DataFrame 合并成一个 DataFrame，可以用 pd.merge() 函数实现：
In[3]: df3 = pd.merge(df1, df2)
df3
Out[3]: employee group hire_date
0 Bob Accounting 2008
1 Jake Engineering 2012
2 Lisa Engineering 2004
3 Sue HR 2014
2. 多对一连接
多对一连接是指，在需要连接的两个列中，有一列的值有重复。通过多对一连接获得的结果 DataFrame 将会保留重复值。请看下面的例子：
In[4]: df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
'supervisor': ['Carly', 'Guido', 'Steve']})
print(df3); print(df4); print(pd.merge(df3, df4))
df3 df4
employee group hire_date group supervisor
0 Bob Accounting 2008 0 Accounting Carly
1 Jake Engineering 2012 1 Engineering Guido
2 Lisa Engineering 2004 2 HR Steve
3 Sue HR 2014
pd.merge(df3, df4)
employee group hire_date supervisor
0 Bob Accounting 2008 Carly
1 Jake Engineering 2012 Guido
2 Lisa Engineering 2004 Guido
3 Sue HR 2014 Steve
3. 多对多连接
多对多连接是个有点儿复杂的概念，不过也可以理解。如果左右两个输入的共同列都包含
重复值，那么合并的结果就是一种多对多连接。用一个例子来演示可能更容易理解。来看
下面的例子，里面有一个 DataFrame 显示不同岗位人员的一种或多种能力。
通过多对多链接，就可以得知每位员工所具备的能力：
In[5]: df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
'Engineering', 'Engineering', 'HR', 'HR'],
'skills': ['math', 'spreadsheets', 'coding', 'linux',
'spreadsheets', 'organization']})
print(df1); print(df5); print(pd.merge(df1, df5))
df1 df5
employee group group skills
0 Bob Accounting 0 Accounting math
1 Jake Engineering 1 Accounting spreadsheets
2 Lisa Engineering 2 Engineering coding
3 Sue HR 3 Engineering linux
4 HR spreadsheets
5 HR organization
pd.merge(df1, df5)
employee group skills
0 Bob Accounting math
1 Bob Accounting spreadsheets
2 Jake Engineering coding
3 Jake Engineering linux
4 Lisa Engineering coding
5 Lisa Engineering linux
6 Sue HR spreadsheets
7 Sue HR organization
```
## 3.8.3　设置数据合并的键
```python
1. 参数on的用法
最简单的方法就是直接将参数 on 设置为一个列名字符串或者一个包含多列名称的列表：
In[6]: print(df1); print(df2); print(pd.merge(df1, df2, on='employee'))
df1 df2
employee group employee hire_date
0 Bob Accounting 0 Lisa 2004
1 Jake Engineering 1 Bob 2008
2 Lisa Engineering 2 Jake 2012
3 Sue HR 3 Sue 2014
pd.merge(df1, df2, on='employee')
employee group hire_date
0 Bob Accounting 2008
1 Jake Engineering 2012
2 Lisa Engineering 2004
3 Sue HR 2014
这个参数只能在两个 DataFrame 有共同列名的时候才可以使用。
2. left_on与right_on参数
有时你也需要合并两个列名不同的数据集，例如前面的员工信息表中有一个字段不是
“employee”而是“name”。在这种情况下，就可以用 left_on 和 right_on 参数来指定
列名：
In[7]:
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'salary': [70000, 80000, 120000, 90000]})
print(df1); print(df3);
print(pd.merge(df1, df3, left_on="employee", right_on="name"))
df1 df3
employee group name salary
0 Bob Accounting 0 Bob 70000
1 Jake Engineering 1 Jake 80000
2 Lisa Engineering 2 Lisa 120000
3 Sue HR 3 Sue 90000
pd.merge(df1, df3, left_on="employee", right_on="name")
employee group name salary
0 Bob Accounting Bob 70000
1 Jake Engineering Jake 80000
2 Lisa Engineering Lisa 120000
3 Sue HR Sue 90000
获取的结果中会有一个多余的列，可以通过 DataFrame 的 drop() 方法将这列去掉：
In[8]:
pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1)
Out[8]: employee group salary
0 Bob Accounting 70000
1 Jake Engineering 80000
2 Lisa Engineering 120000
3 Sue HR 90000
你可以通过设置 pd.merge() 中的 left_index 和 / 或 right_index 参数将索引设置为键来实
现合并：
In[10]:
print(df1a); print(df2a);
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
df1a df2a
group hire_date
employee employee
Bob Accounting Lisa 2004
Jake Engineering Bob 2008
Lisa Engineering Jake 2012
Sue HR Sue 2014
pd.merge(df1a, df2a, left_index=True, right_index=True)
group hire_date
employee
Lisa Engineering 2004
Bob Accounting 2008
Jake Engineering 2012
Sue HR 2014
```
## 3.8.4　设置数据连接的集合操作规则
```pyhton
通过前面的示例，我们总结出数据连接的一个重要条件：集合操作规则。
In[13]: df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
'food': ['fish', 'beans', 'bread']},
columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
'drink': ['wine', 'beer']},
columns=['name', 'drink'])
print(df6); print(df7); print(pd.merge(df6, df7))
df6 df7 pd.merge(df6, df7)
name food name drink name food drink
0 Peter fish 0 Mary wine 0 Mary bread wine
1 Paul beans 1 Joseph beer
2 Mary bread
我们合并两个数据集，在“name”列中只有一个共同的值： Mary。默认情况下，结果中只
会包含两个输入集合的交集，这种连接方式被称为内连接（inner join）。我们可以用 how 参
数设置连接方式，默认值为 'inner'：
In[14]: pd.merge(df6, df7, how='inner')
Out[14]: name food drink
0 Mary bread wine
```
## 3.8.5　重复列名： suffixes参数
```python
最后，你可能会遇到两个输入 DataFrame 有重名列的情况。来看看下面的例子：
In[17]: df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
'rank': [3, 1, 4, 2]})
print(df8); print(df9); print(pd.merge(df8, df9, on="name"))
df8 df9 pd.merge(df8, df9, on="name")
name rank name rank name rank_x rank_y
0 Bob 1 0 Bob 3 0 Bob 1 3
1 Jake 2 1 Jake 1 1 Jake 2 1
2 Lisa 3 2 Lisa 4 2 Lisa 3 4
3 Sue 4 3 Sue 2 3 Sue 4 2
```
# 3.9　累计与分组
```python
在对较大的数据进行分析时，一项基本的工作就是有效的数据累计（summarization）：计算累计（aggregation）指标，如 sum()、 mean()、 median()、 min() 和 max()，其中每一个指
标都呈现了大数据集的特征。
行星数据可以直接通过 Seaborn 下载：
In[2]: import seaborn as sns
planets = sns.load_dataset('planets')
planets.shape
Out[2]: (1035, 6)
In[3]: planets.head()
Out[3]: method number orbital_period mass distance year
0 Radial Velocity 1 269.300 7.10 77.40 2006
1 Radial Velocity 1 874.774 2.21 56.95 2008
2 Radial Velocity 1 763.000 2.60 19.84 2011
3 Radial Velocity 1 326.030 19.40 110.62 2007
4 Radial Velocity 1 516.220 10.50 119.47 2009
数据中包含了截至 2014 年已被发现的一千多颗外行星的资料
```
## 3.9.2 Pandas的简单累计功能
```python
之前我们介绍过 NumPy 数组的一些数据累计指标（详情请参见 2.4 节）。与一维 NumPy 数
组相同， Pandas 的 Series 的累计函数也会返回一个统计值：
In[4]: rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
ser
Out[4]: 0 0.374540
1 0.950714
2 0.731994
3 0.598658
4 0.156019
dtype: float64
In[5]: ser.sum()
Out[5]: 2.8119254917081569
In[6]: ser.mean()
Out[6]: 0.56238509834163142
DataFrame 的累计函数默认对每列进行统计：
In[7]: df = pd.DataFrame({'A': rng.rand(5),
'B': rng.rand(5)})
df
Out[7]: A B
0 0.155995 0.020584
1 0.058084 0.969910
2 0.866176 0.832443
3 0.601115 0.212339
4 0.708073 0.181825
In[8]: df.mean()
Out[8]: A 0.477888
B 0.443420
dtype: float64
设置 axis 参数，你就可以对每一行进行统计了：
In[9]: df.mean(axis='columns')
Out[9]: 0 0.088290
1 0.513997
2 0.849309
3 0.406727
4 0.444949
dtype: float64
```
## 3.9.3 GroupBy： 分割、 应用和组合
```pyhton
简单的累计方法可以让我们对数据集有一个笼统的认识，但是我们经常还需要对某些标签或索引的局部进行累计分析，这时就需要用到 groupby 了。
1. 分割、 应用和组合
一个经典分割 - 应用 - 组合操作示例如图 3-1 所示，其中“apply”的是一个求和函数。
• 分割步骤将 DataFrame 按照指定的键分割成若干组。
• 应用步骤对每个组应用函数，通常是累计、转换或过滤函数。
• 组合步骤将每一组的结果合并成一个输出数组
GroupBy（经常）只需要一行代码，
就可以计算每组的和、均值、计数、最小值以及其他累计值。 GroupBy 的用处就是将这些
步骤进行抽象：用户不需要知道在底层如何计算，只要把操作看成一个整体就够了。
我们可以用 DataFrame 的 groupby() 方法进行绝大多数常见的分割 - 应用 - 组合操作，将
需要分组的列名传进去即可：
In[12]: df.groupby('key')
Out[12]: <pandas.core.groupby.DataFrameGroupBy object at 0x117272160>
2. GroupBy对象
GroupBy 对象是一种非常灵活的抽象类型。在大多数场景中，你可以将它看成是 DataFrame
的集合，在底层解决所有难题。让我们用行星数据来做一些演示。
GroupBy 中最重要的操作可能就是 aggregate、 filter、 transform 和 apply（累计、过滤、转换、应用）了，后文将详细介绍这些内容，现在先来介绍一些 GroupBy 的基本操作方法。
(1) 按列取值。 GroupBy 对象与 DataFrame 一样，也支持按列取值，并返回一个修改过的
GroupBy 对象，例如：
In[14]: planets.groupby('method')
Out[14]: <pandas.core.groupby.DataFrameGroupBy object at 0x1172727b8>
In[15]: planets.groupby('method')['orbital_period']
Out[15]: <pandas.core.groupby.SeriesGroupBy object at 0x117272da0>
(2) 按组迭代。 GroupBy 对象支持直接按组进行迭代，返回的每一组都是 Series 或 DataFrame：
In[17]: for (method, group) in planets.groupby('method'):
print("{0:30s} shape={1}".format(method, group.shape))
Astrometry shape=(2, 6)
Eclipse Timing Variations shape=(9, 6)
Imaging shape=(38, 6)
Microlensing shape=(23, 6)
Orbital Brightness Modulation shape=(3, 6)
Pulsar Timing shape=(5, 6)
Pulsation Timing Variations shape=(1, 6)
Radial Velocity shape=(553, 6)
Transit shape=(397, 6)
Transit Timing Variations shape=(4, 6)
尽管通常还是使用内置的 apply 功能速度更快，但这种方式在手动处理某些问题时非常
有用，后面会详细介绍。
(3) 调用方法。 借助 Python 类的魔力（@classmethod），可以让任何不由 GroupBy 对象直接
实现的方法直接应用到每一组，无论是 DataFrame 还是 Series 对象都同样适用。例如，
你可以用 DataFrame 的 describe() 方法进行累计，对每一组数据进行描述性统计：
In[18]: planets.groupby('method')['year'].describe().unstack()
Out[18]:
count mean std min 25% \\
method
Astrometry 2.0 2011.500000 2.121320 2010.0 2010.75
Eclipse Timing Variations 9.0 2010.000000 1.414214 2008.0 2009.00
Imaging 38.0 2009.131579 2.781901 2004.0 2008.00
Microlensing 23.0 2009.782609 2.859697 2004.0 2008.00
Orbital Brightness Modulation 3.0 2011.666667 1.154701 2011.0 2011.00
Pulsar Timing 5.0 1998.400000 8.384510 1992.0 1992.00
Pulsation Timing Variations 1.0 2007.000000 NaN 2007.0 2007.00
Radial Velocity 553.0 2007.518987 4.249052 1989.0 2005.00
Transit 397.0 2011.236776 2.077867 2002.0 2010.00
Transit Timing Variations 4.0 2012.500000 1.290994 2011.0 2011.75
50% 75% max
method
Astrometry 2011.5 2012.25 2013.0
Eclipse Timing Variations 2010.0 2011.00 2012.0
Imaging 2009.0 2011.00 2013.0
Microlensing 2010.0 2012.00 2013.0
Orbital Brightness Modulation 2011.0 2012.00 2013.0
Pulsar Timing 1994.0 2003.00 2011.0
Pulsation Timing Variations 2007.0 2007.00 2007.0
Radial Velocity 2009.0 2011.00 2014.0
Transit 2012.0 2013.00 2014.0
Transit Timing Variations 2012.5 2013.25 2014.0
3. 累计、 过滤、 转换和应用
虽然前面的章节只重点介绍了组合操作，但是还有许多操作没有介绍，尤其是 GroupBy 对
象的 aggregate()、 filter()、 transform() 和 apply() 方法，在数据组合之前实现了大量
高效的操作。
为了方便后面内容的演示，使用下面这个 DataFrame：
In[19]: rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
'data1': range(6),
'data2': rng.randint(0, 10, 6)},
columns = ['key', 'data1', 'data2'])
df
Out[19]: key data1 data2
0 A 0 5
1 B 1 0
2 C 2 3
3 A 3 3
4 B 4 7
5 C 5 9
4. 设置分割的键
前面的简单例子一直在用列名分割 DataFrame。这只是众多分组操作中的一种，下面将继
续介绍更多的分组方法。
(1) 将列表、 数组、 Series 或索引作为分组键。 分组键可以是长度与 DataFrame 匹配的任意
Series 或列表，例如：
In[25]: L = [0, 1, 0, 1, 2, 0]
print(df); print(df.groupby(L).sum())
df df.groupby(L).sum()
key data1 data2 data1 data2
0 A 0 5 0 7 17
1 B 1 0 1 4 3
2 C 2 3 2 4 7
3 A 3 3
4 B 4 7
5 C 5 9
因此，还有一种比前面直接用列名更啰嗦的表示方法 df.groupby('key')：
In[26]: print(df); print(df.groupby(df['key']).sum())
df df.groupby(df['key']).sum()
key data1 data2 data1 data2
0 A 0 5 A 3 8
1 B 1 0 B 5 7
2 C 2 3 C 7 12
3 A 3 3
4 B 4 7
5 C 5 9
(2) 用字典或 Series 将索引映射到分组名称。另一种方法是提供一个字典，将索引映射到分组键
In[27]: df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2); print(df2.groupby(mapping).sum())
df2 df2.groupby(mapping).sum()
key data1 data2 data1 data2
A 0 5 consonant 12 19
B 1 0 vowel 3 8
C 2 3
A 3 3
B 4 7
C 5 9
(3) 任意 Python 函数。与前面的字典映射类似，你可以将任意 Python 函数传入 groupby，
函数映射到索引，然后新的分组输出：
In[28]: print(df2); print(df2.groupby(str.lower).mean())
df2 df2.groupby(str.lower).mean()
key data1 data2 data1 data2
A 0 5 a 1.5 4.0
B 1 0 b 2.5 3.5
C 2 3 c 3.5 6.0
A 3 3
B 4 7
C 5 9
(4) 多个有效键构成的列表。此外，任意之前有效的键都可以组合起来进行分组，从而返回
一个多级索引的分组结果：
In[29]: df2.groupby([str.lower, mapping]).mean()
Out[29]: data1 data2
a vowel 1.5 4.0
b consonant 2.5 3.5
c consonant 3.5 6.0
5. 分组案例
通过下例中的几行 Python 代码，我们就可以运用上述知识，获取不同方法和不同年份发现
的行星数量：
In[30]: decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)
Out[30]: decade 1980s 1990s 2000s 2010s
method
Astrometry 0.0 0.0 0.0 2.0
Eclipse Timing Variations 0.0 0.0 5.0 10.0
Imaging 0.0 0.0 29.0 21.0
Microlensing 0.0 0.0 12.0 15.0
Orbital Brightness Modulation 0.0 0.0 0.0 5.0
Pulsar Timing 0.0 9.0 1.0 1.0
Pulsation Timing Variations 0.0 0.0 1.0 0.0
Radial Velocity 1.0 52.0 475.0 424.0
Transit 0.0 0.0 64.0 712.0
Transit Timing Variations 0.0 0.0 0.0 9.0
```

