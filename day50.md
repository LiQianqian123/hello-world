# 3.10　数据透视表  
```python
数据透视表（pivottable）是一种类似的操作方法，常见于 Excel 与类似的表格应用中。数据透视表将每一列数据作为输入，输出将数据不断细分成多个维度累计信息的二维数据表。人们有时容易弄混数据透视表与 GroupBy，但我觉得数据透视表更像是一种多维的 GroupBy 累计操作。也就是说，虽然你也可以分割 - 应用 - 组合，但是分割与组合不是发生在一维索引上，而是在二维网格上（行列同时分组）。  
```  
## 3.10.1　演示数据透视表  
```python
这一节的示例将采用泰坦尼克号的乘客信息数据库来演示，可以在 Seaborn 程序库（详情
请参见 4.16 节）获取：
In[1]: import numpy as np
       import pandas as pd
       import seaborn as sns
       titanic = sns.load_dataset('titanic')
In[2]: titanic.head()
Out[2]:
survived pclass sex age sibsp parch fare embarked class \\
      0 0 3 male 22.0 1 0 7.2500 S Third
      1 1 1 female 38.0 1 0 71.2833 C First
      2 1 3 female 26.0 0 0 7.9250 S Third
      3 1 1 female 35.0 1 0 53.1000 S First
      4 0 3 male 35.0 0 0 8.0500 S Third
who adult_male deck embark_town alive alone
 0 man True NaN Southampton no False
 1 woman False C Cherbourg yes False
 2 woman False NaN Southampton yes True
 3 woman False C Southampton yes False
 4 man True NaN Southampton no True
这份数据包含了惨遭厄运的每位乘客的大量信息，包括性别（gender）、年龄（age）、船舱等级（class）和船票价格（fare paid）等。  
```
## 3.10.2　手工制作数据透视表
```python
同时观察不同性别与船舱等级的生还情
况。根据 GroupBy 的操作流程，我们也许能够实现想要的结果：将船舱等级（'class'）与
性别（'sex'） 分组，然后选择生还状态（'survived'）列， 应用均值（'mean'）累计函
数，再将各组结果组合，最后通过行索引转列索引操作将最里层的行索引转换成列索引，
形成二维数组。代码如下所示：
In[4]: titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
Out[4]: class First Second Third
        sex
        female 0.968085 0.921053 0.500000
          male 0.368852 0.157407 0.135447
虽然这样就可以更清晰地观察乘客性别、船舱等级对其是否生还的影响，但是代码看上去有点复杂。尽管这个管道命令的每一步都是前面介绍过的，但是要理解这个长长的语句可
不是那么容易的事。由于二维的 GroupBy 应用场景非常普遍，因此 Pandas 提供了一个快捷方式 pivot_table 来快速解决多维的累计分析任务。
3.10.3　数据透视表语法
用 DataFrame 的 pivot_table 实现的效果等同于上一节的管道命令的代码：
In[5]: titanic.pivot_table('survived', index='sex', columns='class')
Out[5]: class First Second Third
        sex
        female 0.968085 0.921053 0.500000
          male 0.368852 0.157407 0.135447
 与 GroupBy 方法相比，这行代码可读性更强，而且取得的结果也一样。
 1. 多级数据透视表
与 GroupBy 类似，数据透视表中的分组也可以通过各种参数指定多个等级。例如，我们152 ｜ 第 3 章
可能想把年龄（'age'）也加进去作为第三个维度，这就可以通过 pd.cut 函数将年龄进
行分段：
In[6]: age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')
Out[6]: class First Second Third
sex age
female (0, 18] 0.909091 1.000000 0.511628
(18, 80] 0.972973 0.900000 0.423729
male (0, 18] 0.800000 0.600000 0.215686
(18, 80] 0.375000 0.071429 0.133663
对某一列也可以使用同样的策略——让我们用 pd.qcut 将船票价格按照计数项等分为两份，
加入数据透视表看看：
In[7]: fare = pd.qcut(titanic['fare'], 2)
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])
Out[7]:
fare [0, 14.454]
class First Second Third \\
sex age
female (0, 18] NaN 1.000000 0.714286
(18, 80] NaN 0.880000 0.444444
male (0, 18] NaN 0.000000 0.260870
(18, 80] 0.0 0.098039 0.125000
fare (14.454, 512.329]
class First Second Third
sex age
female (0, 18] 0.909091 1.000000 0.318182
(18, 80] 0.972973 0.914286 0.391304
male (0, 18] 0.800000 0.818182 0.178571
(18, 80] 0.391304 0.030303 0.192308
结果是一个带层级索引（详情请参见 3.6 节）的四维累计数据表，通过网格显示不同数值
之间的相关性。
2. 其他数据透视表选项
DataFrame 的 pivot_table 方法的完整签名如下所示：
# Pandas 0.18版的函数签名
DataFrame.pivot_table(data, values=None, index=None, columns=None,
                      aggfunc='mean', fill_value=None, margins=False,
                      dropna=True, margins_name='All')
我们已经介绍过前面三个参数了，现在来看看其他参数。 fill_value 和 dropna 这两个参数用于处理缺失值，用法很简单，我们将在后面的示例中演示其用法。
aggfunc 参数用于设置累计函数类型，默认值是均值（mean）。与 GroupBy 的用法一样，累计函数可以用一些常见的字符串（'sum'、 'mean'、 'count'、 'min'、 'max' 等）表示，也可以用标准的累计函数（np.sum()、 min()、 sum() 等）表示。
同的列指定不同的累计函数：
In[8]: titanic.pivot_table(index='sex', columns='class',
       aggfunc={'survived':sum, 'fare':'mean'})
Out[8]: fare survived
        class First Second Third First Second Third
        sex
        female 106.125798 21.970121 16.118810 91.0 70.0 72.0
        male 67.226127 19.741782 12.661633 45.0 17.0 47.0
需要注意的是，这里忽略了一个参数 values。当我们为 aggfunc 指定映射关系的时候，待透视的数值就已经确定了。当需要计算每一组的总数时，可以通过 margins参数来设置：
In[9]: titanic.pivot_table('survived', index='sex', columns='class', margins=True)
Out[9]: class First Second Third All
        sex
        female 0.968085 0.921053 0.500000 0.742038
        male 0.368852 0.157407 0.135447 0.188908
        All 0.629630 0.472826 0.242363 0.383838
这样就可以自动获取不同性别下船舱等级与生还率的相关信息、不同船舱等级下性别与生还率的相关信息，以及全部乘客的生还率为 38%。 margin 的标签可以通过 margins_name 参数进行自定义，默认值是 "All"。
```
# 3.11　向量化字符串操作
```pyhton
使用 Python 的一个优势就是字符串处理起来比较容易。在此基础上创建的 Pandas 同样提供了一系列向量化字符串操作（vectorized string operation），它们都是在处理（清洗）现实工作中的数据时不可或缺的功能。
```
# 3.11.1 Pandas字符串操作简介
```python
前面的章节已经介绍过如何用 NumPy 和 Pandas 进行一般的运算操作，因此我们也能简便快速地对多个数组元素执行同样的操作，例如：
In[1]: import numpy as np
       x = np.array([2, 3, 5, 7, 11, 13])
       x * 2
Out[1]: array([ 4, 6, 10, 14, 22, 26])
向量化操作简化了纯数值的数组操作语法——我们不需要再担心数组的长度或维度，只需要关心需要的操作。
In[2]: data = ['peter', 'Paul', 'MARY', 'gUIDO']
       [s.capitalize() for s in data]
Out[2]: ['Peter', 'Paul', 'Mary', 'Guido']
虽然这么做对于某些数据可能是有效的，但是假如数据中出现了缺失值，那么这样做就会
引起异常，例如：
In[3]: data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
[s.capitalize() for s in data]
---------------------------------------------------------------------------
---------------------------------------------------------------------------
AttributeError Traceback (most recent call last)
<ipython-input-3-fc1d891ab539> in <module>()
1 data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
----> 2 [s.capitalize() for s in data]
<ipython-input-3-fc1d891ab539> in <listcomp>(.0)
1 data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
----> 2 [s.capitalize() for s in data]
AttributeError: 'NoneType' object has no attribute 'capitalize'
---------------------------------------------------------------------------
Pandas 为包含字符串的 Series 和 Index 对象提供的 str 属性堪称两全其美的方法，它既可以满足向量化字符串操作的需求，又可以正确地处理缺失值。例如，我用前面的数据
data 创建了一个 Pandas 的 Series：
In[4]: import pandas as pd
names = pd.Series(data)
names
Out[4]: 0 peter
        1 Paul
        2 None
        3 MARY
        4 gUIDO
dtype: object
现在就可以直接调用转换大写方法 capitalize() 将所有的字符串变成大写形式，缺失值会被跳过：
In[5]: names.str.capitalize()
Out[5]: 0 Peter
        1 Paul
        2 None
        3 Mary
        4 Guido
dtype: object
在 str 属性后面用 Tab 键，可以看到 Pandas 支持的所有向量化字符串方法。
```
## 3.11.2 Pandas字符串方法列表
```python
1. 与Python字符串方法相似的方法
几乎所有 Python 内置的字符串方法都被复制到 Pandas 的向量化字符串方法中。下面的表格列举了 Pandas 的 str 方法借鉴 Python 字符串方法的内容：
len() lower() translate() islower()
ljust() upper() startswith() isupper()
rjust() find() endswith() isnumeric()
center() rfind() isalnum() isdecimal()
zfill() index() isalpha() split()
strip() rindex() isdigit() rsplit()
rstrip() capitalize() isspace() partition()
lstrip() swapcase() istitle() rpartition()
需要注意的是，这些方法的返回值不同，例如 lower() 方法返回一个字符串 Series：
In[7]: monte.str.lower()
Out[7]: 0 graham chapman
1 john cleese
2 terry gilliam
3 eric idle
4 terry jones
5 michael palin
dtype: object
但是有些方法返回数值：
In[8]: monte.str.len()
Out[8]: 0 14
        1 11
        2 13
        3 9
        4 11
        5 13
dtype: int64
有些方法返回布尔值：
In[9]: monte.str.startswith('T')
Out[9]: 0 False
        1 False
        2 True
        3 False
        4 True
        5 False
dtype: bool
还有些方法返回列表或其他复合值：
In[10]: monte.str.split()
Out[10]: 0 [Graham, Chapman]
1 [John, Cleese]
2 [Terry, Gilliam]
3 [Eric, Idle]
4 [Terry, Jones]
5 [Michael, Palin]
dtype: object
2. 使用正则表达式的方法
还有一些支持正则表达式的方法可以用来处理每个字符串元素。表 3-4 中的内容是 Pandas向量化字符串方法根据 Python 标准库的 re 模块函数实现的 API。表3-4： Pandas向量化字符串方法与Python标准库的re模块函数的对应关系方法 描述
match() 对每个元素调用 re.match()，返回布尔类型值
extract() 对每个元素调用 re.match()，返回匹配的字符串组（groups）
findall() 对每个元素调用 re.findall()
replace() 用正则模式替换字符串
contains() 对每个元素调用 re.search()，返回布尔类型值
count() 计算符合正则模式的字符串的数量
split() 等价于 str.split()，支持正则表达式
rsplit() 等价于 str.rsplit()，支持正则表达式
通过这些方法，你就可以实现各种有趣的操作了。例如，可以提取元素前面的连续字母作为每个人的名字（first name）：
In[11]: monte.str.extract('([A-Za-z]+)')
Out[11]: 0 Graham
1 John
2 Terry
3 Eric
4 Terry
5 Michael
dtype: object
我们还能实现更复杂的操作，例如找出所有开头和结尾都是辅音字母的名字——这可以用正则表达式中的开始符号（^）与结尾符号（$）来实现：
In[12]: monte.str.findall(r'^[^AEIOU].*[^aeiou]$')
Out[12]: 0 [Graham Chapman]
1 []
2 [Terry Gilliam]
3 []
4 [Terry Jones]
5 [Michael Palin]
dtype: object
3. 其他字符串方法
还有其他一些方法也可以实现方便的操作（如表 3-5 所示）。
表3-5 其他Pandas字符串方法
方法 描述
get() 获取元素索引位置上的值，索引从 0 开始
slice() 对元素进行切片取值
slice_replace() 对元素进行切片替换
cat() 连接字符串（此功能比较复杂，建议阅读文档）
repeat() 重复元素
normalize() 将字符串转换为 Unicode 规范形式
pad() 在字符串的左边、右边或两边增加空格
wrap() 将字符串按照指定的宽度换行
join() 用分隔符连接 Series 的每个元素
get_dummies() 按照分隔符提取每个元素的 dummy 变量，转换为独热（one-hot）编码的 DataFrame
```
# 3.12　处理时间序列 
```python
由于 Pandas 最初是为金融模型而创建的，因此它拥有一些功能非常强大的日期、时间、带时间索引数据的处理工具。本节将介绍的日期与时间数据主要包含三类。
• 时间戳表示某个具体的时间点（例如 2015 年 7 月 4 日上午 7 点）。
• 时间间隔与周期表示开始时间点与结束时间点之间的时间长度，例如 2015 年（指的是
2015 年 1 月 1 日至 2015 年 12 月 31 日这段时间间隔）。周期通常是指一种特殊形式的
时间间隔，每个间隔长度相同，彼此之间不会重叠（例如，以 24 小时为周期构成每一天）。
• 时间增量（time delta）或持续时间（duration）表示精确的时间长度（例如，某程序运
行持续时间 22.56 秒）
```
## 3.12.1 Python的日期与时间工具
```python
在 Python 标准库与第三方库中有许多可以表示日期、时间、时间增量和时间跨度（timespan）的工具。尽管 Pandas 提供的时间序列工具更适合用来处理数据科学问题，但是了解 Pandas 与 Python 标准库以及第三方库中的其他时间序列工具之间的关联性将大有裨益。
1. 原生Python的日期与时间工具： datetime与dateutil
Python 基本的日期与时间功能都在标准库的 datetime 模块中。如果和第三方库 dateutil
模块搭配使用，可以快速实现许多处理日期与时间的功能。例如，你可以用 datetime 类型
创建一个日期：
In[1]: from datetime import datetime
datetime(year=2015, month=7, day=4)
Out[1]: datetime.datetime(2015, 7, 4, 0, 0)
或者使用 dateutil 模块对各种字符串格式的日期进行正确解析：
In[2]: from dateutil import parser
date = parser.parse("4th of July, 2015")
date
Out[2]: datetime.datetime(2015, 7, 4, 0, 0)
一旦有了 datetime 对象，就可以进行许多操作了，例如打印出这一天是星期几：
In[3]: date.strftime('%A')
Out[3]: 'Saturday'
2. 时间类型数组： NumPy的datetime64类型
Python 原生日期格式的性能弱点促使 NumPy 团队为 NumPy 增加了自己的时间序列类型。
datetime64 类型将日期编码为 64 位整数，这样可以让日期数组非常紧凑（节省内存）。
datetime64 需要在设置日期时确定具体的输入类型：
In[4]: import numpy as np
date = np.array('2015-07-04', dtype=np.datetime64)
date
Out[4]: array(datetime.date(2015, 7, 4), dtype='datetime64[D]')
datetime64 与 timedelta64对象的一个共同特点是，它们都是在基本时间单位（fundamental time unit）的基础上建立的。由于datetime64对象是64位精度，所以可编码的时间范围可以是基本单元的 264 倍。也就是说， datetime64 在时间精度（time resolution）与最大时间跨度（maximum time span）之间达成了一种衡。
3. Pandas的日期与时间工具： 理想与现实的最佳解决方案
Pandas 所有关于日期与时间的处理方法全部都是通过 Timestamp 对象实现的，它利用numpy.datetime64 的有效存储和向量化接口将 datetime 和 dateutil 的易用性有机结合起来。 Pandas 通过一组 Timestamp 对象就可以创建一个可以作为 Series 或 DataFrame 索引的DatetimeIndex
```
## 3.12.2 Pandas时间序列： 用时间作索引
```python
Pandas 时间序列工具非常适合用来处理带时间戳的索引数据。
```
## 3.12.3 Pandas时间序列数据结构
```python
本节将介绍 Pandas 用来处理时间序列的基础数据类型。
• 针对时间戳数据， Pandas 提供了 Timestamp 类型。与前面介绍的一样，它本质上是
Python 的原生 datetime 类型的替代品，但是在性能更好的 numpy.datetime64 类型的基
础上创建。对应的索引数据结构是 DatetimeIndex。
• 针对时间周期数据， Pandas 提供了 Period 类型。这是利用 numpy.datetime64 类型将固
定频率的时间间隔进行编码。对应的索引数据结构是 PeriodIndex。
• 针对时间增量或持续时间， Pandas 提供了 Timedelta 类型。 Timedelta 是一种代替 Python
原生 datetime.timedelta 类型的高性能数据结构，同样是基于 numpy.timedelta64 类型。
对应的索引数据结构是 TimedeltaIndex。
最基础的日期 / 时间对象是 Timestamp 和 DatetimeIndex。这两种对象可以直接使用，最常用的方法是 pd.to_datetime() 函数，它可以解析许多日期与时间格式。对 pd.to_datetime() 传递一个日期会返回一个 Timestamp 类型，传递一个时间序列会返回一个 DatetimeIndex 类型
有规律的时间序列： pd.date_range()
为了能更简便地创建有规律的时间序列， Pandas 提供了一些方法： pd.date_range() 可以处理时间戳、 pd.period_range() 可以处理周期、 pd.timedelta_range() 可以处理时间间隔。我们已经介绍过， Python 的 range() 和 NumPy 的 np.arange() 可以用起点、终点和步长（可选的）创建一个序列。 pd.date_range() 与之类似，通过开始日期、结束日期和频率代码（同样是可选的）创建一个有规律的日期序列，默认的频率是天
日期范围不一定非是开始时间与结束时间，也可以是开始时间与周期数 periods
可以通过 freq 参数改变时间间隔，默认值是 D
```
## 3.12.4　时间频率与偏移量
```python
Pandas 时间序列工具的基础是时间频率或偏移量（offset）代码。就像之前见过的 D（day）和 H（hour）代码，我们可以用这些代码设置任意需要的时间间隔。
D 天（calendar day，按日历算，含双休日） B 天（business day，仅含工作日）
W 周（weekly）
M 月末（month end） BM 月末（business month end，仅含工作日）
Q 季末（quarter end） BQ 季末（business quarter end，仅含工作日）
A 年末（year end） BA 年末（business year end，仅含工作日）
H 小时（hours） BH 小时（business hours，工作时间）
T 分钟（minutes）
S 秒（seconds）
L 毫秒（milliseonds）
U 微秒（microseconds）
N 纳秒（nanoseconds）
```
## 3.12.5　重新取样、 迁移和窗口
```python
用日期和时间直观地组织与获取数据是 Pandas 时间序列工具最重要的功能之一。 Pandas不仅支持普通索引功能（合并数据时自动索引对齐、直观的数据切片和取值方法等），还专为时间序列提供了额外的操作
1. 重新取样与频率转换
处理时间序列数据时，经常需要按照新的频率（更高频率、更低频率）对数据进行重新取样。你可以通过 resample() 方法解决这个问题，或者用更简单的 asfreq() 方法。这两个方法的主要差异在于， resample() 方法是以数据累计（data aggregation）为基础，而asfreq() 方法是以数据选择（data selection）为基础。
2. 时间迁移
另一种常用的时间序列操作是对数据按时间进行迁移。 Pandas 有两种解决这类问题的方法： shift() 和 tshift()。简单来说， shift() 就是迁移数据，而 tshift() 就是迁移索引。两种方法都是按照频率代码进行迁移。
3. 移动时间窗口
Pandas 处理时间序列数据的第 3 种操作是移动统计值（rolling statistics）。这些指标可以通过 Series 和 DataFrame 的 rolling() 属性来实现，它会返回与 groupby 操作类似的结果（详情请参见 3.9 节）。移动视图（rolling view）使得许多累计操作成为可能。
```
















