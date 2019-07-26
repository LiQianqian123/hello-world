3.10　数据透视表
数据透视表（pivottable）是一种类似的操作方法，常见于 Excel 与类似的表格应用中。数据透视表将每一列数据作为输入，输出将数据不断细分成多个维度累计信息的二维数据表。人们有时容易弄
混数据透视表与 GroupBy，但我觉得数据透视表更像是一种多维的 GroupBy 累计操作。也就是说，虽然你也可以分割 - 应用 - 组合，但是分割与组合不是发生在一维索引上，而是在二维网格上（行列同时分组）。
3.10.1　演示数据透视表
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
3.10.2　手工制作数据透视表
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
需要注意的是，这里忽略了一个参数 values。当我们为 aggfunc 指定映射关系的时候，待
透视的数值就已经确定了。
当需要计算每一组的总数时，可以通过 margins 参数来设置：
In[9]: titanic.pivot_table('survived', index='sex', columns='class', margins=True)
Out[9]: class First Second Third All
        sex
        female 0.968085 0.921053 0.500000 0.742038
        male 0.368852 0.157407 0.135447 0.188908
        All 0.629630 0.472826 0.242363 0.383838
这样就可以自动获取不同性别下船舱等级与生还率的相关信息、不同船舱等级下性别与生还率的相关信息，以及全部乘客的生还率为 38%。 margin 的标签可以通过 margins_name 参数进行自定义，默认值是 "All"。
3.11　向量化字符串操作
使用 Python 的一个优势就是字符串处理起来比较容易。在此基础上创建的 Pandas 同样提供了一系列向量化字符串操作（vectorized string operation），它们都是在处理（清洗）现实工作中的数据时不可或缺的功能。
3.11.1 Pandas字符串操作简介
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
Pandas 为包含字符串的 Series 和 Index 对象提供的 str 属性堪称两全其美的方法，它既
可以满足向量化字符串操作的需求，又可以正确地处理缺失值。例如，我们用前面的数据
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
现在就可以直接调用转换大写方法 capitalize() 将所有的字符串变成大写形式，缺失值会
被跳过：
In[5]: names.str.capitalize()
Out[5]: 0 Peter
        1 Paul
        2 None
        3 Mary
        4 Guido
dtype: object
在 str 属性后面用 Tab 键，可以看到 Pandas 支持的所有向量化字符串方法。
3.11.2 Pandas字符串方法列表
1. 与Python字符串方法相似的方法
几乎所有 Python 内置的字符串方法都被复制到 Pandas 的向量化字符串方法中。下面的表
格列举了 Pandas 的 str 方法借鉴 Python 字符串方法的内容：
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




















