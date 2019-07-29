# 4.7　频次直方图、 数据区间划分和分布密度
```python
只要导入了画图的函数，只用一行代码就可以创建一个简易的频次直方图：
In[1]: %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
data = np.random.randn(1000)
In[2]: plt.hist(data);
hist() 有许多用来调整计算过程和显示效果的选项，下面是一个更加个性化的频次直方图
In[3]: plt.hist(data, bins=30, normed=True, alpha=0.5,
                histtype='stepfilled', color='steelblue',
                edgecolor='none');
关于 plt.hist 自定义选项的更多内容都在它的程序文档中。
我发现在用频次直方图对不同分布特征的样本进行对比时，将 histtype='stepfilled' 与透明性设置参数 alpha 搭配使用的效果非常好：
In[4]: x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)
kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=40)
plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs);
如果你只需要简单地计算频次直方图（就是计算每段区间的样本数），而并不想画图显示
它们，那么可以直接用 np.histogram()：
In[5]: counts, bin_edges = np.histogram(data, bins=5)
print(counts)
[ 12 190 468 301 29]
二维频次直方图与数据区间划分
就像将一维数组分为区间创建一维频次直方图一样，我们也可以将二维数组按照二维区
间进行切分，来创建二维频次直方图。下面将简单介绍几种创建二维频次直方图的方法。
首先，用一个多元高斯分布（multivariate Gaussian distribution）生成 x 轴与 y 轴的样本数据：
In[6]: mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
1. plt.hist2d： 二维频次直方图
画二维频次直方图最简单的方法就是使用 Matplotlib 的 plt.hist2d 函数
In[12]: plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')
与 plt.hist 函数一样， plt.hist2d 也有许多调整图形与区间划分的配置选项，详细内容都
在程序文档中。另外，就像 plt.hist 有一个只计算结果不画图的 np.histogram 函数一样，
plt.hist2d 类似的函数是 np.histogram2d
In[8]: counts, xedges, yedges = np.histogram2d(x, y, bins=30)
2. plt.hexbin： 六边形区间划分
二维频次直方图是由与坐标轴正交的方块分割而成的，还有一种常用的方式是用正六边
形分割。 Matplotlib 提供了 plt.hexbin 满足此类需求，将二维数据集分割成蜂窝状：
In[9]: plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')
3. 核密度估计
还有一种评估多维数据分布密度的常用方法是核密度估计（kernel density estimation，
KDE）。我们将在 5.13 节详细介绍这种方法，现在先来简单地演示如何用 KDE 方法“抹
掉”空间中离散的数据点，从而拟合出一个平滑的函数。在 scipy.stats 程序包里面有一
个简单快速的 KDE 实现方法，下面就是用这个方法演示的简单示例
In[10]: from scipy.stats import gaussian_kde
# 拟合数组维度[Ndim, Nsamples]
data = np.vstack([x, y])
kde = gaussian_kde(data)
# 用一对规则的网格数据进行拟合
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
# 画出结果图
plt.imshow(Z.reshape(Xgrid.shape),
origin='lower', aspect='auto',
extent=[-3.5, 3.5, -6, 6],
cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")
KDE 方法通过不同的平滑带宽长度（smoothing length）在拟合函数的准确性与平滑性之
间作出权衡（无处不在的偏差与方差的取舍问题的一个例子）。想找到恰当的平滑带宽长
度是件很困难的事， gaussian_kde 通过一种经验方法试图找到输入数据平滑长度的近似
最优解。
```
# 4.8　配置图例
```python
想在可视化图形中使用图例，可以为不同的图形元素分配标签。前面介绍过如何创建简单
的图例，现在将介绍如何在 Matplotlib 中自定义图例的位置与艺术风格。
可以用 plt.legend() 命令来创建最简单的图例，它会自动创建一个包含每个图形元素的图例：
In[1]: import matplotlib.pyplot as plt
plt.style.use('classic')
In[2]: %matplotlib inline
import numpy as np
In[3]: x = np.linspace(0, 10, 1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend();
但是，我们经常需要对图例进行各种个性化的配置。例如，我们想设置图例的位置，并取
消外边框：
In[4]: ax.legend(loc='upper left', frameon=False)
       fig
还可以用 ncol 参数设置图例的标签列数（如图 4-43 所示）：
In[5]: ax.legend(frameon=False, loc='lower center', ncol=2)
       fig
还可以为图例定义圆角边框（fancybox）、增加阴影、改变外边框透明度（framealpha 值），
或者改变文字间距
In[6]: ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
       fig
```
## 4.8.1　选择图例显示的元素
```python
图例会默认显示所有元素的标签。如果你不想显示全部，可以通过一些图形命令来指定显示
图例中的哪些元素和标签。 plt.plot() 命令可以一次创建多条线，返回线条实例列表。一
种方法是将需要显示的线条传入 plt.legend()，另一种方法是只为需要在图例中显示的线条
设置标签。
In[7]: y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))
       lines = plt.plot(x, y)
       # lines变量是一组plt.Line2D实例
       plt.legend(lines[:2], ['first', 'second']);
当然也可以只为需要在图例中显示的元素设置标签：
In[8]: plt.plot(x, y[:, 0], label='first')
plt.plot(x, y[:, 1], label='second')
plt.plot(x, y[:, 2:])
plt.legend(framealpha=1, frameon=True);
```
## 4.8.2　在图例中显示不同尺寸的点
```python
有时，默认的图例仍然不能满足我们的可视化需求。例如，你可能需要用不同尺寸的点来
表示数据的特征，并且希望创建这样的图例来反映这些特征。下面的示例将用点的尺寸来
表明美国加州不同城市的人口数量。如果我们想要一个通过不同尺寸的点显示不同人口数
量级的图例，可以通过隐藏一些数据标签来实现这个效果
In[9]: import pandas as pd
cities = pd.read_csv('data/california_cities.csv')
# 提取感兴趣的数据
lat, lon = cities['latd'], cities['longd']
population, area = cities['population_total'], cities['area_total_km2']
# 用不同尺寸和颜色的散点图表示数据，但是不带标签
plt.scatter(lon, lat, label=None,
c=np.log10(population), cmap='viridis',
s=area, linewidth=0, alpha=0.5)
plt.axis(aspect='equal')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.colorbar(label='log$_{10}$(population)')
plt.clim(3, 7)
# 下面创建一个图例：
# 画一些带标签和尺寸的空列表
for area in [100, 300, 500]:
plt.scatter([], [], c='k', alpha=0.3, s=area,
label=str(area) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False,
labelspacing=1, title='City Area')
plt.title('California Cities: Area and Population');
```
## 4.8.3　同时显示多个图例
```python
有时，我们可能需要在同一张图上显示多个图例。不过，用 Matplotlib 解决这个问题并不
容易，因为通过标准的 legend 接口只能为一张图创建一个图例。如果你想用 plt.legend()
或 ax.legend() 方法创建第二个图例，那么第一个图例就会被覆盖。但是，我们可以通
过从头开始创建一个新的图例艺术家对象（legend artist），然后用底层的（lower-level）
ax.add_artist() 方法在图上添加第二个图例：
In[10]: fig, ax = plt.subplots()
lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)
for i in range(4):
lines += ax.plot(x, np.sin(x - i * np.pi / 2),
styles[i], color='black')
ax.axis('equal')
# 设置第一个图例要显示的线条和标签
ax.legend(lines[:2], ['line A', 'line B'],
loc='upper right', frameon=False)
# 创建第二个图例，通过add_artist方法添加到图上
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
loc='lower right', frameon=False)
ax.add_artist(leg);
```
# 4.9　配置颜色条
```python
图例通过离散的标签表示离散的图形元素。然而，对于图形中由彩色的点、线、面构成的
连续标签，用颜色条来表示的效果比较好。在 Matplotlib 里面，颜色条是一个独立的坐标
轴，可以指明图形中颜色的含义。由于本书是单色印刷，你可以在本书在线附录（https://
github.com/jakevdp/PythonDataScienceHandbook）中查看这一节图形的彩色版本。首先还是
导入需要使用的画图工具：
In[1]: import matplotlib.pyplot as plt
plt.style.use('classic')
In[2]: %matplotlib inline
import numpy as np
和在前面看到的一样，通过 plt.colorbar 函数就可以创建最简单的颜色条：
In[3]: x = np.linspace(0, 10, 1000)
I = np.sin(x) * np.cos(x[:, np.newaxis])
plt.imshow(I)
plt.colorbar();
```
## 4.9.1　配置颜色条
```python
可以通过 cmap 参数为图形设置颜色条的配色方案：
In[4]: plt.imshow(I, cmap='gray');
所有可用的配色方案都在 plt.cm 命名空间里面，在 IPython 里通过 Tab 键就可以查看所有的配置方案：
plt.cm.<TAB>
1. 选择配色方案
顺序配色方案
由一组连续的颜色构成的配色方案（例如 binary 或 viridis）。
互逆配色方案
通常由两种互补的颜色构成，表示正反两种含义（例如 RdBu 或 PuOr）。
定性配色方案
随机顺序的一组颜色（例如 rainbow 或 jet）。
jet 是一种定性配色方案，曾是 Matplotlib 2.0 之前所有版本的默认配色方案。把它作为默
认配色方案实在不是个好主意，因为定性配色方案在对定性数据进行可视化时的选择空间
非常有限。随着图形亮度的提高，经常会出现颜色无法区分的问题。
可以通过把 jet 转换为黑白的灰度图看看具体的颜色（如图 4-51 所示）：
In[5]:
from matplotlib.colors import LinearSegmentedColormap
def grayscale_cmap(cmap):
"""为配色方案显示灰度图"""
cmap = plt.cm.get_cmap(cmap)
colors = cmap(np.arange(cmap.N))
# 将RGBA色转换为不同亮度的灰度值
# 参考链接http://alienryderflex.com/hsp.html
RGB_weight = [0.299, 0.587, 0.114]
luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
colors[:, :3] = luminance[:, np.newaxis]
return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)
def view_colormap(cmap):
"""用等价的灰度图表示配色方案"""
cmap = plt.cm.get_cmap(cmap)
colors = cmap(np.arange(cmap.N))
cmap = grayscale_cmap(cmap)
grayscale = cmap(np.arange(cmap.N))
fig, ax = plt.subplots(2, figsize=(6, 2),
subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow([colors], extent=[0, 10, 0, 1])
ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
In[6]: view_colormap('jet')
2. 颜色条刻度的限制与扩展功能的设置
Matplotlib 提供了丰富的颜色条配置功能。由于可以将颜色条本身仅看作是一个 plt.Axes
实例，因此前面所学的所有关于坐标轴和刻度值的格式配置技巧都可以派上用场。颜色条
有一些有趣的特性。例如，我们可以缩短颜色取值的上下限，对于超出上下限的数据，通
过 extend 参数用三角箭头表示比上限大的数或者比下限小的数。这种方法很简单，比如你
想展示一张噪点图
In[10]: # 为图形像素设置1%噪点
speckles = (np.random.random(I.shape) < 0.01)
I[speckles] = np.random.normal(0, 3, np.count_nonzero(speckles))
plt.figure(figsize=(10, 3.5))
plt.subplot(1, 2, 1)
plt.imshow(I, cmap='RdBu')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(I, cmap='RdBu')
plt.colorbar(extend='both')
plt.clim(-1, 1);
3. 离散型颜色条
虽然颜色条默认都是连续的，但有时你可能也需要表示离散数据。最简单的做法就是使用
plt.cm.get_cmap() 函数，将适当的配色方案的名称以及需要的区间数量传进去即可
In[11]: plt.imshow(I, cmap=plt.cm.get_cmap('Blues', 6))
plt.colorbar()
plt.clim(-1, 1);
```
## 4.9.2　案例： 手写数字
```python
数据在 Scikit—Learn 里面，包含近 2000 份 8× 8 的手写数字缩略图。
先下载数据，然后用 plt.imshow() 对一些图形进行可视化（如图 4-57 所示）：
In[12]: # 加载数字0~5的图形，对其进行可视化
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)
fig, ax = plt.subplots(8, 8, figsize=(6, 6))
for i, axi in enumerate(ax.flat):
axi.imshow(digits.images[i], cmap='binary')
axi.set(xticks=[], yticks=[])
由于每个数字都由 64 像素的色相（hue）构成，因此可以将每个数字看成是一个位于 64
维空间的点，即每个维度表示一个像素的亮度。但是想通过可视化来描述如此高维度的空
间是非常困难的。一种解决方案是通过降维技术，在尽量保留数据内部重要关联性的同时
降低数据的维度，，例如流形学习（manifold learning）。降维是无监督学习的重要内容
暂且不提具体的降维细节，先来看看如何用流形学习将这些数据投影到二维空间进行可视化：
In[13]: # 用IsoMap方法将数字投影到二维空间
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)
我们将用离散型颜色条来显示结果，调整 ticks 与 clim 参数来改善颜色条：
In[14]: # 画图
plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
c=digits.target, cmap=plt.cm.get_cmap('cubehelix', 6))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5, 5.5)
这个投影结果还向我们展示了一些数据集的有趣特性。例如，数字 5 与数字 3 在投影中有
大面积重叠，说明一些手写的 5 与 3 难以区分，因此自动分类算法也更容易搞混它们。其
他的数字，像数字 0 与数字 1，隔得特别远，说明两者不太可能出现混淆。这个观察结果
也符合我们的直观感受，因为 5 和 3 看起来确实要比 0 和 1 更像。
```
# 4.10　多子图
```python
有时候需要从多个角度对数据进行对比。 Matplotlib 为此提出了子图（subplot）的概念：在
较大的图形中同时放置一组较小的坐标轴。这些子图可能是画中画（inset）、网格图（grid
of plots），或者是其他更复杂的布局形式。在这一节中，我们将介绍四种用 Matplotlib 创建
子图的方法。首先，在 Notebook 中导入画图需要的程序库：
In[1]: %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
```
## 4.10.1 plt.axes： 手动创建子图
```python
创建坐标轴最基本的方法就是使用 plt.axes 函数。前面已经介绍过，这个函数的默认配置
是创建一个标准的坐标轴，填满整张图。它还有一个可选参数，由图形坐标系统的四个值
构成。这四个值分别表示图形坐标系统的 [bottom, left, width, height]（底坐标、左坐
标、宽度、高度），数值的取值范围是左下角（原点）为 0，右上角为 1。
如果想要在右上角创建一个画中画，那么可以首先将 x 与 y 设置为 0.65（就是坐标轴原点
位于图形高度 65% 和宽度 65% 的位置），然后将 x 与 y 扩展到 0.2（也就是将坐标轴的宽
度与高度设置为图形的 20%）。 
In[2]: ax1 = plt.axes() # 默认坐标轴
ax2 = plt.axes([0.65, 0.65, 0.2, 0.2])
面向对象画图接口中类似的命令有 fig.add_axes()。用这个命令创建两个竖直排列的坐标轴：
In[3]: fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
xticklabels=[], ylim=(-1.2, 1.2))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
ylim=(-1.2, 1.2))
x = np.linspace(0, 10)
ax1.plot(np.sin(x))
ax2.plot(np.cos(x));
现在就可以看到两个紧挨着的坐标轴（上面的坐标轴没有刻度）：上子图（起点 y 坐标为
0.5 位置）与下子图的 x 轴刻度是对应的（起点 y 坐标为 0.1，高度为 0.4）。
```
## 4.10.2 plt.subplot： 简易网格子图
```python
若干彼此对齐的行列子图是常见的可视化任务， Matplotlib 拥有一些可以轻松创建它们的
简便方法。最底层的方法是用 plt.subplot() 在一个网格中创建一个子图。这个命令有三
个整型参数——将要创建的网格子图行数、列数和索引值，索引值从 1 开始，从左上角到
右下角依次增大
In[4]: for i in range(1, 7):
plt.subplot(2, 3, i)
plt.text(0.5, 0.5, str((2, 3, i)),
fontsize=18, ha='center')
plt.subplots_adjust 命令可以调整子图之间的间隔。用面向对象接口的命令 fig.add_
subplot() 可以取得同样的效果：
In[5]: fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 7):
ax = fig.add_subplot(2, 3, i)
ax.text(0.5, 0.5, str((2, 3, i)),
fontsize=18, ha='center')
我们通过 plt.subplots_adjust 的 hspace 与 wspace 参数设置与图形高度与宽度一致的子图
间距，数值以子图的尺寸为单位
```
## 4.10.3 plt.subplots： 用一行代码创建网格
```python
当你打算创建一个大型网格子图时，就没办法使用前面那种亦步亦趋的方法了，尤其是当
你想隐藏内部子图的 x 轴与 y 轴标题时。出于这一需求， plt.subplots() 实现了你想要的
功能（需要注意此处 subplots 结尾多了个 s）。这个函数不是用来创建单个子图的， 而是
用一行代码创建多个子图，并返回一个包含子图的 NumPy 数组。关键参数是行数与列数，
以及可选参数 sharex 与 sharey，通过它们可以设置不同子图之间的关联关系。
我们将创建一个 2× 3 网格子图，每行的 3 个子图使用相同的 y 轴坐标，每列的 2 个子图
使用相同的 x 轴坐标
In[6]: fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
设置 sharex 与 sharey 参数之后，我们就可以自动去掉网格内部子图的标签，让图形看起
来更整洁。坐标轴实例网格的返回结果是一个 NumPy 数组，这样就可以通过标准的数组
取值方式轻松获取想要的坐标轴了：
In[7]: # 坐标轴存放在一个NumPy数组中，按照[row, col]取值
for i in range(2):
for j in range(3):
ax[i, j].text(0.5, 0.5, str((i, j)),
fontsize=18, ha='center')
fig
与 plt.subplot()1 相比， plt.subplots() 与 Python 索引从 0 开始的习惯保持一致。
```
## 4.10.4 plt.GridSpec： 实现更复杂的排列方式
```python
如果想实现不规则的多行多列子图网格， plt.GridSpec() 是最好的工具。 plt.GridSpec()
对象本身不能直接创建一个图形，它只是 plt.subplot() 命令可以识别的简易接口。例如，
一个带行列间距的 2× 3 网格的配置代码如下所示：
In[8]: grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)
可以通过类似 Python 切片的语法设置子图的位置和扩展尺寸：
In[9]: plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:])
plt.subplot(grid[1, :2])
plt.subplot(grid[1, 2])
In[10]: # 创建一些正态分布数据
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T
# 设置坐标轴和网格配置方式
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
# 主坐标轴画散点图
main_ax.plot(x, y, 'ok', markersize=3, alpha=0.2)
# 次坐标轴画频次直方图
x_hist.hist(x, 40, histtype='stepfilled',
orientation='vertical', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled',
orientation='horizontal', color='gray')
y_hist.invert_xaxis()
```
#4.11　文字与注释
```python
一个优秀的可视化作品就是给读者讲一个精彩的故事。虽然在一些场景中，这个故事可以
完全通过视觉来表达，不需要任何多余的文字。但在另外一些场景中，辅之以少量的文字
提示（textual cue）和标签是必不可少的。虽然最基本的注释（annotation）类型可能只是
坐标轴标题与图标题，但注释可远远不止这些。让我们可视化一些数据，看看如何通过添
加注释来更恰当地表达信息。还是先在 Notebook 里面导入画图需要用到的一些函数：
In[1]: %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
```
## 4.11.1　案例： 节假日对美国出生率的影响
```python
让我们用 3.10.4 节介绍过的的数据来演示。在那个案例中，我们画了一幅图表示美国每
一年的出生人数。和前面一样，数据可以在 https://raw.githubusercontent.com/jakevdp/dataCDCbirths/master/births.csv 下载。
首先用前面介绍过的清洗方法处理数据，然后画出结果（如图 4-67 所示）：
In[2]:
births = pd.read_csv('births.csv')
quartiles = np.percentile(births['births'], [25, 50, 75])
mu, sig = quartiles[1], 0.74 * (quartiles[2] - quartiles[0])
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year +
100 * births.month +
births.day, format='%Y%m%d')
births_by_date = births.pivot_table('births',
[births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day)
for (month, day) in births_by_date.index]
In[3]: fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax);
在用这样的图表达观点时，如果可以在图中增加一些注释，就更能吸引读者的注意了。可
以通过 plt.text/ ax.text 命令手动添加注释，它们可以在具体的 x / y 坐标点上放上文字
（如图 4-68 所示）：
In[4]: fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax)
# 在图上增加文字标签
style = dict(size=10, color='gray')
ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style)
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas ", ha='right', **style)
# 设置坐标轴标题
ax.set(title='USA births by day of year (1969-1988)',
ylabel='average daily births')
# 设置x轴刻度值，让月份居中显示
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'));
ax.text 方法需要一个 x 轴坐标、一个 y 轴坐标、一个字符串和一些可选参数，比如文字
的颜色、字号、风格、对齐方式以及其他文字属性。这里用了 ha='right' 与 ha='center'，
ha 是水平对齐方式（horizonal alignment）的缩写。关于配置参数的更多信息，请参考
plt.text() 与 mpl.text.Text() 的程序文档。
```
## 4.11.2　坐标变换与文字位置
```python
前面的示例将文字放在了目标数据的位置上。但有时候可能需要将文字放在与数据无关的位置
上，比如坐标轴或者图形中。在 Matplotlib 中，我们通过调整坐标变换（transform）来实现。
任何图形显示框架都需要一些变换坐标系的机制。例如，当一个位于 (x, y) = (1, 1) 位置的
点需要以某种方式显示在图上特定的位置时，就需要用屏幕的像素来表示。用数学方法处
理这种坐标系变换很简单， Matplotlib 有一组非常棒的工具可以实现类似功能（这些工具
位于 matplotlib.transforms 子模块中）。
虽然一般用户并不需要关心这些变换的细节，但是了解这些知识对在图上放置文字大有帮
助。一共有三种解决这类问题的预定义变换方式。
ax.transData
以数据为基准的坐标变换。
ax.transAxes
以坐标轴为基准的坐标变换（以坐标轴维度为单位）。
fig.transFigure
以图形为基准的坐标变换（以图形维度为单位）。
```
## 4.11.3　箭头与注释
```python
除了刻度线和文字，简单的箭头也是一种有用的注释标签。
在 Matplotlib 里面画箭头通常比你想象的要困难。虽然有一个 plt.arrow() 函数可以实现这
个功能，但是我不推荐使用它，因为它创建出的箭头是 SVG 向量图对象，会随着图形分
辨率的变化而改变，最终的结果可能完全不是用户想要的。我要推荐的是 plt.annotate()
函数。这个函数既可以创建文字，也可以创建箭头，而且它创建的箭头能够进行非常灵活
的配置。
下面用 annotate 的一些配置选项来演示：
In[7]: %matplotlib inline
fig, ax = plt.subplots()
x = np.linspace(0, 20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')
ax.annotate('local maximum', xy=(6.28, 1), xytext=(10, 4),
arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('local minimum', xy=(5 * np.pi, -1), xytext=(2, -6),
arrowprops=dict(arrowstyle="->",
connectionstyle="angle3,angleA=0,angleB=-90"));
箭头的风格是通过 arrowprops 字典控制的，里面有许多可用的选项。
```








































