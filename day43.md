# K-均值聚类
```python
1.随机产生K个分类特征的中心点(cluster center)
2.计算数据点到中心点的距离(distence)
3.数据点到那个中心点的距离最近就分到哪个类(cluster)
4.迭代：更新中心点位置，重新计算距离并分配类别，直到总体距离最小
```
# 代码实现
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans   # 导入K-均值函数
# coding=utf-8
# 读取网页中的数据表
table = []
for i in range(1, 7):
    table.append(pd.read_html('https://nba.hupu.com/stats/players/pts/%d' % i)[0])  # 获取网页数据
# 所有数据纵向合并为数据框
players = pd.concat(table)
players.drop(0, inplace=True) # 删除行标签为0的记录，因为换完页，行标签为0时，没有数据

X = players.iloc[1:, 9].values  # 自变量为罚球命中率
Y = players.iloc[1:, 5].values  # 因变量为命中率
# 将带百分号的字符型转化为float型
x = []
for i in X:
     x.append(float(i.strip('%')))  # 去掉百分号
x = np.array(x)/100
# print(x)

y = []
for j in Y:
     y.append(float(j.strip('%')))
y = np.array(y)/100
# print(y)
# 合并成矩阵
n = np.array([x.ravel(), y.ravel()]).T
# print (n)
# 绘制原始数据散点图
plt.style.use('ggplot')  # 设置绘图风格
plt.scatter(n[:, 0], n[:, 1])  # 画散点图
plt.xlabel('free throw hit rate')
plt.ylabel('hit rate')
plt.show()
# 选择最佳K值
X = n[:]
K = range(1, int(np.sqrt(n.shape[0])))  # 确定K值的范围
GSSE = []
for k in K:  # 统计不同簇数下的平方误差
    SSE = []
    kmeans = KMeans(n_clusters=k, random_state=10)  # 构造聚类器
    kmeans.fit(X)  # 聚类
    labels = kmeans.labels_  # 获取聚类标签

    centers = kmeans.cluster_centers_  # 获取每个簇的形心
    for label in set(labels):  # set创建不重复集合
# 不同簇内的数据减去该簇内的形心
        SSE.append(np.sum((np.array(n[labels == label, ])-np.array(centers[label, :]))**2))
# 总的误差
    GSSE.append(np.sum(SSE))
# 绘制K的个数与GSSE的关系
plt.plot(K, GSSE, 'b*-')
plt.xlabel('K')
plt.ylabel('Error')
plt.title('optimal solution')
plt.show()

#调用sklearn的库函数
num_clusters = 6
kmeans = KMeans(n_clusters=num_clusters, random_state=1)
kmeans.fit(X)

# 聚类中心
centers = kmeans.cluster_centers_

# 绘制簇散点图
plt.scatter(x=X[:, 0], y=X[:, 1], c=kmeans.labels_)
# 绘制形心散点图
plt.scatter(centers[:, 0], centers[:, 1], c='k', marker='*')
plt.xlabel('free throw hit rate')
plt.ylabel('hit rate')
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_43_1.png)
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_1-43_2.png)
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_1-43_3.png)
