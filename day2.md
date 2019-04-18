# Step 1 : 数据预处理  
```python
# matplotlib.pyplot: 是python上的一个2D绘图库
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('studentscores.csv')
# iloc是通过行列号来获取数据，loc是通过标签索引数据
# X获取所有行第0列到第一列的数据
X = dataset.iloc[:, : 1].values
# Y获取所有行第一列的数据
Y = dataset.iloc[:, 1].values
print(X)
print(Y)
# 训练集 ：(training set)是用来训练模型或确定模型参数的；测试集 ： （test set）则纯粹是为了测试已经训练好的模型的推广能力
# train_test_split()是sklearn.model_selection中的分离器函数，用于将数组或矩阵划分为训练集和测试集
# random_state
# 随机数种子控制每次划分训练集和测试集的模式，其取值不变时划分得到的结果一模一样，其值改变时，划分得到的结果不同,此处为了保证程序每次运行都分割一样的训练集和测试集。否则，同样的算法模型在不同的训练集和测试集上的效果不一样。
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)  
```
# step 2 : 训练集使用简单线性回归模型来训练  
```python
# sklearn.linear_model模型实现了广义线性模型，包括线性回归、Ridge回归、Bayesian回归等
# LinearRegression模型: 简单的线性回归模型
regressor = LinearRegression()
# fit( ): 求得训练集X的均值，方差，最大值，最小值这些训练集固有的属性,可以理解为一个训练过程
# transform( ): 在fit的基础上，进行标准化，降维，归一化等操作
regressor = regressor.fit(X_train, Y_train)
print(regressor.coef_)
print(regressor.intercept_)
# 其中coef_存放回归系数，intercept_则存放截距。  
```
# step 3 ：预测结果  
```python
Y_pred = regressor.predict(X_test)
print(Y_pred)  
```
# step 4 ：可视化
## 训练集结果可视化  
```python
# plt.scatter：绘制散点图，plt.scatter(x,y,s,c,marker) s:散点大小  c:散点颜色  maker:散点形状
# plt.plot：绘制连续曲线
plt.figure(1)
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()  
```
## 测试集结果可视化  
```python
plt.figure(2)
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.show()  
```  
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_1.png)  
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_2.png)

