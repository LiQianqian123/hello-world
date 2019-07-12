# 导入需要用到的python库  
```python
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
# 导入数据集
```python
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
```
# 将数据集拆分为训练集和测试集
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)
```
# 特征缩放
```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
# 对测试集进行决策树分类拟合  
```python 
# sklearn中分类决策树的重要参数：
# 1.criterion：确定不纯度的计算方法，帮忙找出最佳节点和最佳分枝，不纯度越低，决策树对训练集的拟合越好；
# 不填默认基尼系数，填写gini使用基尼系数，填写entropy使用信息增益
# 2. random_state和splitter：
# 用来设置分枝中的随机模式的参数，默认None，在高维度时随机性会表现更明显，低维度的数据（比如鸢尾花数据集）随机性几乎不会显现。输入任意整数，会一直长出同一棵树，让模型稳定下来。
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)  
```
# 预测测试集结果
```python
y_pred = classifier.predict(X_test)
```
# 制作混淆矩阵
```python
cm = confusion_matrix(y_test, y_pred)
```
# 将训练集结果进行可视化  
```python
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -
                               1, stop=X_set[:, 0].max() +
                               1, step=0.01), np.arange(start=X_set[:, 1].min() -
                                                        1, stop=X_set[:, 1].max() +
                                                        1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Decision Tree Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()  
```
# 将测试集结果进行可视化  
```python
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() -
                               1, stop=X_set[:, 0].max() +
                               1, step=0.01), np.arange(start=X_set[:, 1].min() -
                                                        1, stop=X_set[:, 1].max() +
                                                        1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Decision Tree Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()  
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_25_1.png)
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_25_2.png)
