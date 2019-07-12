# 导入库  
```python
from matplotlib.colors import ListedColormap  
from sklearn.metrics import confusion_matrix  
from sklearn.svm import SVC  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
```
# 导入数据  
```python
dataset = pd.read_csv('Social_Network_Ads.csv')  
X = dataset.iloc[:, [2, 3]].values  
y = dataset.iloc[:, 4].values  
```  
# 拆分数据集为训练集合和测试集合
```python  
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0) 
```
# 特征量化  
```python  
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.fit_transform(X_test)  
```  
# 配适SVM到训练集合
```python  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(X_train, y_train)  
```
# 预测测试集合结果  
```python  
y_pred = classifier.predict(X_test)
```  
# 创建混淆矩阵  
```python  
cm = confusion_matrix(y_test, y_pred)  
```
# 训练集合结果可视化
```python  
from matplotlib.colors import ListedColormap  
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
plt.title('SVM (Training set)')  
plt.xlabel('Age')  
plt.ylabel('Estimated Salary')
# 标注右上角的图例  
plt.legend()  
plt.show()  
```
# 测试集合结果可视化
```python
from matplotlib.colors import ListedColormap
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
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
# 标注右上角的图例
plt.legend()
plt.show()  
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_12_1.png)  
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_12_2.png)
