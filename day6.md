# Step1 : 数据预处理  
## 导入库  
```python
from matplotlib.colors import ListedColormap  
from sklearn.metrics import confusion_matrix  
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import train_test_split  
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
```  
## 导入数据集  
```python
dataset = pd.read_csv('Social_Network_Ads.csv')  
X = dataset.iloc[:, [2, 3]].values  
Y = dataset.iloc[:, 4].values 
```
## 将数据集分成训练集和测试集  
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)  
```
## 特征缩放  
```python
# sklearn中的StandardScaler，计算训练集的平均值和标准差，以便测试数据集使用相同的变换  
# ss = sklearn.preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)  
# copy; with_mean;with_std 默认的值都是True.  
# copy 如果为false,就会用归一化的值替代原来的值;如果被标准化的数据不是np.array或scipy.sparse CSR matrix, 原来的数据还是被copy而不是被替代  
# with_mean 在处理sparse CSR或者 CSC matrices 一定要设置False不然会超内存  
# scale_： 缩放比例，同时也是标准差；mean_：  
# 每个特征的平均值；var_:每个特征的方差；n_sample_seen_:样本数量，可以通过patial_fit 增加  
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)  
X_test = sc.transform(X_test)  
```
# Step2 :  逻辑回归模型  
```python
# 该项工作的库将会是一个线性模型库，之所以被称为线性是因为逻辑回归是一个线性分类器，这意味着我们在二维空间中，我们两类用户（购买和不购买）将被一条直线分割。然后导入逻辑回归类。下一步我们将创建该类的对象，它将作为我们训练集的分类器。  
# 将逻辑回归应用于训练集  
from sklearn.linear_model import LogisticRegression  
classifier = LogisticRegression()  
classifier.fit(X_train, y_train)
```
# Step3 : 预测（预测测试集结果）
```python
y_pred = classifier.predict(X_test) 
```
# Step4 : 评估预测  
## 生成混淆矩阵
```python
# 混淆矩阵（confusion matrix），又称为可能性表格或是错误矩阵。它是一种特定的矩阵用来呈现算法性能的可视化效果，通常是监督学习。其每一列代表预测值，每一行代表的是实际的类别。混淆矩阵就是为了进一步分析性能而对该算法测试结果做出的总结  
# 所有正确的预测结果都在对角线上，所以从混淆矩阵中可以很方便直观的看出哪里有错误，因为他们呈现在对角线外面。  
cm = confusion_matrix(y_test, y_pred)  
```
## 可视化  
```python
X_set, y_set = X_train, y_train
X1, X2 = np. meshgrid(np. arange(start=X_set[:, 0].min() -
                                 1, stop=X_set[:, 0].max() +
                                 1, step=0.01), np. arange(start=X_set[:, 1].min() -
                                                           1, stop=X_set[:, 1].max() +
                                                           1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np. unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt. title(' LOGISTIC(Training set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()

X_set, y_set = X_test, y_test
X1, X2 = np. meshgrid(np. arange(start=X_set[:, 0].min() -
                                 1, stop=X_set[:, 0].max() +
                                 1, step=0.01), np. arange(start=X_set[:, 1].min() -
                                                           1, stop=X_set[:, 1].max() +
                                                           1, step=0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np. unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt. title(' LOGISTIC(Test set)')
plt. xlabel(' Age')
plt. ylabel(' Estimated Salary')
plt. legend()
plt. show()
```
![image](https://github.com/LiQianqian123/hello-world/blob/master/Figure_1_6.png)
![image](https://github.com/LiQianqian123/hello-world/blob/master/Figure_2_6.png)
