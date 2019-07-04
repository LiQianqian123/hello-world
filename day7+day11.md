# K近邻法 
# 含义
```python  
1.给出训练数据集，根据给定的距离度量在训练集中找到与实例特征向量x最邻近的K个点，涵盖这k个点的邻域记为N(kx)  
2.在N(kx)中根据分类决策规则(如多数表决)决定实例的输出(实例的类别)  
```
## 三要素  
### 1.距离度量  
```python  
距离度量用实例向量与第i个预测集向量的P范数来表示，P的一般取值为：P=1，P=2(欧式距离)，P=正无穷  
其中无穷范数是指一个向量里面绝对值最大的那个元素的绝对值  
```
### 2.K值的选择  
```python  
交叉验证法:将整个训练集分成训练集和验证集，拿出验证集里的实例给出这个实例对应的向量，然后在训练集上取出不同的K值，用K近邻模型来看这个实例对应的类别，因为已知验证集中该实例的真实类别，，将两种类别进行对比就能得到合适的K值  
```  
### 3.分类决策规则  
```python
一般为多数表决法，即在最邻近的K个点里属于哪一类别的点最多，那么这个实例就是这个类别。
多数表决法的经验风险(本质)：[I(1)+I(2)+I(3)+....+I(k)]/k;假定邻域内所有点的类别是红色，那么I表示邻域内的点不是红色的概率; 同时假定邻域内所有点的类别是红色和蓝色，分别算出它们的损失，损失小的就是该邻域内所有点的类别。这种方法就是多数表决法；  
```  
# 程序实现
## 导入相关库  
```python
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
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
## 使用K-NN对训练集数据进行训练
```python
# 构建了一个NeighborsClassifier类
# n_neighbors表示k的值，即邻域内点的个数；minkowski度量的幂参数，当P=2时，等于标准的欧几里德度量
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
```
# 对测试集进行预测
```python
y_pred = classifier.predict(X_test)
```
## 生成混淆矩阵
```python
cm = confusion_matrix(y_test, y_pred)
```
