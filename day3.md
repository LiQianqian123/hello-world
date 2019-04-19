# step 1:数据预处理   
## 导入库  
```python
import pandas as pd
import numpy as np  
```
## 导入数据集  
```python
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,4].values
# print(X)
# print(Y)  
```
## 将类别数据数字化  
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labelencoder = LabelEncoder()
# 对第0列到第3列所有行进行labelencoder编码
X[: , 3] = labelencoder.fit_transform(X[: , 3])
# 仅对第3列进行onehotencoder编码
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()  
```
## 躲避虚拟变量陷阱  
```python
# 虚拟变量陷阱指两个或多个变量之间高度相关的情形。简单的说就是一个变量可以通过其它变量预测。例如男性类别可以通过女性类别判断（女性值为 0 时，表示男性），所以对于男女问题，变量应该只有一元。
# 假设有 m 个相互关联的类别，为了解决虚拟变量陷阱，可以在模型构建时只选取 m-1 个虚拟变量。
# 此处去掉了第0列
X = X[: ,1:]  
```
## 拆分数据集为训练集和测试集  
```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2,random_state = 0)  
```  
# step 2  : 在训练集上训练多元线性回归模型  
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)  
```
# step 3:在测试集上预测结果  
```python
y_pred = regressor.predict(X_test)
print(y_pred)  
```


