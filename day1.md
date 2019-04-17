# Step 1:导入库
#numpy : 是Python的一个程序扩展库，用来存储和处理大型矩阵  
#pandas : 是Python的一个数据分析包，提供大量处理数据的函数及方法  
```python  
  import numpy as np  
  import pandas as pd  
```
# Step 2 : 导入数据集  
#pd.read_csv()的作用是将csv文件读入并转化为数据框形式  
#iloc 方法：通过行号获取行数据，不能是字符（输出所有第0列到倒数第二列的内容  
#[a]表示第a行或列,-1表示从左边第一个开始，每次向右边开始读，读到左边显示数的前一个
```python  
  dataset = pd.read_csv('C:\data\data.csv')  
  X = dataset.iloc[ : , :-1].values  
  Y = dataset.iloc[ : , 3].values  # 取第三行所有数据
```
# Step 3 : 处理丢失的数据  
#用Imputer类对缺失数据进行处理，在Imputer类种mean是均值，median是中值，most_frequent是最大值  
#sklearn processing 的作用 : 填补缺失值  
#missing values: 缺失值 ，python中缺失的默认值为'NaN'表示(not a number)  
```python  
  from sklearn.preprocessing import Imputer  
  imputer = Imputer(missing_values="NaN", strategy="mean",axis=0)  
  imputer = imputer.fit(X[ : , 1:3])  # 用数据拟合fit X的前两列  
  X[ : , 1:3] = imputer.transform(X[ : , 1:3])  # 计算缺失数据  
```
# step 4 :编码分类数据  
#sklearn.preprocessing import labelencoder : 标准化标签，将标签值统一换成range(标签个数-1)范围内，将变量编码成能运算的数值  
#onehotencoder: 独热编码，为了处理距离的量度，认为每个类别之间的距离是一样的，将类别与向量对应  
```python  
  from sklearn.preprocessing import LabelEncoder,OneHotEncoder  
  labelencoder_X = LabelEncoder()  
  X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])  
```  
## 创建虚拟变量  
#Dummy Variable（虚拟变量），用以反映质的属性的人工变量，是量化了的质变量，通常取值为0或1  
```python  
  onehotencoder = OneHotEncoder(categorical_features = [0])  
  X = onehotencoder.fit_transform(X).toarray()
  labelencoder_Y = LabelEncoder()
  Y = labelencoder_Y.fit_transform(Y)  
```
# step 5: 将数据分成训练集和测试集  
#测试集的比重一般为0.2或0.25，0是指测试集没有数据，是指全部都是测试集  
#令random_state为0，得到一样的训练集和测试集  
```python  
  from sklearn .model_selection import train_test_split  
  X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size = 0.20,random_state = 0)  
```  
#print(X_train)  
#print(X_test)  
# step 6:功能缩放  
#对数据进行特征缩放  
```python  
  from sklearn.preprocessing import StandardScaler  
  sc_X = StandardScaler()  # 类里面的对象  
  X_train = sc_X.fit_transform(X_train)  # 拟合后转换  
  X_test = sc_X.fit_transform(X_test)  # 不需要再拟合，y表示某类不需要特征缩放  
```




