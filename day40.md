```python
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "C:/python_pycharm/venv/PetImages"
CATEGORIES = ["Dog","Cat"]

for category in CATEGORIES:
    path = os.path.join(DATADIR,category)  #创建路径
    for img in os.listdir(path):   #迭代遍历每个照片
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)   #转化为array
        plt.imshow(img_array, cmap='gray')  #转化为图像展示
        plt.show() #display!
        break
    break

#看一下array中存储的图像数据
print (img_array)
#看一下array的形状
print (img_array.shape)

IMG_SIZE = 50
new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
plt.imshow(new_array,cmap='gray')
plt.show()

IMG_SIZE = 150
new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
plt.imshow(new_array,cmap='gray')
plt.show()

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category) #得到分类，其中0=dog 1=cat
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE)) #大小转换
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
            #except OSError as e:
            #    print（"OSErrroBad img most likely",e,os.path.join(path,img)）
            #except Exception as e:
            #    print("general exception",e,os.path.join(path,img))
create_training_data()
print(len(training_data))

import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)
print(X[0].reshape(-1,IMG_SIZE,IMG_SIZE,1))
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
# 让我们保存这些数据，这样我们就不需要每次想要使用神经网络模型时继续计算它：
import pickle

pickle_out = open("C:/python_pycharm/venv/PetImages/X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("C:/python_pycharm/venv/PetImages/Y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
# We can always load it in to our current script, or a totally new one by doing:

pickle_in = open("C:/python_pycharm/venv/PetImages/X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("C:/python_pycharm/venv/PetImages/y.pickle","rb")
y = pickle.load(pickle_in)
```
