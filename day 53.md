```python
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')   #给项目传递关键字“3d”
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-1.png)
```python
ax = plt.axes(projection='3d')
#线数据
# Data for a three-dimensional line
zline = np.linspace(0, 15, 1000)  #画网格，0-15均分成1000份
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')
#点数据
# Data for three-dimensional scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')  #画三维点图
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-2.png)
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-3.png)
```python
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show ()
ax.view_init(60, 35)
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-4.png)
```python
fig
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
plt.show()

ax.set_title('wireframe')
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-5.png)
```python
r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
r, theta = np.meshgrid(r, theta)
X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X, Y)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-6.png)
```python
theta = 2 * np.pi * np.random.random(1000)
r = 6 * np.random.random(1000)
x = np.ravel(r * np.sin(theta))
y = np.ravel(r * np.cos(theta))
z = f(x, y)
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-7.png)
```python
ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z,
                cmap='viridis', edgecolor='none')
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-8.png)
```python
theta = np.linspace(0, 2 * np.pi, 30)
w = np.linspace(-0.25, 0.25, 8)
w, theta = np.meshgrid(w, theta)
phi = 0.5 * theta
# radius in x-y plane
r = 1 + w * np.cos(phi)
x = np.ravel(r * np.cos(theta))
y = np.ravel(r * np.sin(theta))
z = np.ravel(w * np.sin(phi))
# triangulate in the underlying parametrization
from matplotlib.tri import Triangulation
tri = Triangulation(np.ravel(w), np.ravel(theta))

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, triangles=tri.triangles,
                cmap='viridis', linewidths=0.2)

ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
plt.show()
```
![Image text](https://github.com/LiQianqian123/hello-world/blob/master/Figure_53-9.png)
