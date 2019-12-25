#coding:UTF-8
'''椭球面'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
a,b,c= 5.0,25.0,7.0
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = a * np.outer(np.cos(u), np.sin(v))
y = b * np.outer(np.sin(u), np.sin(v))
z = c * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the surface
ax.plot_surface(x, y, z, color='b',cmap=cm.coolwarm)
'''
cset = ax.contourf(x, y, z, zdir='x', offset=-2*a, cmap=cm.coolwarm)
cset = ax.contourf(x, y, z, zdir='y', offset=1.8*b, cmap=cm.coolwarm)
cset = ax.contourf(x, y, z, zdir='z', offset=-2*c, cmap=cm.coolwarm)
'''
ax.set_xlabel('X')
ax.set_xlim(-2*a, 2*a)
ax.set_ylabel('Y')
ax.set_ylim(-1.8*b, 1.8*b)
ax.set_zlabel('Z')
ax.set_zlim(-2*c, 2*c)
plt.grid()
plt.axhline(y=0.8, ls='--', c='r')
plt.show()


fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0, 10, 1000)

for i in range(4):
    lines += ax.plot(x, np.sin(x - i * np.pi / 2),styles[i], color='black')
ax.axis('equal')

# 设置第一组标签
ax.legend(lines[:2], ['line A', 'line B'],
          loc='upper right', frameon=False)

# 创建第二组标签
from matplotlib.legend import Legend
leg = Legend(ax, lines[2:], ['line C', 'line D'],
             loc='lower right', frameon=False)
ax.add_artist(leg)
plt.show()


for i in range(1, 7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2, 3, i)),fontsize=18, ha='center')
plt.show()

mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

# Set up the axes with gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

# scatter points on the main axes
main_ax.scatter(x, y,s=3,alpha=0.2)

# histogram on the attached axes
x_hist.hist(x, 40, histtype='stepfilled',
            orientation='vertical')
x_hist.invert_yaxis()

y_hist.hist(y, 40, histtype='stepfilled',
            orientation='horizontal')
y_hist.invert_xaxis()
plt.show()
import pandas as pd
dates = pd.date_range('today',periods=6) # 定义时间序列作为 index
num_arr = np.random.randn(6,4) # 传入 numpy 随机数组
columns = ['A','B','C','D'] # 将列表作为列名
df = pd.DataFrame(num_arr, index = dates, columns = columns)
print(df)