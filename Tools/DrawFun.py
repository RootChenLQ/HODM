#coding:utf-8
from mpl_toolkits import mplot3d
# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
def draw3D(df,label):
    
    assert(len(df) == len(label)),"label size unenqual!"
    cmap_list = ['r','g','b','c']
    cmap = mpl.colors.ListedColormap(cmap_list)
    cmap.set_over(cmap_list[0])
    cmap.set_under(cmap_list[-1])

    ax = plt.axes(projection='3d')
    index = df.columns.values
    x = df[index[0]]
    y = df[index[1]]
    z = df[index[2]]
    #三维线的数据
    #x,y,z
    #ax.plot3D(xline, yline, zline, 'gray')
    color = [0.8 if lab == 0 else 0.1 for lab in label]
    ax.scatter(x, y, z, c=color, cmap = cmap_list)
    plt.show()

if __name__ == "__main__":
    pass