import numpy as np

#欧氏距离源自欧氏空间中两点间的直线距离，是最常见的一种距离计算方式。
# 计算公式是两个矩阵中对应元素之差的平方和再开方。
def euclidean_distance(X, Y):
     x=X
     y=Y
     euclidean_distance = np.sqrt(np.sum(np.square(x - y)))

     return euclidean_distance
#曼哈顿距离的计算公式是两个矩阵中对应元素差的绝对值之和。
def manhattan_distance(X,Y):
    x = X
    y = Y
    manhattan_distance = np.sum(np.abs(x - y))
    return manhattan_distance

#标准化欧氏距离是对欧氏距离的改进，将数据各维的分量都归一化到均值和方差相等。
# 标准化欧氏距离也可以看成是一种加权欧氏距离。
def standardized_euclidean_distance(X,Y):
    x = X
    y = Y
    Z = np.vstack([x, y])

    sk = np.var(Z, axis=0, ddof=1)
    standardized_euclidean_distance = np.sqrt(((x - y) ** 2 / sk).sum())
    print(standardized_euclidean_distance)
    return standardized_euclidean_distance

#coding:utf-8




