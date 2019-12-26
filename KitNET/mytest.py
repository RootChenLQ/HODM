'''
import numpy as np
a = np.random.randint(low=1, high=5, size = (5,1))
b = np.ones((5,))
c = np.outer(a,b)
index = np.triu_indices(3, 1)
#Return the indices for the upper-triangle of an (n, m) array.
print(c)
print(index)
'''
# 相关性计算
'''
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

Z = linkage(X, 'ward')
fig = plt.figure(figsize=(8, 6))
dn = dendrogram(Z)
print(Z)
plt.show()

Z = linkage(X, 'single')
fig = plt.figure(figsize=(8, 6))
dn = dendrogram(Z)
plt.show()
'''


import numpy as np
from scipy.cluster import hierarchy
x = np.random.rand(10).reshape(5, 2)
x_ = x.T
print(x)
print(x_)
Z = hierarchy.linkage(x)
hierarchy.to_tree(Z)

rootnode, nodelist = hierarchy.to_tree(Z, rd=True)
print(rootnode)
print(nodelist)
print(len(nodelist))
rng =  np.random.mtrand.RandomState(1234)
print(x)
t = rng.binomial(size=x.shape,n=2, p= 1 - 0.5) 
t = t * x
print(t)