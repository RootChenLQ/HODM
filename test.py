#coding:utf-8
'''
import pandas as pd
d1 = pd.DataFrame([1,2,3])
b = d1.copy()
#print(b.head())
#print(d1.head())
b[0][1]=3
#print(b.head())
#print(d1.head())


#read yaml
import yaml
_PARAMS_PATH = "params.yaml"

with open(_PARAMS_PATH, "r") as f:
  modelParams = yaml.safe_load(f)

modelParams
#write yaml
print(modelParams['CHParams']['member_size'])

modelParams['CHParams']['member_size']=10000

with open("out.yaml","w") as outoperation:
    yaml.dump(modelParams,outoperation,default_flow_style=False,allow_unicode=True,indent=4)


import Fun
tp = 290
tn = 90
fp = 10
fn = 10
kapp = Fun.getKappaScore(tp,tn,fp,fn)
print(kapp)
'''
'''
import numpy as np
a = np.array([0,0,0])

for i in range(1,10):
  a = np.vstack((a,np.array([i,i,i])))

print(a)
# 排序np矩阵的数据。
b = np.mean(a,axis=0)
print(b)
delta = abs(a-b)
print(delta)
delta = delta*delta
print(delta)
sum = np.sum(delta,axis=1)
sum = np.sort(sum)
#sum = np.unique(sum)  #unique sum in order 
print(sum)
dataSize =  a.shape[0]
pick = int(dataSize*0.3)
print(pick)
thres = sum[-pick]
print(thres)

#b = b/a.size
print('Hello')
'''

'''
import pandas as pd
a = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],columns = ['x','y','z'])
print(a)
#a.drop(index=0,inplace=True)
a = a.reset_index(drop=True)
print(a)
a.loc[(a['x']==1),'x'] +=1 
print(a)

##ROC  曲线
from sklearn import metrics
from sklearn.metrics import auc 
import numpy as np
y = np.array([1, 1, 2, 2])  
scores = np.array([0.1, 0.4, 0.35, 0.8])  
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
print(metrics.auc(fpr, tpr) )
'''

l = [1,2,3]
if 1 in l:
  print('in')
l1 = [i for i in  range(0,10)]
l = l+l1
print(l)