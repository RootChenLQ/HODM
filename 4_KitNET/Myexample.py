#coding:utf-8
import os
import sys
sys.path.append("../")

import KitNET as kit
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import InsertNoise
import os
import sys

#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".."))+"Tools")
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
import Fun
import Tools.InsertNoise


'''
import os
print '***获取当前目录***'
print os.getcwd()
print os.path.abspath(os.path.dirname(__file__))
# __file__ 为当前文件, 若果在ide中运行此行会报错,可改为 #d = path.dirname('.') 
# 但是改为.后，就是获得当前目录，接着使用dirname函数访问上级目录
print '***获取上级目录***'
print os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print os.path.abspath(os.path.dirname(os.getcwd()))
print os.path.abspath(os.path.join(os.getcwd(), ".."))
print '***获取上上级目录***'
print os.path.abspath(os.path.join(os.getcwd(), "../.."))
'''

##############################################################################
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates KitNET's ability to incrementally learn, and detect anomalies.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 2.7.14   #######################

# Load sample dataset (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...
'''
print("Unzipping Sample Dataset...")
import zipfile
with zipfile.ZipFile("dataset.zip","r") as zip_ref:
    zip_ref.extractall()

print("Reading Sample dataset...")
'''
# KitNET params:
maxAE = 10  #maximum size for any autoencoder in the ensemble layer
FMgrace = 1000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 5000 #the number of instances used to train the anomaly detector (ensemble itself)


repeat_time = modelParams['CommonParams']['repeat_time']
# load data
X = pd.read_csv("datasets/E0/node43.csv",header=None) #an m-by-n dataset with m observations
X = X[0:30000].copy()
testDataSize = X.shape[0] - ADgrace - FMgrace
anomalyRate = 0.05
anomalyDataSize = (int)(testDataSize * anomalyRate)
#X,outlier_pos1  = InsertNoise.insert_noise_error(X,[0],FMgrace+ADgrace,300,0.5,1.5)

X,outlier_pos1 = Tools.InsertNoise.insert_anomaly(X, FMgrace+ADgrace, anomalyDataSize, error_type='constant', type_l=[], delta_mean = 2, delta_std_times = 1.5)
label = np.zeros(X.shape[0])
for index in outlier_pos1:
    label[index] = 1
#np.to_csv(outlier_pos1,'outlier.csv')
label.tofile('label.csv')

# Build KitNET
#(self,n,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75, feature_map = None):
K = kit.KitNET(X.shape[1],max_autoencoder_size= 10,FM_grace_period=FMgrace,
                    AD_grace_period=ADgrace,learning_rate=0.1,hidden_ratio=2/3, feature_map = None)
RMSEs = np.zeros(X.shape[0]) # a place to save the scores


print("Running KitNET:")
start = time.time()
# Here we process (train/execute) each individual observation.
# In this way, X is essentially a stream, and each observation is discarded after performing process() method.
for i in range(X.shape[0]):
    if i % 6000 == 0:
        print(i)
    RMSEs[i] = K.process(X.loc[i]) #will train during the grace periods, then execute on all the rest.



#detected_l  = [index for index in range(RMSEs.shape[0]) if RMSEs[index]>1 ]
detected_l = []
for i in range(RMSEs.shape[0]):
    if RMSEs[i]>1:
       detected_l.append(i)
    
print(len(outlier_pos1))
print(len(detected_l))

tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,detected_l,testDataSize) 
                        

outputData = np.vstack((label,RMSEs))
outputData  = outputData.T
data_ = pd.DataFrame(outputData)
data_.to_csv('RMSEs.csv')


#data_ = pd.DataFrame(np.array(outlier_pos1))
#data_.to_csv('outlier_pos1.csv')
'''
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))
x = [i for i in range(len(RMSEs)-ADgrace-FMgrace)]
# RMES> 0.3
for i in range(len(x)):
    if i in outlier_pos1:
        plt.plot(x[i],RMSEs[ADgrace+FMgrace+i],c='r')
    else:
        plt.plot(x[i],RMSEs[ADgrace+FMgrace+i])
plt.show()
'''
# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
'''
from scipy.stats import norm
beginSample = np.log(RMSEs[FMgrace+ADgrace:])
logProbs = norm.logsf(np.log(RMSEs[FMgrace+ADgrace:]), np.mean(beginSample), np.std(beginSample))

plt.plot(x,logProbs)
plt.show()
'''
#fig = plt.scatter(x,RMSEs[FMgrace+ADgrace:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')


# plot the RMSE anomaly scores
'''
print("Plotting results")
from matplotlib import pyplot as plt
from matplotlib import cm
plt.figure(figsize=(10,5))
timestamps = pd.read_csv("KitNET/mirai3_ts.csv",header=None).as_matrix()
fig = plt.scatter(timestamps[FMgrace+ADgrace+1:],RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
plt.yscale("log")
plt.title("Anomaly Scores from KitNET's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time elapsed [min]")
plt.annotate('Mirai C&C channel opened [Telnet]', xy=(timestamps[71662],RMSEs[71662]), xytext=(timestamps[58000],1),arrowprops=dict(facecolor='black', shrink=0.05),)
plt.annotate('Mirai Bot Activated\nMirai scans network for vulnerable devices', xy=(timestamps[72662],1), xytext=(timestamps[55000],5),arrowprops=dict(facecolor='black', shrink=0.05),)
figbar=plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.show()'''