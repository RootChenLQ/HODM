import os
import sys
sys.path.append("../")

import KitNET as kit
import numpy as np
import pandas as pd
import time
#import matplotlib.pyplot as plt
import os
import sys
import yaml
#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".."))+"Tools")
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
import Fun
import Tools.InsertNoise
import Structure
import sklearn.metrics
def run_kitnet(exp_str,maxAE,FMgrace,ADgrace,data,times,typeName,type,anomalyDatasize):

    print(typeName,type)
    #插入异常
    X,outlier_pos1 = Tools.InsertNoise.insert_anomaly(data, FMgrace+ADgrace, 
                    anomalyDatasize, error_type=typeName, type_l=type, delta_mean = 2, delta_std_times = 1.5)
    testDataSize = data.shape[0] - ADgrace - FMgrace
    X.to_csv('noisedata.csv')
    #程序运行时间
    timestamp = time.time() 
    #构造kitten
    #K = kit.KitNET(X.shape[1],max_autoencoder_size= 10,FM_grace_period=FMgrace,AD_grace_period=ADgrace,learning_rate=0.1,hidden_ratio=2/3, feature_map = [[0,1],[0,1,2]])
    K = kit.KitNET(X.shape[1],max_autoencoder_size= 10,FM_grace_period=FMgrace,
                    AD_grace_period=ADgrace,learning_rate=0.1,hidden_ratio=2/3, feature_map = [[0,1],[1,0],[0,1,2],[1,0,2]])
    #K = kit.KitNET(X.shape[1],max_autoencoder_size= 10,FM_grace_period=FMgrace,
    #                AD_grace_period=ADgrace,learning_rate=0.1,hidden_ratio=2/3)
    
    RMSEs = np.zeros(X.shape[0]) # a place to save the scores
    #检测
    for i in range(X.shape[0]):
        if i % 6000 == 0:
            print(i)
        RMSEs[i] = K.process(X.loc[i]) #will train during the grace periods, then execute on all the rest.
    detected_l = []
    #输出rmses
    
    for i in range(len(RMSEs)):
        if RMSEs[i]>1:
            detected_l.append(i)
    # detected list
    d_l =  [1 if val >0 else 0 for val in RMSEs]
    # label list
    label = np.zeros(len(d_l))
    for index in outlier_pos1:
        label[index] = 1
    #计算 tpr fpr
    outputData = np.vstack((label,RMSEs))
    outputData  = outputData.T
    fpr1,tpr1,thres = sklearn.metrics.roc_curve(label,RMSEs,pos_label=1)
    auc = sklearn.metrics.auc(fpr1,tpr1)
    print(auc)
    data_ = pd.DataFrame(outputData)
    data_.to_csv('RMSEs.csv')
    tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,detected_l,testDataSize) 
    update_times = 0
    s1 = pd.Series([exp_str,i, typeName,type, tn,fn,fp,tp,acc,fpr,tpr,p,f1,update_times,time.time()-timestamp],
                            index= Structure.Output_DF_Type)
    Output_DF = Structure.Output_DF.copy() 
    Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
    return Output_DF

def run(exp):
    exp_str = 'E'+str(exp)  #实验名 E0、E1、E2
    #output performance
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(exp_str+'Kitnet.csv',mode='a')
    Parent_PATH = ".."
    _PARAMS_PATH = os.path.join(Parent_PATH,"params2.yaml")
    #_PARAMS_PATH = "params2.yaml"  #读取参数文件
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)

    repeat_time = modelParams['CommonParams']['repeat_time']  #重复实验次数

    Filled_DF_Type = ['Temperature','Humidity','Voltage']     #intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    
    datafile_dic =  modelParams['datafile'][exp_str] # anomaly parameters
    
    #attributes = modelParams['CommonParams']['attributes']  # 属性维度

    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate'])  #插入异常比例
    #nomalyRate = 0.01
    datasize = modelParams['CommonParams']['datasize']  #总数据量 30000
    #datasize = 30000
    maxAE = 10  #maximum size for any autoencoder in the ensemble layer 编码器的个数
    FMgrace = 0 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = 10000 #the number of instances used to train the anomaly detector (ensemble itself)
    #bufferSize = modelParams['SNParams']['buffer_size']  #4.23 缓存大小
    #bufferSize = 2000 #预先训练数据量

    anomaly_num = (int)((datasize - FMgrace-ADgrace)*anomalyRate) #异常数量
    anomalyType = modelParams['anomaly_type']   #异常名，异常类型词典 
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    try:
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    times = 0           
    while times < repeat_time: #重复实验数
        for typeName in anomalyType: #异常类型  #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]: #0：T 1：H 2：V
                typeName = 'outlier'
                type = [0]
                type =  anomalyType[typeName][subtype] # 异常类型 list
                data1 = dataframe1[0:datasize].copy()  # 复制原数据
                df = run_kitnet(exp_str,maxAE,FMgrace,ADgrace,data1,times,typeName,type,anomaly_num)
                df.to_csv(exp_str+'Kitnet.csv',header=0,mode='a')
        times+=1


def TrainOnceAndRun(exp):
    exp_str = 'E'+str(exp)  #实验名 E0、E1、E2
    #output performance
    outputFileName = exp_str+'Kitnet.csv'
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(outputFileName,mode='a')
    Parent_PATH = ".."
    _PARAMS_PATH = os.path.join(Parent_PATH,"params2.yaml")
    #_PARAMS_PATH = "params2.yaml"  #读取参数文件
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)

    repeat_time = modelParams['CommonParams']['repeat_time']  #重复实验次数

    Filled_DF_Type = ['Temperature','Humidity','Voltage']     #intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    
    datafile_dic =  modelParams['datafile'][exp_str] # anomaly parameters
    
    #attributes = modelParams['CommonParams']['attributes']  # 属性维度

    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate'])  #插入异常比例
    #nomalyRate = 0.01
    datasize = modelParams['CommonParams']['datasize']  #总数据量 30000
    #datasize = 4000
    maxAE = 10  #maximum size for any autoencoder in the ensemble layer 编码器的个数
    FMgrace = 0 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = 10000 #the number of instances used to train the anomaly detector (ensemble itself)
    #bufferSize = modelParams['SNParams']['buffer_size']  #4.23 缓存大小
    #bufferSize = 2000 #预先训练数据量

    anomaly_num = (int)((datasize - FMgrace-ADgrace)*anomalyRate) #异常数量
    anomalyType = modelParams['anomaly_type']   #异常名，异常类型词典 
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    try:
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    times = 0         
    #预训练模型
    K = kit.KitNET(dataframe1.shape[1],max_autoencoder_size= 10,
        FM_grace_period = FMgrace,AD_grace_period=ADgrace,learning_rate=0.1,hidden_ratio=2/3, feature_map = [[0,1],[1,0],[0,1,2],[1,0,2]])
    for i in range(FMgrace+ADgrace):
        K.train(dataframe1.loc[i])  
    # 完成训练
    while times < repeat_time: #重复实验数
        for typeName in anomalyType: #异常类型  #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]: #0：T 1：H 2：V
                #typeName = 'outlier'
                #type = [0]
                type =  anomalyType[typeName][subtype] # 异常类型 list
                data1 = dataframe1[0:datasize].copy()  # 复制原数据
                #df = run_kitnet(exp_str,maxAE,FMgrace,ADgrace,data1,times,typeName,type,anomaly_num)
                print(typeName,type)
                #插入异常
                data1,outlier_pos1 = Tools.InsertNoise.insert_anomaly(data1, FMgrace+ADgrace, 
                                anomaly_num, error_type=typeName, type_l=type)
                testDataSize = data1.shape[0] - ADgrace - FMgrace
                data1.to_csv('noisedata.csv')
                #程序运行时间
                timestamp = time.time() 
                #构造kitten
                #K = kit.KitNET(X.shape[1],max_autoencoder_size= 10,FM_grace_period=FMgrace,AD_grace_period=ADgrace,learning_rate=0.1,hidden_ratio=2/3, feature_map = [[0,1],[0,1,2]])
                #K = kit.KitNET(X.shape[1],max_autoencoder_size= 10,FM_grace_period=FMgrace,
                #                AD_grace_period=ADgrace,learning_rate=0.1,hidden_ratio=2/3, feature_map = [[0,1],[1,0],[0,1,2],[1,0,2]])
                #K = kit.KitNET(X.shape[1],max_autoencoder_size= 10,FM_grace_period=FMgrace,
                #                AD_grace_period=ADgrace,learning_rate=0.1,hidden_ratio=2/3)
                
                RMSEs = np.zeros(data1.shape[0]) # a place to save the scores
                #检测
                for i in range(FMgrace+ADgrace,data1.shape[0]):
                    if i % 6000 == 0:
                        print(i)
                    RMSEs[i] = K.execute(data1.loc[i]) #will train during the grace periods, then execute on all the rest.
                detected_l = []
                #输出rmses
                
                for i in range(len(RMSEs)):
                    if RMSEs[i]>1:
                        detected_l.append(i)
                # detected list
                d_l =  [1 if val >0 else 0 for val in RMSEs]
                # label list
                label = np.zeros(len(d_l))
                for index in outlier_pos1:
                    label[index] = 1
                #计算 tpr fpr
                #outputData = np.vstack((label,RMSEs))
                #outputData  = outputData.T
                #fpr1,tpr1,thres = sklearn.metrics.roc_curve(label,RMSEs,pos_label=1)
                #auc = sklearn.metrics.auc(fpr1,tpr1)
                #print(auc)
                #data_ = pd.DataFrame(outputData)
                #data_.to_csv('RMSEs.csv')
                tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,detected_l,testDataSize) 
                update_times = 0
                s1 = pd.Series([exp_str,i, typeName,type, tn,fn,fp,tp,acc,fpr,tpr,p,f1,update_times,time.time()-timestamp],
                                        index= Structure.Output_DF_Type)
                Output_DF = Structure.Output_DF.copy() 
                Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
                Output_DF.to_csv(outputFileName,header=0,mode='a')
        times+=1

if __name__ == "__main__":
    for i in range(3,4):
        TrainOnceAndRun(i)