import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats
import Structure
import pandas as pd 
from Tools.InsertNoise import *
import yaml
import Fun
from scipy import stats
from sklearn import preprocessing
import time

def run(fileNo):
    exp_str = 'E'+str(fileNo)
    Output_DF = Structure.Output_DF.copy() 
    #Output_DF.to_csv('LOF'+exp_str+'.csv',header=0,mode='a')
    #加载实验参数
    times = 0
    update_times = 0
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(exp_str+'.csv',mode='a')

    _PARAMS_PATH = "params.yaml"
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    repeat_time = modelParams['CommonParams']['repeat_time']
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str]
    # Presition:
    #presition_dic = modelParams['Presition']
    #common
    attributes = modelParams['CommonParams']['attributes']  # 3
    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate']) 
    anomalyRate = 0.05
    continueErrorThres = modelParams['CommonParams']['continueErrorThres'] 
    #pos_buffer_size = modelParams['CommonParams']['pos_buffer_size']  #
    datasize = modelParams['CommonParams']['datasize']
    datasize = 6000
    statistic_analysis_data_size = modelParams['CommonParams']['statistic_analysis_data_size']   
    # MN
    bufferSize = modelParams['SNParams']['buffer_size']  #4.23 
    #bufferSize = 2000
    #sqrt_thres = modelParams['SNParams']['sqrt_thres']
    store_pro = modelParams['SNParams']['store_pro']
    sampleforK = (int)(modelParams['SNParams']['sample_size_rate']*bufferSize)
    #Sink 
    member_size = modelParams['CHParams']['member_size'] #三个节点

    anomalyType = modelParams['anomaly_type']    
    
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    try:
            ##读取txt 文件
            #print('data read success')
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')


    # 开始代码
    while times < repeat_time:
        #repeat time
        for typeName in anomalyType:
            #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]:
                #anomaly type [0,1,2]
                timestamp = time.time()
                type =  anomalyType[typeName][subtype]  # 加载异常
                typeName = 'noise'
                type = [0]
                #获取数据，插入异常
                print(typeName+subtype)
                begin_ = 2000
                end_ = begin_ + datasize
                 #traindata = dataframe1[:begin_]
                #print(typeName+subtype)
                data1 = dataframe1[0:end_].copy()
                data1 = data1.reset_index(drop=True)
                #
                train_data = data1[data1.index.values%4==0].copy()
                test_data = data1[data1.index.values%4>0].copy()
                train_data = train_data.reset_index(drop=True)
                test_data = test_data.reset_index(drop=True)
                
                #插入异常
                outlier_pos1 = []
                anomaly_num = (int)((datasize)*anomalyRate)
                test_data,outlier_pos1  = insert_anomaly(test_data, 0, anomaly_num, typeName, 
                                                        type, delta_mean = 0.5, delta_std_times = 1.5)
                detected_list = []
                #data1_scale = preprocessing.scale(data1)
                
                
                # fit the model
                #LocalOutlierFactor 参数
                #n_neighbors=20, algorithm=’auto’, leaf_size=30, 
                #metric=’minkowski’, p=2, metric_params=None, contamination=0.1, n_jobs=1
                #clf = LocalOutlierFactor(n_neighbors=70, contamination=anomalyRate)  # 加载LOF分类器
                #(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, 
                # n_jobs=None, behaviour='deprecated', random_state=None, verbose=0, warm_start=False)[source]
                #clf = IsolationForest(n_estimators=200,contamination=0.05,random_state=0).fit(train_data)
                n_estimators_ = [(int)(val) for val in np.linspace(1,100,10)]
                #nus = [nu for nu in np.linspace(0.01,0.2,20)]
                max_n = 0
                max_f1 = 0
                for n_ in n_estimators_:
                    print('n_estimators_',n_)
                    detected_list = []
                    clf = IsolationForest(n_estimators=10,contamination=0.1,random_state=0).fit(train_data)
                    
                
                    
                    result = clf.predict(test_data)
                    
                    for i in range(len(result)):
                        if result[i]==-1:
                            detected_list.append(i)
                        else:
                            pass
                    #print(count)
                    tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,detected_list,datasize)
                    if f1 > max_f1:
                        max_f1 = f1
                        max_n = n_
                    
                    s1 = pd.Series([exp_str,i, typeName,type, tn,fn,fp,tp,acc,fpr,tpr,p,f1,update_times,time.time()-timestamp],
                                index= Structure.Output_DF_Type)
                    #Output_DF = Structure.Output_DF.copy() 
                    #Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
                    #Output_DF.to_csv('LOF'+exp_str+'.csv',header=0,mode='a')
                print(max_n,max_f1)
        times+=1

if __name__ == "__main__":
    X = [[-1.1], [0.3], [0.5], [100]]
    clf = IsolationForest(random_state=0).fit(X)
    clf.predict([[0.1], [0], [90]])
    run(1)
