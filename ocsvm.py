import numpy as np
import matplotlib.pyplot as plt
#from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from scipy import stats
import Structure
import pandas as pd 
from Tools.InsertNoise import *
import yaml
import Fun
from scipy import stats
from sklearn import preprocessing
import Queue
import matplotlib.pyplot as plt
def run_(fileNo):
    exp_str = 'E'+str(fileNo)
    #加载实验参数
    
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(exp_str+'.csv',mode='a')

    _PARAMS_PATH = "params.yaml"
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    repeat_time = (int)(modelParams['CommonParams']['repeat_time'])
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str]
    # Presition:
    #presition_dic = modelParams['Presition']
    #common

    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate']) 
    anomalyRate = 0.05

    #pos_buffer_size = modelParams['CommonParams']['pos_buffer_size']  #
    datasize = modelParams['CommonParams']['datasize']
    datasize = 6000
    # MN
    #bufferSize = modelParams['SNParams']['buffer_size']  #4.23 
    bufferSize = 2000
    #sqrt_thres = modelParams['SNParams']['sqrt_thres']
    
    anomalyType = modelParams['anomaly_type']    
    '''
    # 构造训练样本
    n_samples = 200  #样本总数
    outliers_fraction = 0.25  #异常样本比例
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(n_inliers // 2, 2)
    X_train = np.r_[X + 2, X - 2]   #正常样本
    X_train = np.r_[X_train, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]  #正常样本加上异常样本
    '''
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    try:
            ##读取txt 文件
            #print('data read success')
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')

    times = 0
    # 开始代码
    while times < repeat_time:
        #repeat time
        for typeName in anomalyType:
            #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]:
                #anomaly type [0,1,2]
                type =  anomalyType[typeName][subtype]  # 加载异常
                typeName = 'normal'
                type = []
                #获取数据，插入异常
                begin_ = 2000
                end_ = begin_ + datasize  # 8000
                #traindata = dataframe1[:begin_]
                print(typeName+subtype)
                data1 = dataframe1[begin_:end_].copy()
                data1 = data1.reset_index(drop=True)
                #插入异常
                outlier_pos1 = []
                anomaly_num = (int)((datasize)*anomalyRate)
                data1,outlier_pos1  = insert_anomaly(data1, begin_, anomaly_num, typeName, 
                                                        type, delta_mean = 0.5, delta_std_times = 1.5)

               
                
                data1Scaler = preprocessing.scale(data1)
                count = 0
                #for nu_ in np.linspace(0.1,0.15,100):
                    #print(nu_)
                clf = svm.OneClassSVM(nu=0.105, kernel='rbf', gamma='auto')
                detected_l = clf.fit(data1Scaler)

                #clf.fit_predict(trainDatascaler)
                #clf.predict(data1Scaler)
                count = 0
                for i in range(len(detected_l)):
                    if detected_l[i]==1:
                        count+=1
                print(' count ',count)
def run(fileNo):
    exp_str = 'E'+str(fileNo)
    #加载实验参数
    
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(exp_str+'.csv',mode='a')

    _PARAMS_PATH = "params.yaml"
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    repeat_time = (int)(modelParams['CommonParams']['repeat_time'])
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str]
    # Presition:
    #presition_dic = modelParams['Presition']
    #common

    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate']) 
    anomalyRate = 0.05

    #pos_buffer_size = modelParams['CommonParams']['pos_buffer_size']  #
    datasize = modelParams['CommonParams']['datasize']
    datasize = 6000
    # MN
    #bufferSize = modelParams['SNParams']['buffer_size']  #4.23 
    bufferSize = 2000
    #sqrt_thres = modelParams['SNParams']['sqrt_thres']
    
    anomalyType = modelParams['anomaly_type']    
    '''
    # 构造训练样本
    n_samples = 200  #样本总数
    outliers_fraction = 0.25  #异常样本比例
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(n_inliers // 2, 2)
    X_train = np.r_[X + 2, X - 2]   #正常样本
    X_train = np.r_[X_train, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]  #正常样本加上异常样本
    '''
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    try:
            ##读取txt 文件
            #print('data read success')
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')

    times = 0
    # 开始代码
    while times < repeat_time:
        #repeat time
        for typeName in anomalyType:
            #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]:
                #anomaly type [0,1,2]
                type =  anomalyType[typeName][subtype]  # 加载异常
                typeName = 'outlier'
                type = [0]
                #获取数据，插入异常
                begin_ = 2000
                end_ = begin_ + datasize  # 8000
                #traindata = dataframe1[:begin_]
                print(typeName+subtype)
                data1 = dataframe1[0:end_].copy()
                data1 = data1.reset_index(drop=True)
                #插入异常
                outlier_pos1 = []
                anomaly_num = (int)((datasize)*anomalyRate)
                data1,outlier_pos1  = insert_anomaly(data1, begin_, anomaly_num, typeName, 
                                                        type, delta_mean = 0.5, delta_std_times = 1.5)

               
                trainData = data1[:begin_].copy()
                testData = data1[begin_:].copy()

                scaler = preprocessing.StandardScaler().fit(trainData)
                trainDatascaler = scaler.transform(trainData)
                queue1 = Queue.MNQueue(bufferSize) #存储原始的数据
                for i in range(bufferSize):
                    series = data1.loc[[i]]
                    queue1.enqueue(series)  
                #预训练
                #nus=（0，1）
                #for nu_ in np.linspace(0.001,0.01,9):
                #print(nu_)
                clf = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
                clf.fit(trainDatascaler)
                queue1.clear_count()
                detect_l = []
                
                result = clf.decision_function(trainDatascaler)
                x = [i for i in range(len(result))]
                plt.plot(x,result)
                plt.show()
                count_ = 0
                error_ = []
                for i in range(len(result)):
                    if result[i] > 0:
                        count_ += 1 
                    else:
                        error_.append(i)
                print(count_)
                print(error_)
                min_result = 0
                for row in range(bufferSize,data1.shape[0]):
                    if row%1000==0:
                        print(row)
                    series = data1.loc[[row]]
                    X = np.array(series.values.tolist()[0]).reshape(1,data1.shape[1])
                    X_scaler = scaler.transform(X)
                    #X = np.array(X_scaler)
                    #X = X.reshape(1,data1.shape[1])
                    result = clf.decision_function(X_scaler)
                    if result < min_result:
                        min_result = result
                    if result < -0.1:
                        detect_l.append(row)
                        queue1.enqueue(series)
                    else:
                        queue1.enqueue(series)
                    if queue1.is_trigger_update():
                        queue1.clear_count()
                        scaler = preprocessing.StandardScaler().fit(queue1.df)
                        trainDatascaler = scaler.transform(queue1.df)
                        clf.fit(trainDatascaler)
                count_ = 0
                print('min_result',min_result)
                for index in detect_l:
                    if index in outlier_pos1:
                        count_+=1
                print(len(detect_l))
                print(count_)
                        
        times+=1              
def run_noscalar(fileNo):
    exp_str = 'E'+str(fileNo)
    #加载实验参数
    
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(exp_str+'.csv',mode='a')

    _PARAMS_PATH = "params.yaml"
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    repeat_time = (int)(modelParams['CommonParams']['repeat_time'])
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str]
    # Presition:
    #presition_dic = modelParams['Presition']
    #common

    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate']) 
    anomalyRate = 0.05

    #pos_buffer_size = modelParams['CommonParams']['pos_buffer_size']  #
    datasize = modelParams['CommonParams']['datasize']
    datasize = 6000
    # MN
    #bufferSize = modelParams['SNParams']['buffer_size']  #4.23 
    bufferSize = 2000
    #sqrt_thres = modelParams['SNParams']['sqrt_thres']
    
    anomalyType = modelParams['anomaly_type']    
    '''
    # 构造训练样本
    n_samples = 200  #样本总数
    outliers_fraction = 0.25  #异常样本比例
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(n_inliers // 2, 2)
    X_train = np.r_[X + 2, X - 2]   #正常样本
    X_train = np.r_[X_train, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]  #正常样本加上异常样本
    '''
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    try:
            ##读取txt 文件
            #print('data read success')
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')

    times = 0
    # 开始代码
    while times < repeat_time:
        #repeat time
        for typeName in anomalyType:
            #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]:
                #anomaly type [0,1,2]
                type =  anomalyType[typeName][subtype]  # 加载异常
                typeName = 'noise'
                type = [0]
                #获取数据，插入异常
                begin_ = 2000
                end_ = begin_ + datasize  # 8000
                #traindata = dataframe1[:begin_]
                print(typeName+subtype)
                data1 = dataframe1[0:end_].copy()
                data1 = data1.reset_index(drop=True)
                #插入异常
                outlier_pos1 = []
                anomaly_num = (int)((datasize)*anomalyRate)
                data1,outlier_pos1  = insert_anomaly(data1, begin_, anomaly_num, typeName, 
                                                        type, delta_mean = 0.5, delta_std_times = 1.5)

               
                #trainData = data1[:begin_]
                #testData = data1[begin_:]

                #scaler = preprocessing.StandardScaler().fit(trainData)
                #trainDatascaler = scaler.transform(trainData)
                
                queue1 = Queue.MNQueue(bufferSize) #存储原始的数据
                for i in range(bufferSize):
                    series = data1.loc[[i]]
                    queue1.enqueue(series)  
                #预训练
                #nus=（0，1）
                #for nu_ in np.linspace(0.001,0.01,9):
                #print(nu_)
                clf = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
                clf.fit(trainDatascaler)
                queue1.clear_count()
                detect_l = []

                result = clf.decision_function(trainDatascaler)
                count_ = 0
                error_ = []
                for i in range(len(result)):
                    if result[i] > 0:
                        count_+=1 
                    else:
                        error_.append(i)
                print(count_)
                print(error_)
                min_result = 0
                for row in range(bufferSize,data1.shape[0]):
                    if row%1000==0:
                        print(row)
                    series = data1.loc[[row]]
                    X = np.array(series.values.tolist()[0]).reshape(1,data1.shape[1])
                    X_scaler = scaler.transform(X)
                    #X = np.array(X_scaler)
                    #X = X.reshape(1,data1.shape[1])
                    result = clf.decision_function(X_scaler)
                    if result < min_result:
                        min_result = result
                    if result < -0.1:
                        detect_l.append(row)
                    else:
                        queue1.enqueue(series)
                    if queue1.is_trigger_update():
                        queue1.clear_count()
                        scaler = preprocessing.StandardScaler().fit(queue1.df)
                        trainDatascaler = scaler.transform(queue1.df)
                        clf.fit(trainDatascaler)
                count_ = 0
                print('min_result',min_result)
                for index in detect_l:
                    if index in outlier_pos1:
                        count_+=1
                print(len(detect_l))
                print(count_)
                        
        times+=1

def run_1_day(fileNo):
    exp_str = 'E'+str(fileNo)
    #加载实验参数
    
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(exp_str+'.csv',mode='a')

    _PARAMS_PATH = "params.yaml"
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    repeat_time = (int)(modelParams['CommonParams']['repeat_time'])
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str]
    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate']) 
    #anomalyRate = 0.05

    datasize = 6000
    # MN
    bufferSize = 4000

    anomalyType = modelParams['anomaly_type']    
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    try:
            ##读取txt 文件
            #print('data read success')
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')

    times = 0
    # 开始代码
    while times < repeat_time:
        #repeat time
        for typeName in anomalyType:
            #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]:
                #anomaly type [0,1,2]
                type =  anomalyType[typeName][subtype]  # 加载异常
                #typeName = 'noise'
                #type = [0]
                #获取数据，插入异常
                begin_ = 4000
                end_ = begin_ + datasize  # 8000
                #traindata = dataframe1[:begin_]
                print(typeName+subtype)
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
                test_data,outlier_pos1  = insert_anomaly(test_data, begin_, anomaly_num, typeName, 
                                                        type, delta_mean = 0.5, delta_std_times = 1.5)
                '''
                queue1 = Queue.MNQueue(bufferSize) #存储原始的数据
                for i in range(bufferSize):
                    series = data1.loc[[i]]
                    queue1.enqueue(series)  
                '''
                #预训练
                #scaler = preprocessing.StandardScaler().fit(queue1.df)
                #trainDatascaler = scaler.transform(queue1.df)
                #nus = [nu for nu in np.linspace(0.01,0.2,20)]
                nus = [0.01]
                #gamma = [ga for ga in np.linspace(0.01,1,20)]
                for val in nus: #grid search
                    clf = svm.OneClassSVM(nu=0.01, kernel='rbf', gamma=0.01)
                    #clf.fit(trainDatascaler)
                    clf.fit(train_data)
                    result_l = clf.predict(test_data)
                    detected_ = []
                    for i in range(len(result_l)):
                        if result_l[i] == -1:
                            detected_.append(i)
                    print('gamma',val)
                    tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,detected_,datasize)   

                    #for i in range(len(result_l)):
                    #    if result_l[i] == 1:
                    #        detected_.append(i)
                    '''
                    print(nu_)
                    for row in range(begin_,end_):
                        if row%1000==0:
                            print(row)
                        series = data1.loc[[row]]
                        X = np.array(series.values.tolist()[0]).reshape(1,data1.shape[1])
                        #X_scaler = scaler.transform(X)
                        result = clf.predict(X)
                        #para = clf.get_params()
                        if result==1:
                            pass
                        else:
                            detected_.append(row)
                            print('result')
                    tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,detected_,datasize)
                    '''
                '''
                queue1.clear_count()
                
                result_l = clf.predict(trainDatascaler)

                for row in range(begin_,end_):
                    if row%1000==0:
                        print(row)
                    series = data1.loc[[row]]
                    X = np.array(series.values.tolist()[0]).reshape(1,data1.shape[1])
                    X_scaler = scaler.transform(X)
                    result = clf.predict(X_scaler)
                    #para = clf.get_params()
                    if result==1:
                        pass
                    else:
                        print('result')
               
                '''
if __name__ == "__main__":
    for i in range(3):
        run_1_day(i)