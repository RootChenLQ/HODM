#encoding:UTF-8
import pandas as pd
import numpy as np
import Fun
import random
random.seed = 0
import time
from sklearn import preprocessing
from Tools.InsertNoise import insert_outlier_error
# centralized model 
# 每一段数据集中采集，计算均值，方差等标准量，
# 计算每一段数据

def train_knn(df):
    #预处理数据集，得到标准化参数 scalar
    data_scale = preprocessing.scale(df)
    scaler = preprocessing.StandardScaler().fit(df)
    #print(data_scale)
    thres = 0
    #获取 记录数据集之间的距离
    euclidean_dis = np.zeros((len(data_scale),len(data_scale)))
    for i in range(len(data_scale)):
        temp = data_scale[i,0]
        humi = data_scale[i,1]
        volt = data_scale[i,2]
        #print(temp,humi,volt)
        j = i
        while j<len(df)-1:
            j+=1
            temp_ = data_scale[j,0]
            humi_ = data_scale[j,1]
            volt_ = data_scale[j,2]
            euclidean_dis[i][j] = np.sqrt((temp-temp_)**2+(humi-humi_)**2+(volt-volt_)**2)
            euclidean_dis[j][i] = euclidean_dis[i][j]
    #排序每个数据到其他数据的距离值:按照行排列 。二维数组。
    sort_index = np.argsort(euclidean_dis, axis=1)
    #     print(sort_index)  #获取每个数据最近邻的序号
    #print('sort_index',sort_index)
    knn_meandis = np.zeros(euclidean_dis.shape[0]) # 距离值倒叙排序，取前K个值求平均。 记录平均值
    #     print(knn_meandis)
     #选取每个节点离其他节点的第K近的距离作为其判别条件
    for i in range(sort_index.shape[0]): #累加K值
        mean_knn_dis = 0
        for j in range(k):
            mean_knn_dis += euclidean_dis[i][sort_index[i][j]]
            #print('euclidean_dis(%d,%d)'%(i,sort_index[i][j]),euclidean_dis[i][sort_index[i][j]])
        mean_knn_dis /= k
        knn_meandis[i] = mean_knn_dis  #获取排序后，第K个取值
    #按照数组的值，降序排列。
    knn_meandis_sort = np.argsort(-knn_meandis)  #按照第K个距离值，升序排序
    thres = knn_meandis[knn_meandis_sort[0]]
    return thres,scaler

def detect(df,thres,scaler):
    #
    detect = np.array([])
    data_scale = scaler.transform(df)
     #获取
    euclidean_dis = np.zeros((len(data_scale),len(data_scale)))
    for i in range(len(data_scale)):
        temp = data_scale[i,0]
        humi = data_scale[i,1]
        volt = data_scale[i,2]
        #print(temp,humi,volt)
        j = i
        while j<len(df)-1:
            j+=1
            temp_ = data_scale[j,0]
            humi_ = data_scale[j,1]
            volt_ = data_scale[j,2]
            euclidean_dis[i][j] = np.sqrt((temp-temp_)**2+(humi-humi_)**2+(volt-volt_)**2)
            euclidean_dis[j][i] = euclidean_dis[i][j]
    #排序每个数据到其他数据的距离值:按照行排列 。二维数组。
    sort_index = np.argsort(euclidean_dis, axis=1)
    #     print(sort_index)  #获取每个数据最近邻的序号
    #print('sort_index',sort_index)
    knn_meandis = np.zeros(euclidean_dis.shape[0]) # 距离值倒叙排序，取前K个值求平均。 记录平均值
    #     print(knn_meandis)
     #选取每个节点离其他节点的第K近的距离作为其判别条件
    for i in range(sort_index.shape[0]):
        mean_knn_dis = 0
        for j in range(k):
            mean_knn_dis += euclidean_dis[i][sort_index[i][j]]
            #print('euclidean_dis(%d,%d)'%(i,sort_index[i][j]),euclidean_dis[i][sort_index[i][j]])
        mean_knn_dis /= k-1   # exit zero 
        knn_meandis[i] = mean_knn_dis  #获取排序后，第K个取值
    #按照数组的值，降序排列。
    knn_meandis_sort = np.argsort(-knn_meandis)  #按照第K个距离值，升序排序
    for i in range(len(knn_meandis_sort)):
        if knn_meandis[knn_meandis_sort[i]] >= thres:
            detect = np.append(detect,knn_meandis_sort[i])
        else:
            break
    return detect

if __name__ == '__main__':## 其他
    
    times = 0
    k = 10
    save_type = ['ID','TN','FN','FP','TP','ACC','FPR','TPR','P']
    df = pd.DataFrame(columns = save_type) 
    while times <1:
        #导入数据
        # Filled_DF_Type = ['Temperature','Humidity','Light','Voltage'] #四维
        Filled_DF_Type = ['Temperature','Humidity','Voltage']
        #intel         
        datefile1 = 'datasets/node43op.csv'

        #datefile2 = 'datasets/node44op.csv'
        #datefile3 = 'datasets/node45op.csv'
        try:
            ##读取txt 文件
        #             print('data read success')
            data1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
        except IOError:
            print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
        '''
        try:
            ##读取txt 文件
        #             print('data read success')
            data2 = pd.read_csv(datefile2,names = Filled_DF_Type,sep=',')
        except IOError:
            print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
        try:
            ##读取txt 文件
        #             print('data read success')
            data3 = pd.read_csv(datefile3,names = Filled_DF_Type,sep=',')
        except IOError:
            print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')  
        '''
        datalen = 3000
        bufferSize = 300
        data1 = data1[0:datalen].copy()
        outlier_pos = []
        #data1,outlier_pos = Fun.insert_noise_error(data1,data2,data3,bufferSize,5,'Intel')
        data1,outlier_pos = insert_outlier_error(data1,bufferSize ,30)
        #创建节点
        #data1 插入异常
        detect_all = np.array([])
        #train_len = 300
        loops = datalen//bufferSize

        detect_pos = np.array([])

        for i in range(loops):
            subdata = data1[i*bufferSize:(i+1)*bufferSize].copy()
            if i == 0: #预训练
                thres,scaler = train_knn(subdata)
            else:      #检测阶段，正常的数据，作为后续的训练集
                outlier = detect(subdata,thres,scaler)
                outlier = outlier+(i*bufferSize)
                detect_pos = np.append(detect_pos,outlier)
                thres,scaler = train_knn(subdata)

        print('node8')
        tn,fn,fp,tp,acc,fpr,tpr,p = Fun.compute_same_num(outlier_pos,detect_pos,datalen-bufferSize)
        s1 = pd.Series([8,tn,fn,fp,tp,acc,fpr,tpr,p],
                       index= save_type)
        df = df.append(s1,ignore_index = True,sort=False)
        times+=1
#df.to_csv('knn intel 8_9_10 3000_500 5_1.5.csv')