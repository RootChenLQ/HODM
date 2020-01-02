import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import Structure
import pandas as pd 
from Tools.InsertNoise import *
import yaml
import Fun
from scipy import stats
from sklearn import preprocessing
def run(fileNo):
    exp_str = 'E'+str(fileNo)
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
    anomalyRate = 0.01
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
                print(typeName+subtype)
                begin_ = 2000
                end_ = begin_ + datasize
                data1 = dataframe1[begin_:end_].copy()
                data1 = data1.reset_index(drop=True)

                #插入异常
                outlier_pos1 = []
                anomaly_num = (int)((datasize)*anomalyRate)
                data1,outlier_pos1  = insert_anomaly(data1, 0, anomaly_num, typeName, 
                                                        type, delta_mean = 0.5, delta_std_times = 1.5)

                data1_scale = preprocessing.scale(data1)
                
                np.savetxt('lof.csv',data1_scale)
                # fit the model
                #LocalOutlierFactor 参数
                #n_neighbors=20, algorithm=’auto’, leaf_size=30, 
                #metric=’minkowski’, p=2, metric_params=None, contamination=0.1, n_jobs=1
                clf = LocalOutlierFactor(n_neighbors=100, contamination=anomalyRate)  # 加载LOF分类器
                y_pred = clf.fit_predict(data1_scale)
                scores_pred = clf.negative_outlier_factor_
                #threshold = stats.scoreatpercentile(scores_pred, 100 * anomalyRate)  # 根据异常样本比例，得到阈值，用于绘图
                thres = np.percentile(scores_pred,100 * anomalyRate)
                count = 0
                detected_list = []
                for i in range(len(scores_pred)):
                    if scores_pred[i]<=thres:
                        detected_list.append(i)

                '''
                for i  in range(len(y_pred)):
                    if y_pred[i] == -1:
                        count+=1
                        detected_list.append(i)
                '''
                #print(count)
                tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,detected_list,datasize)

                scores_pred = clf.negative_outlier_factor_
                threshold = stats.scoreatpercentile(scores_pred, 100 * anomalyRate)  # 根据异常样本比例，得到阈值，用于绘图
                
                '''
                # plot the level sets of the decision function
                xx, yy = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))
                Z = clf._decision_function(np.c_[xx.ravel(), yy.ravel()])  # 类似scores_pred的值，值越小越有可能是异常点
                Z = Z.reshape(xx.shape)
                
                plt.title("Local Outlier Factor (LOF)")
                # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
                
                plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)  # 绘制异常点区域，值从最小的到阈值的那部分
                a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')  # 绘制异常点区域和正常点区域的边界
                plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='palevioletred')  # 绘制正常点区域，值从阈值到最大的那部分
                
                b = plt.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white',
                                    s=20, edgecolor='k')
                c = plt.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black',
                                    s=20, edgecolor='k')
                plt.axis('tight')
                plt.xlim((-7, 7))
                plt.ylim((-7, 7))
                plt.legend([a.collections[0], b, c],
                        ['learned decision function', 'true inliers', 'true outliers'],
                        loc="upper left")
                plt.show()
                '''

if __name__ == "__main__":
    for i in range(3):
        run(i)