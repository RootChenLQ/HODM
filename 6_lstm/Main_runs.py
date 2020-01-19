import numpy as np
import time
import argparse
import json
from math import sqrt, ceil
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
import matplotlib.pyplot as plt
import lstmfun
import os
import sys

sys.path.append("../")
import Structure
import yaml
import Tools.InsertNoise
import Fun


def run(exp,file_path,size,anomalytype,anomalyRate,timesteps = 5,ouput_timestep = 1, neurons = 20,batches = 50,train_epochs=30):   
    timestamp = time.time()
    #参数加载
    FilePath = file_path #'../datasets/E0/node43.csv'
    train_percentage = 1/3  #比例
    datasize = size
    header_row_index = 0
    best_rmse = np.inf
    col_to_predict = 'Temperature'  #列名
    n_neurons = neurons
    n_batch = batches
    n_epochs = train_epochs #循环次数
    loss_function = 'mae' #损失函数名
    # loss_function = LossFunction.EMD_loss
    optimizer_function = optimizers.SGD(lr=0.05, momentum=0.01, decay=0.0001, nesterov=False) #0.0118
    
     #= 'sgd'#'adam' #优化器名字
    n_in_timestep = timesteps  # step 多少数据预测未来数据
    n_out_timestep = ouput_timestep # 预测的个数
    verbose = 2        #
    index_col_name = None
    cols_to_drop = None
    is_stateful = False
    has_memory_stack = False
    draw_loss_plot = False
    dropnan = True
    draw_prediction_fit_plot = True #用于画图 
    n_features = 3


    #加载数据
    #data,label = lstmfun._load_dataset(FilePath, header_row_index,index_col_name, col_to_predict, cols_to_drop,datasize,train_percentage,anomalytype,anomaly_l)
    try:
        ##读取txt 文件
        #print('data read success')
        dataset = pd.read_csv(FilePath,names = Structure.Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    data = dataset[:datasize].copy()
    #data 不添加异常
    #标准化
    scaler, data_scale = lstmfun._scale_dataset(data, None)
    #print('values before _series_to_supervised\n', values, '\nvalue shape:', values.shape)
    agg1 = lstmfun._series_to_supervised(data_scale, n_in_timestep, n_out_timestep, dropnan, verbose) #将数据转化为标记形式
    print(agg1.shape)
 

    #划分测试集和训练集
    train_X, train_Y, test_X, test_Y = lstmfun._split_data_to_train_test_sets(agg1.values, n_in_timestep, n_features,
                                                                    train_percentage, verbose)
    #训练模型
    model, compatible_n_batch = lstmfun._create_model(train_X, train_Y, test_X, test_Y, n_neurons, n_batch, n_epochs,   #训练过程
                                            is_stateful, has_memory_stack, loss_function, optimizer_function,
                                            draw_loss_plot, verbose)
    trainDataSize = 10000
    anomaly_num = (int)((datasize-trainDataSize)*anomalyRate)

    #获取训练集合预测误差，前百分比作为阈值
    predict_train_y = model.predict(train_X, batch_size=compatible_n_batch, verbose=1 if verbose else 0)
    
    mse = predict_train_y - train_Y
    
    mse0 = abs(mse[:,0])
    mse0.sort()
    mse1 = abs(mse[:,1])
    mse1.sort()
    mse2 = abs(mse[:,2])
    mse2.sort()
    
    #scores = scores.sort()
    cut_point = (int)(0.9*len(mse0))
    thres = [mse0[-1],mse1[-1] ,mse2[-1] ]
    thres2 = [1,1,1]
    #thres = [0.2,0.2,0.2]
    #repeat time
    for typeName in anomalytype:
        #anomaly type normal outlier constant noise
        for subtype in anomalytype[typeName]:
            #anomaly type [0,1,2]
            type =  anomalytype[typeName][subtype] 
            # 获取测试集
            data_ = dataset[:datasize].copy()
            data_ = data_.reset_index(drop=True)
            #获得异常数据 outlier_pos1 original pos
            data_,outlier_pos1  = Tools.InsertNoise.insert_anomaly(data_, trainDataSize, anomaly_num, typeName,type)
            #标准化
            data_scaled = scaler.fit_transform(data_)
            #滑动窗口，形成标记数据集
            agg_test = lstmfun._series_to_supervised(data_scaled, n_in_timestep, n_out_timestep, dropnan, verbose) #将数据转化为标记形式,去除前面NAN值

            train_X, train_Y, test_X, test_Y = lstmfun._split_data_to_train_test_sets(agg_test.values, n_in_timestep, n_features,train_percentage, verbose)
   
            predict_norm = model.predict(test_X, batch_size=compatible_n_batch, verbose=1 if verbose else 0)
            
            #real_d = scaler.inverse_transform(test_Y)
            #actual_test_data, predicted_test_data, rmse_  = lstmfun._make_prediction(model,   #测试
            #                                                                        test_X, test_Y, compatible_n_batch,
            #                                                                        n_in_timestep, n_features, scaler,
            #                                                                        draw_prediction_fit_plot,
            #                                                                        verbose)
            detected = []
           
            for i in range(len(predict_norm)): # 行数
                for j in range(len(predict_norm[0])): #列数
                    if abs(predict_norm[i,j]-test_Y[i,j]) > thres[j]:
                        detected.append(i+len(train_X)+timesteps)
                        break
            #print(outlier_pos1)
            #print(detected)
            
            tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,detected,len(test_X))
            Output_DF = Structure.Output_DF.copy() 
            s1 = pd.Series([exp,n_epochs, typeName,type, tn,fn,fp,tp,acc,fpr,tpr,p,f1,1,time.time()-timestamp],
                            index= Structure.Output_DF_Type)
            Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
            Output_DF.to_csv('lstm.csv',header=0,mode='a')

    # predicted_target 预测值
    '''
    actual_test_data, predicted_test_data, rmse_  = lstmfun._make_prediction(model,   #测试
                                                                                    test_X, test_Y, compatible_n_batch,
                                                                                    n_in_timestep, n_features, scaler,
                                                                                    draw_prediction_fit_plot,
                                                                                    verbose)
                                                                                    '''
    '''
    for i in range(3):
        plt.plot(predicted_test_data[:,i], label="predict data")
        plt.plot(actual_test_data[:,i], label="real data")
        plt.legend()
        plt.show()
    '''
    '''

    s1 = pd.Series([n_batch,n_in_timestep,n_neurons,rmse_,time.time()-timestamp],
                            index= Structure.Output_DF_lstm_Type)
    Output_DF = Structure.Output_DF_lstm.copy() 
    Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
    Output_DF.to_csv('lstm.csv',header=0,mode='a')  
    '''                                                                                                                      
    '''
    if error_value< best_rmse:
       best_rmse =  error_value
       model.save('./my_model_in time step_%d_out_timestep_%d.h5' % (n_in_timestep,n_out_timestep)) #保存
    '''
def run_s(exp):
    #加载配置文件
    # 
    #params.yaml output data
    #选择实验数据
    exp_str = 'E'+str(exp)
    #加载实验参数
    times = 0
    update_times = 0
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv('lstm.csv',mode='a')
    Parent_PATH = ".."
    _PARAMS_PATH = os.path.join(Parent_PATH,"params2.yaml")
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    repeat_time = modelParams['CommonParams']['repeat_time']
    #Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str]
 
    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate']) 

    datasize = modelParams['CommonParams']['datasize']
    #datasize = 6000

    anomalyType = modelParams['anomaly_type']    

    #read data
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    

    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv('lstm.csv',header=0,mode='a')
    ''' 
    for b in range(10,200,30):
        for t in range(5,10,2):
            for n in range(20,50,10):
    '''
    while times < repeat_time:
        
        # 170 7 30
        # 70 5 30
        #
        #def run(file_path,anomalyRate,size,anomalytype,timesteps = 5,ouput_timestep = 1, neurons = 20,batches = 50,train_epochs=30):   
        #for neurons_ in range(30,50,10):
        run(exp_str,datefile1,datasize,anomalyType,anomalyRate,timesteps = 7,ouput_timestep = 1, neurons = 50,batches = 170,train_epochs=500)
if __name__ == "__main__":
    #读取配置文件，datafile文件，配置文件
    '''
    exp_str = 'E'+str(1)
    #加载实验参数
    times = 0
    update_times = 0
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(exp_str+'my.csv',mode='a')
    Parent_PATH = ".."
    _PARAMS_PATH = os.path.join(Parent_PATH,"params.yaml")
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    '''
    for i in range(4):
        run_s(i)
   

