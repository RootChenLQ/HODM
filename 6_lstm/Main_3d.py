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
import matplotlib.pyplot as plt
import lstmfun
import os
import sys
sys.path.append("../")
import Structure
import yaml

def run(file_path,timesteps = 5,ouput_timestep = 1, neurons = 20,batches = 50,train_epochs=30):   
    timestamp = time.time()
    #参数加载
    FilePath = file_path #'../datasets/E0/node43.csv'
    train_percentage = 1/3  #比例
    header_row_index = 0
    best_rmse = np.inf
    col_to_predict = 'Temperature'  #列名
    n_neurons = neurons
    n_batch = batches
    n_epochs = train_epochs #循环次数
    loss_function = 'mae' #损失函数名
    # loss_function = LossFunction.EMD_loss
    optimizer_function = 'adam' #优化器名字
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
    #加载数据
    col_names, values, n_features, output_col_name,anomaly_list = lstmfun._load_dataset(FilePath, header_row_index,
                                                                index_col_name, col_to_predict, cols_to_drop,train_percentage,anomalyname,anomalyL)
    #标准化
    scaler, values = lstmfun._scale_dataset(values, None)
    #print('values before _series_to_supervised\n', values, '\nvalue shape:', values.shape)
    agg1 = lstmfun._series_to_supervised(values, n_in_timestep, n_out_timestep, dropnan, col_names, verbose) #将数据转化为标记形式
    print(agg1.shape)
 

    #划分测试集和训练集
    train_X, train_Y, test_X, test_Y = lstmfun._split_data_to_train_test_sets(agg1.values, n_in_timestep, n_features,
                                                                    train_percentage, verbose)
    #训练模型
    model, compatible_n_batch = lstmfun._create_model(train_X, train_Y, test_X, test_Y, n_neurons, n_batch, n_epochs,   #训练过程
                                            is_stateful, has_memory_stack, loss_function, optimizer_function,
                                            draw_loss_plot, output_col_name, verbose)
    


    # predicted_target 预测值
    actual_target, predicted_target, error_value, error_percentage = lstmfun._make_prediction(model, train_X, train_Y,  #测试
                                                                                    test_X, test_Y, compatible_n_batch,
                                                                                    n_in_timestep, n_features, scaler,
                                                                                    draw_prediction_fit_plot,
                                                                                    output_col_name,
                                                                                    verbose)
    s1 = pd.Series([n_batch,n_in_timestep,n_neurons,error_value,time.time()-timestamp],
                            index= Structure.Output_DF_lstm_Type)
    Output_DF = Structure.Output_DF_lstm.copy() 
    Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
    Output_DF.to_csv('lstm.csv',header=0,mode='a')                                                                                                                        

    if error_value< best_rmse:
       best_rmse =  error_value
       model.save('./my_model_in time step_%d_out_timestep_%d.h5' % (n_in_timestep,n_out_timestep)) #保存
    
def run_s(file_path):
    for b in range(10,200,30):
        for t in range(5,10,2):
            for n in range(10,30,2):
                run(file_path,timesteps = t,ouput_timestep = 1, neurons = n,batches = b,train_epochs=30)
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
    Output_DF = Structure.Output_DF_lstm.copy() 
    Output_DF.to_csv('lstm.csv',mode='a')    
    run_s('../datasets/E0/node43.csv')

