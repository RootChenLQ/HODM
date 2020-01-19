
import numpy as np
import time
import argparse
import json
from math import sqrt, ceil
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
#import Datarejust
import matplotlib.pyplot as plt
import lstmfun

if __name__ == "__main__":
    #!input


    
    file_path = 'node43.csv'
    header_row_index = 0
    index_col_name = None

    col_to_predict = 'Tep'  #列名
    cols_to_drop = None

    train_percentage = 1/3  #比例
    col_names, values, n_features, output_col_name,anomaly_list = lstmfun._load_dataset(file_path, header_row_index,
                                                                index_col_name, col_to_predict, cols_to_drop,train_percentage)
    scaler, values = lstmfun._scale_dataset(values, None)

    print('values before _series_to_supervised\n', values, '\nvalue shape:', values.shape)

    # !input
    n_in_timestep = 5  # step 多少数据预测未来数据
    n_out_timestep = 1 # 预测的个数
    verbose = 2        #
    dropnan = True
    agg1 = lstmfun._series_to_supervised(values, n_in_timestep, n_out_timestep, dropnan, col_names, verbose)
    print(agg1)
    # agg2 = _series_to_supervised(values, 1, 2, dropnan, col_names, verbose)
    # agg3 = _series_to_supervised(values, 2, 1, dropnan, col_names, verbose)
    # agg4 = _series_to_supervised(values, 3, 2, dropnan, col_names, verbose)

    '''
    #不懂_series_to_supervised()中n_in和n_out作用的话把下面被注释掉的列表一打出来就明白了
    print('agg1:\n', agg1.columns)
    print('agg2:\n', agg2.columns)
    print('agg3:\n', agg3.columns)
    print('agg4:\n', agg4.columns)
    #print(agg1)
    agg3
    '''
    # print('agg1.value:\n', agg1.values, '\nagg1.shape:', agg1.shape, '\nagg1.columns:',
    #       agg1.columns)  # agg1和agg1.value是不一样的，agg1是DataFrame，agg1.value是np.array
    # # print('\nagg1\n', agg1)

    #对数据进行分组
    # !input

    train_X, train_Y, test_X, test_Y = lstmfun._split_data_to_train_test_sets(agg1.values, n_in_timestep, n_features,
                                                                    train_percentage, verbose)

    # !input
    n_neurons = 20
    n_batch = 50
    n_epochs = 50 #循环次数
    is_stateful = False
    has_memory_stack = False
    loss_function = 'mae' #损失函数名
    # loss_function = LossFunction.EMD_loss
    optimizer_function = 'adam' #优化器名字
    draw_loss_plot = False
    model, compatible_n_batch = lstmfun._create_model(train_X, train_Y, test_X, test_Y, n_neurons, n_batch, n_epochs,   #训练过程
                                            is_stateful, has_memory_stack, loss_function, optimizer_function,
                                            draw_loss_plot, output_col_name, verbose)
    # model.save('./my_model_%s.h5'%datetime.datetime.now())
    model.save('./my_model_in time step_%d_out_timestep_%d.h5' % (n_in_timestep,n_out_timestep)) #保存

    # !input
    draw_prediction_fit_plot = True
    # predicted_target 预测值
    actual_target, predicted_target, error_value, error_percentage = lstmfun._make_prediction(model, train_X, train_Y,  #测试
                                                                                    test_X, test_Y, compatible_n_batch,
                                                                                    n_in_timestep, n_features, scaler,
                                                                                    draw_prediction_fit_plot,
                                                                                    output_col_name,
                                                                                    verbose)
                                                                                                                                        



