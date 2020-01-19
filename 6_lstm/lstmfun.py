#coding:utf-8
import numpy as np
import time
import argparse
import json
from math import sqrt, ceil
from matplotlib import pyplot
import pandas as pd
#from pandas import read_csv
#from pandas import DataFrame
#from pandas import concat
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
#from sklearn.preprocessing import 
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#import Datarejust
import matplotlib.pyplot as plt
#coding:utf-8
import os
import sys
sys.path.append("../")
import Structure
import Tools.InsertNoise

def _load_dataset(file_path, header_row_index, index_col_name, col_to_predict, cols_to_drop,datasize,train_percentage):
    """
    file_path: the csv file path
    header_row_index: the header row index in the csv file
    index_col_name: the index column (can be None if no index is there)
    col_to_predict: the column name/index to predict
    cols_to_drop: the column names/indices to drop (single label or list-like)
    """
    # read dataset from disk
    try:
        ##读取txt 文件
        #print('data read success')
        dataset = pd.read_csv(file_path,names = Structure.Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')

    #dataset = pd.read_csv(file_path, header=header_row_index, index_col=False)
    # print(dataset)

    # set index col，设置索引列，参数输入列的名字列表
    if index_col_name:
        dataset.set_index(index_col_name, inplace=True)

    # drop nonused colums，删除不需要的列，参数输入列的名字列表
    '''if cols_to_drop:
        if type(cols_to_drop[0]) == int:
            dataset.drop(index=cols_to_drop, axis=0, inplace=True)
        else:
            dataset.drop(columns=cols_to_drop, axis=1, inplace=True)'''
    if cols_to_drop:
        dataset.drop(cols_to_drop, axis=1, inplace=True)

    # print('\nprint data set again\n',dataset)
    # get rows and column names
    #col_names = dataset.columns.values.tolist()  # 获得列表名称
    #values = dataset.values  #获得数值
    # print(col_names, '\n values\n', values)

    # move the column to predict to be the first col: 把预测列调至第一列
    '''
    col_to_predict_index = col_to_predict if type(col_to_predict) == int else col_names.index(col_to_predict)
    output_col_name = col_names[col_to_predict_index]
    if col_to_predict_index > 0:
        col_names = [col_names[col_to_predict_index]] + col_names[:col_to_predict_index] + col_names[
                                                                                           col_to_predict_index + 1:]
    values = np.concatenate((values[:, col_to_predict_index].reshape((values.shape[0], 1)),
                             values[:, :col_to_predict_index], values[:, col_to_predict_index + 1:]), axis=1) #转化为矩阵形式 按照列拼接
    '''
    #将预测数据放到第一列
    # print(col_names, '\n values2\n', values)
    # ensure all data is float
    #nu = 30000
    #values = values[0:datasize,:] #取数据集的前30000个数据
    #values = values.astype("float32")
    #anomaly_per=0.05
    dataset = dataset[0:datasize].copy()
    #trainDataSize = (int)(datasize*train_percentage)
    #anomaly_num = (int)((datasize-trainDataSize)*anomaly_per)
    #dataset,outlier_pos1  = Tools.InsertNoise.insert_anomaly(dataset, trainDataSize, anomaly_num, anomalyName,anomalyList)
    #label = [1 if i in outlier_pos1 else 0 for i in range(len(dataset))]
    ''''
    # 添加异常节点 anomaly_per为异常比例 index表示添加异常的属性
      异常只在test数据集中添加，
      insert_Continue_anomaly 添加长时间异常
    '''''
    #anomaly_per=0.05 #异常比例
    #index = 0;  #if index=0 表示温度异常 =1 表示湿度异常 =2表示电压异常
    '''
    train_intervals = ceil(values.shape[0] * train_percentage)
    train_ = values[:train_intervals, :] # 训练区间
    test_ = values[train_intervals:, :]  # 测试区间
    label = np.zeros((datasize-train_intervals, 1))  #标记'''
    #anomaly_list = []
    
    #插入异常
    


    # test_sample, label, anomaly_per = InsertAnmoly.insert_Continue_anomaly(test_,anomaly_per,label,index=index)
    #test_sample, label, anomaly_per,anomaly_list = InsertAnmoly.insert_test_anomaly(test_, anomaly_per, label, index=index)
    #合并 将正常的数据和异常数据合并
    #插入异常
    #values = np.vstack((train_,test_))  #处理后的 总的数据集 
    # plt.plot(label)
    # plt.show()
    # plt.plot(test_sample[:, 0],color='red',label='have error data',linestyle='--')
    # plt.plot(test_[:,0],color='black',label='normal data')
    # plt.show()

    # print(col_names, '\n values3\n', values)
    #       列的标题名  数值     列数             输出列名            异常列表    
    return dataset#,label

# scale dataset
# def _scale_dataset(values, scale_range = (0,1)):
def _scale_dataset(values, scale_range):
    """
    values: dataset values
    scale_range: scale range to fit data in
    """
    # normalize features
    scaler = MinMaxScaler(feature_range=scale_range or (0, 1))
    scaled = scaler.fit_transform(values)

    return (scaler, scaled)
# convert series to supervised learning (ex: var1(t)_row1 = var1(t-1)_row2)，列表打印出来一看就明白了
# def _series_to_supervised(values, n_in=3, n_out=1, dropnan=True, col_names, verbose=True):
#转化成监督模式
def _series_to_supervised(values, n_in, n_out, dropnan,verbose,col_names=None):
    """
    values: dataset scaled values
    n_in: number of time lags (intervals) to use in each neuron, 与多少个之前的time_step相关,和后面的n_intervals是一样
    n_out: number of time-steps in future to predict，预测未来多少个time_step
    dropnan: whether to drop rows with NaN values after conversion to supervised learning
    col_names: name of columns for dataset
    verbose: whether to output some debug data
    """
  
    n_vars = 1 if type(values) is list else values.shape[1]
    if col_names is None: col_names = ["var%d" % (j + 1) for j in range(n_vars)]

    df = pd.DataFrame(values)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))   #形成滑动窗口的列表
        names += [("%s(t-%d)" % (col_names[j], i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):     #
        cols.append(df.shift(-i))  # 这里循环结束后cols是个列表，每个列表都是一个shift过的矩阵
        if i == 0:
            names += [("%s(t)" % (col_names[j])) for j in range(n_vars)]
        else:
            names += [("%s(t+%d)" % (col_names[j], i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols,
                 axis=1)  # 将cols中的每一行元素一字排开，连接起来，vala t-n_in, valb t-n_in ... valta t, valb t... vala t+n_out-1, valb t+n_out-1
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    if verbose:
        print("\nsupervised data shape:", agg.shape)
    return agg

# split into train and test sets
# def _split_data_to_train_test_sets(values, n_intervals=3, n_features, train_percentage=0.67, verbose=True):
def _split_data_to_train_test_sets(values, n_intervals, n_features, train_percentage, verbose):
    """
    values: dataset supervised values
    n_intervals: number of time lags (intervals) to use in each neuron
    n_features: number of features (variables) per neuron
    train_percentage: percentage of train data related to the dataset series size; (1-train_percentage) will be for test data
    verbose: whether to output some debug data
    """

    n_train_intervals = ceil(values.shape[0] * train_percentage)  # ceil(x)->得到最接近的一个不小于x的整数，如ceil(2.001)=3
    train = values[:n_train_intervals, :]
    test = values[n_train_intervals:, :]

    # split into input and outputs
    n_obs = n_intervals * n_features
    #train_X, train_y = train[:, :n_obs], train[:, -n_features]  # train_Y直接赋值倒数第六列，刚好是t + n_out_timestep-1时刻的0号要预测列
    train_X, train_y = train[:, :n_obs], train[:, n_obs:] # 0116
    
    
    # train_X此时的shape为[train.shape[0], timesteps * features]
    # print('before reshape\ntrain_X shape:', train_X.shape)
    #test_X, test_y = test[:, :n_obs], test[:, -n_features]  # 测试数据
    test_X, test_y = test[:, :n_obs], test[:, n_obs:]  # 0116
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_intervals, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_intervals, n_features))

    if verbose:
        print("")
        print("train_X shape:", train_X.shape)
        print("train_y shape:", train_y.shape)
        print("test_X shape:", test_X.shape)
        print("test_y shape:", test_y.shape)

    return (train_X, train_y, test_X, test_y)


# create the nn model
# def _create_model(train_X, train_y, test_X, test_y, n_neurons=20, n_batch=50, n_epochs=60, is_stateful=False, has_memory_stack=False, loss_function='mae', optimizer_function='adam', draw_loss_plot=True, output_col_name, verbose=True):
def _create_model(train_X, train_y, test_X, test_y, n_neurons, n_batch, n_epochs, is_stateful, has_memory_stack,
                  loss_function, optimizer_function, draw_loss_plot, verbose,output_col_name=''):
    """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs
    test_y: test targets
    n_neurons: number of neurons for LSTM nn
    n_batch: nn batch size
    n_epochs: training epochs
    is_stateful: whether the model has memory states
    has_memory_stack: whether the model has memory stack
    loss_function: the model loss function evaluator
    optimizer_function: the loss optimizer function
    draw_loss_plot: whether to draw the loss history plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
    """

    # design network
    model = Sequential()

    if is_stateful:
        # calculate new compatible batch size
        for i in range(n_batch, 0, -1):
            if train_X.shape[0] % i == 0 and test_X.shape[0] % i == 0:#行数是0
                if verbose and i != n_batch:
                    print(
                        "\n*In stateful network, batch size should be dividable by training and test sets; had to decrease it to %d." % i)
                n_batch = i
                break

        model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True,
                       return_sequences=has_memory_stack))
        if has_memory_stack:
            model.add(LSTM(n_neurons, batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), stateful=True))
    else:
        model.add(LSTM(n_neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(3))  ## 0116
    #model.add(Dense(1)) #全联接层输出个数
    # wasserstein =Wasserstein(train_X,train_y)
    # loss_function=wasserstein.dist(C=0.1,nsteps=10)
    
    model.summary()  #打印模型

    #plt(model, to_file='lstm_model.png')

    model.compile(loss=loss_function, optimizer=optimizer_function)

    if verbose:
        print("")

    # fit network
    losses = []
    val_losses = []
    if is_stateful:
        for i in range(n_epochs):
            history = model.fit(train_X, train_y, epochs=1, batch_size=n_batch,
                                validation_data=(test_X, test_y), verbose=0, shuffle=False)

            if verbose:
                print("Epoch %d/%d" % (i + 1, n_epochs))
                print("loss: %f - val_loss: %f" % (history.history["loss"][0], history.history["val_loss"][0]))

            losses.append(history.history["loss"][0])
            val_losses.append(history.history["val_loss"][0])

            model.reset_states()
    else:
        #history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch,
        #                    validation_data=(test_X, test_y), verbose=2 if verbose else 0, shuffle=False)
        history = model.fit(train_X, train_y, epochs=n_epochs, batch_size=n_batch,
                            validation_data= None,validation_split=0.1, verbose=2 if verbose else 0, shuffle=False)
    #损失函数的迭代图，在history中
    ''' 
    if draw_loss_plot:
        plt.plot(history.history["loss"] if not is_stateful else losses, label="Train Loss (%s)" % output_col_name)
        plt.plot(history.history["val_loss"] if not is_stateful else val_losses,
                    label="Test Loss (%s)" % output_col_name)
        plt.legend()
        plt.show()
    '''
    print(history.history)
    # model.save('./my_model_%s.h5'%datetime.datetime.now())
    return (model, n_batch)


# make a prediction
# def _make_prediction(model, train_X, train_y, test_X, test_y, compatible_n_batch, n_intervals=3, n_features, scaler=(0,1), draw_prediction_fit_plot=True, output_col_name, verbose=True):
def _make_prediction(model, test_X, test_y, compatible_n_batch, n_intervals, n_features, scaler,
                     draw_prediction_fit_plot, verbose, output_col_name=''):
    """
    train_X: train inputs
    train_y: train targets
    test_X: test inputs   
    test_y: test targets
    compatible_n_batch: modified (compatible) nn batch size
    n_intervals: number of time lags (intervals) to use in each neuron 每个神经元使用的时差（间期）数
    n_features: number of features (variables) per neuron
    scaler: the scaler object used to invert transformation to real scale 用于将转换反转为实际比例的scaler对象
    draw_prediction_fit_plot: whether to draw the the predicted vs actual fit plot
    output_col_name: name of the output/target column to be predicted
    verbose: whether to output some debug data
    """

    if verbose:
        print("")
    #batch_size
    predict_norm = model.predict(test_X, batch_size=compatible_n_batch, verbose=1 if verbose else 0)
    predict_d = scaler.inverse_transform(predict_norm)



    test_X = test_X.reshape((test_X.shape[0], n_intervals * n_features))
    #real_norm = test_y[:, -n_features:]
    real_d = scaler.inverse_transform(test_y)
    '''
    yhat2 = model.predict(train_X, batch_size=compatible_n_batch, verbose=1 if verbose else 0)
    train_X = train_X.reshape((train_X.shape[0], n_intervals * n_features))
    '''
    # invert scaling for forecast invert scaling
    # 预测的反向缩放
    #inv_yhat = np.concatenate((yhat, test_X[:, (1 - n_features):]), axis=1)
    #inv_yhat = scaler.inverse_transform(inv_yhat)     #报错
    #inv_yhat = inv_yhat[:, 0]  # 看输出结构
    # invert scaling for actual 反转  实际缩放
    #test_y = test_y.reshape((len(test_y), 1))
    #inv_y = np.concatenate((test_y, test_X[:, (1 - n_features):]), axis=1) #按照列拼接
    #inv_y = scaler.inverse_transform(inv_y)
    #print(inv_y)
    #inv_y = inv_y[:, 0]
    '''
    inv_yhat2 = np.concatenate((yhat2, train_X[:, (1 - n_features):]), axis=1)
    inv_yhat2 = scaler.inverse_transform(inv_yhat2)
    inv_yhat2 = inv_yhat2[:, 0]#训练集合的预测值

    train_y = train_y.reshape((len(train_y), 1))
    inv_y2 = np.concatenate((train_y, train_X[:, (1 - n_features):]), axis=1)
    inv_y2 = scaler.inverse_transform(inv_y2)
    inv_y2 = inv_y2[:, 0]#训练集合的真是值
    '''
    # calculate RMSE
    rmse_ = []
    rmse_.append(sqrt(mean_squared_error(predict_d[:,0], real_d[:,0])))
    rmse_.append(sqrt(mean_squared_error(predict_d[:,1], real_d[:,1])))
    rmse_.append(sqrt(mean_squared_error(predict_d[:,2], real_d[:,2])))

    plt.plot(rmse_[0])

    # calculate average error percentage
    #avg = np.average(inv_y)
    #error_percentage = rmse / avg 

    #f verbose:
    #    print("")
    #    print("Test Root Mean Square Error: %.3f" % rmse)
    #    print("Test Average Value for %s: %.3f" % (output_col_name, avg))
    #   print("Test Average Error Percentage: %.2f/100.00" % (error_percentage * 100))
    '''
    if draw_prediction_fit_plot:
        pyplot.plot(inv_y, label="test Actual (%s)" % output_col_name)
        pyplot.plot(inv_yhat, label="test Predicted (%s)" % output_col_name)
        pyplot.legend()
        pyplot.show()

        
    if draw_prediction_fit_plot:
        pyplot.plot(inv_y2, label="train Actual2(%s)" % output_col_name)
        pyplot.plot(inv_yhat2, label="train Predicted2 (%s)" % output_col_name)
        pyplot.legend()
        pyplot.show()
    '''
    # 真实 预测值
    return (real_d, predict_d, rmse_)

