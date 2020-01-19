import numpy as np
import random
import h5py
import pandas as pd#引入pandas
import matplotlib.pylab as plt

# 本函数功能：实现异常值插入
#      分为三种：噪声异常 ； 长时间异常 ； outlier异常

#插入噪声异常
#参数：sample 为样本数据 anomaly_per为异常样本的百分比
def insert_point_anomaly(test_sample, anomaly_per, label ,index):
    # 找到需要插入点的位置和属性（温度，湿度，光照），插入方法为随机插入
    m, n = test_sample.shape
    anomaly_numbers =int(m*anomaly_per)
    anomaly_list = np.random.choice(m, size=anomaly_numbers, replace=False) #插入异常位置
    anomaly_properties = np.random.choice(3, size=anomaly_numbers) #插入异常的属性 0为te 1为RH 2为light
    #anomaly_list = anomaly_list.reshape([anomaly_numbers,1])      #将list改为(200，)->( 200*1)
    #anomaly_properties=anomaly_properties.reshape([anomaly_numbers,1]) #将anomaly_properties改为(200，)->( 200*1)
    anomaly_list.sort()  #对插入异常点的位置进行排序
    #计算test_sample样本的平均值
    test_sample_te_mean=np.mean(test_sample[:,0])
    test_sample_RH_mean = np.mean(test_sample[:,1])
    test_sample_ligth_mean = np.mean(test_sample[:,2])
    #计算测试样本的标准差
    test_sample_te_std = np.std(test_sample[:,0])
    test_sample_RH_std = np.std(test_sample[:,1])
    test_sample_ligth_std = np.std(test_sample[:,2])
    #计算异常样本的平均值和标准差
    erros_mean=0.2
    erros_std=1.2
    te_erro_number = sum(anomaly_properties == 0)
    RH_erro_number = sum(anomaly_properties == 1)
    Light_erro_number = sum(anomaly_properties == 2)
    # 给数据添加高斯噪声
    erros_te = np.random.normal(test_sample_te_mean+erros_mean,test_sample_te_std*erros_std,te_erro_number)
    erros_RH = np.random.normal(test_sample_RH_mean + erros_mean, test_sample_RH_std * erros_std, RH_erro_number)
    erros_light = np.random.normal(test_sample_ligth_mean + erros_mean, test_sample_ligth_std * erros_std, Light_erro_number)
    te=0
    RH=0
    light=0
    for i in range(anomaly_numbers):
        if anomaly_properties[i] == 0:
            test_sample[anomaly_list[i], 0] =erros_te[te]
            label[anomaly_list[i],0] = 1
            te = te+1
        elif anomaly_properties[i] == 1:
            test_sample[anomaly_list[i], 1] =erros_RH[RH]
            label[anomaly_list[i],0] = 1
            RH = RH+1
        else:
            test_sample[anomaly_list[i], 2] =erros_light[light]
            label[anomaly_list[i],0] = 1
            light = light+1

    # plt.plot(test_sample)
    # plt.show()

    return test_sample, label, anomaly_per


#插入长时间异常
#参数：sample 为样本数据 anomaly_per为异常样本的百分比
def insert_Continue_anomaly(test_sample, anomaly_per, label, index):
    # 找到需要插入点的位置和属性（温度，湿度，光照），插入方法为随机插入
    # 插入异常持续时间为20个数据采样，
    Continue = 50
    m, n = test_sample.shape
    # anomaly_numbers =int(m*anomaly_per)
    anomaly_numbers= 2
    seq_error_number= anomaly_numbers  #计算需要插入多少个异常序列
    anomaly_list=random.sample(range(200,m-Continue),seq_error_number) #插入异常位置
    # anomaly_properties = np.random.choice(3, size=anomaly_numbers) #插入异常的属性 0为te 1为RH 2为light
    anomaly_properties=index
    anomaly_list.sort()  #对插入异常点的位置进行排序
    #计算test_sample样本的平均值
    test_sample_te_mean=np.mean(test_sample[:,0])
    test_sample_RH_mean = np.mean(test_sample[:,1])
    test_sample_ligth_mean = np.mean(test_sample[:,2])
    #计算测试样本的标准差
    test_sample_te_std = np.std(test_sample[:,0])
    test_sample_RH_std = np.std(test_sample[:,1])
    test_sample_ligth_std = np.std(test_sample[:,2])
    #计算异常样本的平均值和标准差
    erros_mean=3
    erros_std=1.5
    # te_erro_number = sum(anomaly_properties == 0)
    # RH_erro_number = sum(anomaly_properties == 1)
    # Light_erro_number = sum(anomaly_properties == 2)
    # 给数据添加高斯噪声
    erros_te = np.random.normal(test_sample_te_mean+erros_mean,test_sample_te_std*erros_std,anomaly_numbers*Continue)
    # erros_RH = np.random.normal(test_sample_RH_mean + erros_mean, test_sample_RH_std * erros_std, RH_erro_number)
    # erros_light = np.random.normal(test_sample_ligth_mean + erros_mean, test_sample_ligth_std * erros_std, Light_erro_number)
    te=0
    # RH=0
    # light=0
    for i in range(seq_error_number):
        #我们只在一维上添加长时间的异常
        for j in range(Continue):
            test_sample[anomaly_list[i]+j, 0] =erros_te[te]
            label[anomaly_list[i]+j,0] = 1
            te = te+1
        # elif anomaly_properties[i] == 1:
        #     test_sample[anomaly_list[i], 1] =erros_RH[RH]
        #     RH = RH+1
        # else:
        #     test_sample[anomaly_list[i], 2] =erros_light[light]
        #     light = light+1
    #
    # plt.plot(test_sample)
    # plt.show()

    return test_sample,label,anomaly_per

#插入outlier异常
#参数：sample 为样本数据 anomaly_per为异常样本的百分比
def insert_outlier_anomaly(test_sample, anomaly_per,label,index):
    # 找到需要插入点的位置和属性（温度，湿度，光照），插入方法为随机插入
    m, n = test_sample.shape
    anomaly_numbers =int(m*anomaly_per)
    anomaly_list = np.random.choice(m, size=anomaly_numbers, replace=False) #插入异常位置
    anomaly_properties=np.zeros([anomaly_numbers,1])
    # anomaly_properties = np.random.choice(3, size=anomaly_numbers) #插入异常的属性 0为te 1为RH 2为light
    #anomaly_list = anomaly_list.reshape([anomaly_numbers,1])      #将list改为(200，)->( 200*1)
    #anomaly_properties=anomaly_properties.reshape([anomaly_numbers,1]) #将anomaly_properties改为(200，)->( 200*1)
    anomaly_list.sort()  #对插入异常点的位置进行排序
    #计算test_sample样本的平均值
    test_sample_te_mean=np.mean(test_sample[:,0])
    test_sample_RH_mean = np.mean(test_sample[:,1])
    test_sample_ligth_mean = np.mean(test_sample[:,2])
    #计算测试样本的标准差
    test_sample_te_std = np.std(test_sample[:,0])
    test_sample_RH_std = np.std(test_sample[:,1])
    test_sample_ligth_std = np.std(test_sample[:,2])
    #计算异常样本的平均值和标准差
    erros_mean=1
    erros_std=3
    te_erro_number = sum(anomaly_properties == 0)
    RH_erro_number = sum(anomaly_properties == 1)
    Light_erro_number = sum(anomaly_properties == 2)
    # 给数据添加高斯噪声
    erros_te = np.random.normal(test_sample_te_mean+erros_mean,test_sample_te_std*erros_std,te_erro_number)
    erros_RH = np.random.normal(test_sample_RH_mean + erros_mean, test_sample_RH_std * erros_std, RH_erro_number)
    erros_light = np.random.normal(test_sample_ligth_mean + erros_mean, test_sample_ligth_std * erros_std, Light_erro_number)
    te=0
    RH=0
    light=0
    for i in range(anomaly_numbers):
        if anomaly_properties[i,0] == 0:
            test_sample[anomaly_list[i], 0] =test_sample[anomaly_list[i], 0]*1.2
            label[anomaly_list[i],0] = 1
            te = te+1

        elif anomaly_properties[i] == 1:
            test_sample[anomaly_list[i], 1] =test_sample[anomaly_list[i], 1]*1.2
            label[anomaly_list[i],0] = 1
            RH = RH+1
        else:
            test_sample[anomaly_list[i], 2] =test_sample[anomaly_list[i], 2]*1.2
            label[anomaly_list[i],0] = 1
            light = light+1

    # plt.plot(test_sample)
    # plt.show()

    return test_sample,label, anomaly_per
def insert_test_anomaly(test_sample, anomaly_per,label,index):
    # 找到需要插入点的位置和属性（温度，湿度，光照），插入方法为随机插入
    # 插入异常持续时间为20个数据采样，
    Continue = 14
    m, n = test_sample.shape
    # anomaly_numbers =int(m*anomaly_per)
    anomaly_numbers= 3
    seq_error_number= anomaly_numbers  #计算需要插入多少个异常序列
    anomaly_list=random.sample(range(200,m-Continue),seq_error_number) #插入异常位置
    # anomaly_properties = np.random.choice(3, size=anomaly_numbers) #插入异常的属性 0为te 1为RH 2为light
    anomaly_properties=index
    anomaly_list.sort()  #对插入异常点的位置进行排序
    #计算test_sample样本的平均值
    test_sample_te_mean=np.mean(test_sample[:,0])
    test_sample_RH_mean = np.mean(test_sample[:,1])
    test_sample_ligth_mean = np.mean(test_sample[:,2])
    #计算测试样本的标准差
    test_sample_te_std = np.std(test_sample[:,0])
    test_sample_RH_std = np.std(test_sample[:,1])
    test_sample_ligth_std = np.std(test_sample[:,2])
    #计算异常样本的平均值和标准差
    erros_mean=3
    erros_std=1.5
    # te_erro_number = sum(anomaly_properties == 0)
    # RH_erro_number = sum(anomaly_properties == 1)
    # Light_erro_number = sum(anomaly_properties == 2)
    # 给数据添加高斯噪声
    # erros_te = np.random.normal(test_sample_te_mean+erros_mean,test_sample_te_std*erros_std,anomaly_numbers*Continue)
    erros_te = 50
    # erros_RH = np.random.normal(test_sample_RH_mean + erros_mean, test_sample_RH_std * erros_std, RH_erro_number)
    # erros_light = np.random.normal(test_sample_ligth_mean + erros_mean, test_sample_ligth_std * erros_std, Light_erro_number)
    te = 0
    # RH=0
    # light=0
    for i in range(seq_error_number):
        #我们只在一维上添加长时间的异常
        for j in range(Continue):
            test_sample[anomaly_list[i]+j, 0] =erros_te
            label[anomaly_list[i]+j,0] = 1
            # te = te+1
        # elif anomaly_properties[i] == 1:
        #     test_sample[anomaly_list[i], 1] =erros_RH[RH]
        #     RH = RH+1
        # else:
        #     test_sample[anomaly_list[i], 2] =erros_light[light]
        #     light = light+1
    #
    # plt.plot(test_sample)
    # plt.show()

    return test_sample,label,anomaly_per,anomaly_list








