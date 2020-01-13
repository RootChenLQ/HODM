#coding:utf-8
import os
import sys
sys.path.append("../")
import Fun
import Node
import Structure
import pandas as pd
import numpy as np
import random
from Tools.commonFun import sort_byNum,getMaxChebyshevDistance_index
from Tools.InsertNoise import *
from Tools.commonFun import *
random.seed = 0
import time
import yaml
import Queue
#using incremental learning, basic edition from HODM.py
#记录多次实验的性能
    
def update(bufferQue,H,sample_for_k):
    MN_info = Structure.MN_selfconfigure_DF.copy()
    for i in range(len(bufferQue)):
        local_info =  Fun.get_localMN_info(bufferQue[i].df)
        MN_info = MN_info.append(local_info,ignore_index=True,sort=False)
    global_info = Fun.get_Normal_Profile(MN_info,H)
    pos_global,num_global,K = [],[],[]
    for i in range(len(bufferQue)):
        normDF = Fun.localdata_norm(global_info,bufferQue[i].df.copy())
        #get local pos num
        pos_local,num_local,maha_last = Fun.local_data_distribution(global_info,normDF)
        #get local K
        K_local = Fun.calculate_K(normDF,sample_for_k,global_info,pos_local,num_local)
        K.append(K_local)
        pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local,num_local)
    NP_df = sort_byNum(pos_global,num_global)
    #K_ = (int)((K_local1+K_local2+K_local3)/3)
    K_ = np.floor(np.mean(K))
    #print(NP_df)
    return global_info, NP_df , K_

def online_detetion(NP_,K_,Global_info_,s1,row):
    data_is_normal, cells_num,maha_now,pos_now,count,sum = Fun.onePoint_detect(
                                                         NP_['pos'].tolist(),NP_['num'].tolist(),K_,
                                                         Global_info_,s1['Temperature'],
                                                         s1['Humidity'],s1['Voltage'],row) 
    return data_is_normal , pos_now ,count




def run(exp):
    #params.yaml output data
    #选择实验数据
    exp_str = 'E'+str(exp)
    #加载实验参数
    times = 0
    update_times = 0
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv(exp_str+'HGDB.csv',mode='a')
    Parent_PATH = ".."
    _PARAMS_PATH = os.path.join(Parent_PATH,"params.yaml")
    
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    repeat_time = modelParams['CommonParams']['repeat_time']
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str]
    # Presition:
    presition_dic = modelParams['Presition']
    #common
    attributes = modelParams['CommonParams']['attributes']  # 3
    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate']) 
    continueErrorThres = modelParams['CommonParams']['continueErrorThres'] 
    #pos_buffer_size = modelParams['CommonParams']['pos_buffer_size']  #
    datasize = modelParams['CommonParams']['datasize']
    #datasize = 6000
    #statistic_analysis_data_size = modelParams['CommonParams']['statistic_analysis_data_size']   
    # MN
    bufferSize = modelParams['SNParams']['buffer_size']  #4.23 
    #bufferSize = 2000
    #sqrt_thres = modelParams['SNParams']['sqrt_thres']
    store_pro = modelParams['SNParams']['store_pro']
    sampleforK = (int)(modelParams['SNParams']['sample_size_rate']*bufferSize)
    #Sink 
    member_size = modelParams['CHParams']['member_size'] #三个节点
    #sink = Node.CH_Node(0,member_size,attributes,bufferSize)
    #H_range = [val/100 for val in range((int)(H_l*100),(int)(H_h*100))]  # HGDM_
    H_l,H_h = Fun.get_HG_H(attributes,bufferSize)
    H = H_h

    anomalyType = modelParams['anomaly_type']    
    #read data
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    datefile2 = datafile_dic['data2']   #'datasets/node44.csv'
    datefile3 = datafile_dic['data3']    #'datasets/node45.csv'
    
    


    try:
        ##读取txt 文件
        #print('data read success')
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    try:
        ##读取txt 文件
        #print('data read success')
        dataframe2 = pd.read_csv(datefile2,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    try:
        ##读取txt 文件
        #print('data read success')
        dataframe3 = pd.read_csv(datefile3,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')   

    # 开始代码
    while times < repeat_time:
        #repeat time
        for typeName in anomalyType:
            #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]:
                #anomaly type [0,1,2]
                type =  anomalyType[typeName][subtype]
                #print(type)
                #type = [0]
                #typeName = 'noise'
                #type = [0,1,2]
                timestamp = time.time()
                #拷贝数据集,避免替代原来的数据集
                begin_ = 0
                end_ = begin_ + datasize
                data1 = dataframe1[begin_:end_].copy()
                data2 = dataframe2[begin_:end_].copy()
                data3 = dataframe3[begin_:end_].copy()
                data1 = data1.reset_index(drop=True)
                data2 = data2.reset_index(drop=True)
                data3 = data3.reset_index(drop=True)
                #记录更新次数
                update_times = 0
                ###################################
                #插入异常
                outlier_pos1 = []
                start = bufferSize
                #异常数量 datasize = traindata_size + test_datasize
                anomaly_num = (int)((datasize - bufferSize)*anomalyRate)
                
                #选择插入异常类型
                #type = [2]
                #data1,outlier_pos1,noise_data = Fun.insert_noise_error(data1,data2,data3,1000,2,'Intel')
                #data1,outlier_pos1,noise_data = Fun.insert_noise(data1,bufferSize,int(datasize*0.05),3,2)
                #data1,outlier_pos1  = insert_outlier_error(data1,type, start,300)
                #data1,outlier_pos1  = insert_constant_error(data1,type, start,5)
                #data1,outlier_pos1  = insert_noise_error(data1,type,start,anomaly_num,0.5,1.5)

                #插入异常
                data1,outlier_pos1  = insert_anomaly(data1, start, anomaly_num, typeName,type)
    
                continue_error_count = [0,0,0] #记录连续异常数量
                '''output noie data'''
                #data1.to_csv("debug.csv")
                #导出异常数据集
                data_label1 = data1.copy()
                data_label1.insert(0,'label', np.zeros(data1.shape[0]))
                for outlier_index in outlier_pos1:
                    data_label1.loc[outlier_index,'label'] = 1
                out_noisefilestr_ = str(times)+'_'+ typeName +'_'+ str(subtype)+'HGDB.csv'
                data_label1.to_csv(out_noisefilestr_)
                #data1['Temperature'].plot()
                #node 2,3 not exit anomaly 
                outlier_pos2 = []
                outlier_pos3 = []

                dataArr = [data1.copy(),data2.copy(),data3.copy()]
                label_l = [outlier_pos1,outlier_pos2,outlier_pos3]

                #pretrain 300
                #buffer 存储 正确数据，提出明显异常
                #定义缓存队列
                queue1 = Queue.MNQueue(bufferSize)
                queue2 = Queue.MNQueue(bufferSize)
                queue3 = Queue.MNQueue(bufferSize)
                bufferQue = [queue1, queue2, queue3]
                #预训练填充数据
                for i in range(bufferSize):
                    for index in range(member_size):
                        series = dataArr[index].iloc[[i]]
                        bufferQue[index].enqueue(series)  
                #预训练NP
                #begin = 0 
                if bufferQue[0].is_trigger_update():
                    Global_info , NP_df , K = update(bufferQue,H,sampleforK)
                    '''
                    thres_t = (float)(Global_info['side_t']*H_h*Global_info['std_T'])
                    thres_h = (float)(Global_info['side_h']*H_h*Global_info['std_H'])
                    thres_v = (float)(Global_info['side_v']*H_h*Global_info['std_V'])
                    thres = [thres_t,thres_h,thres_v]'''
                    thres = []
                    update_times +=1
                    bufferQue[0].clear_count()
                    bufferQue[1].clear_count()
                    bufferQue[2].clear_count()
                #K = 3  # fixed K
                    NP_local = [NP_df.copy(),NP_df.copy(),NP_df.copy()]

                #recorddf1 = Structure.QueueBuffer_DF.copy()    # record normal data
                #recorddf2 = Structure.QueueBuffer_DF.copy()   
                #recorddf3 = Structure.QueueBuffer_DF.copy()  
                #buffer recent data
                
                # record error index
                l1 = []
                l2 = []
                l3 = []
                error_l = [l1,l2,l3]
                
                # start online operation
                #data_size = 2000
                for row in range(bufferSize,datasize):
                    ## online detection node8
                    #print(row)
                    for i in range(member_size):
                        if row%1000 == 0:
                            print(i,' ',row)
                        if row in outlier_pos1 and i ==0:
                            #print('debug')
                            pass 
                        #if row > 537 and i == 0:
                        #    print('test')
                        ############
                        series = dataArr[i].loc[row,:]
                        #print(series)
                        #HGDB检测
                        is_normal,pos,count = online_detetion(NP_local[i],K,Global_info,series,row)
                        if is_normal:
                            p = np.random.rand()  ##enqueue
                            #print(p)              #产生一个概率数，
                            if p <= store_pro:    #概率存储
                                bufferQue[i].enqueue(series)
                        else:
                            error_l[i].append(row) 
                          
                        if bufferQue[i].is_trigger_update():# or continue_error_count[i] > continueErrorThres:
                            #print('continue_error_count',continue_error_count)
                            Global_info , NP_df , K = update(bufferQue,H,sampleforK)
                            #thres_t = (float)(Global_info['side_t']*H_h*Global_info['std_T'])
                            #thres_h = (float)(Global_info['side_h']*H_h*Global_info['std_H'])
                            #thres_v = (float)(Global_info['side_v']*H_h*Global_info['std_V'])
                            #thres = [thres_t,thres_h,thres_v]
                            update_times +=1
                            #print(K) 
                            for j in range(member_size):
                                bufferQue[j].clear_count()
                            #K = 3 
                            NP_local = [NP_df.copy(),NP_df.copy(),NP_df.copy()]
                            continue_error_count = [0,0,0] 
                    
                for i in range(member_size):
                    #print(error_l[i])
                    tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(label_l[i],error_l[i],datasize-bufferSize)
                    #['Exp','ID','anomalyType','TN','FN','FP','TP','ACC','FPR','TPR','P','F1','Update_times','runtime']
                    s1 = pd.Series([exp_str,i, typeName,type, tn,fn,fp,tp,acc,fpr,tpr,p,f1,update_times,time.time()-timestamp],
                            index= Structure.Output_DF_Type)
                    Output_DF = Structure.Output_DF.copy() 
                    Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
                    Output_DF.to_csv(exp_str+'HGDB.csv',header=0,mode='a')
            #one test
                #print('update_times',update_times)
        times+=1
    #Output_DF.to_csv(exp_str+'.csv')

if __name__ == '__main__':## 其他
    
    
    for i in range(3):
        print(i)
        run(i)
    
    