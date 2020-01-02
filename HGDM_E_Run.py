#coding:utf-8
import Fun
import Node
import Structure
import pandas as pd
import numpy as np
import random
import Queue
from Tools.commonFun import sort_byNum,getMaxChebyshevDistance_index
from Tools.InsertNoise import insert_outlier_error,insert_anomaly
random.seed = 0
import time
import yaml
#using incremental learning, basic edition from HODM.py
#记录多次实验的性能
save_type = ['ID','TN','FN','FP','TP','ACC','FPR','TPR','P','Recv_pkt','Send_pkt','Update_times','Rate_of_DetectedArea','Max_Storage']

def update_(df1,H_range,sample_for_k):
    global_info_l = []
    NP_df_l = [] 
    K_l = []
    local_info1 = Fun.get_localMN_info(df1)  
    #
    #整合三个df，获取global data
    MN_info = Structure.MN_selfconfigure_DF.copy()
    MN_info = MN_info.append(local_info1,ignore_index=True,sort=False)
    
    #print(MN_info)
    for H in H_range:
        print(H)

        global_info = Fun.get_Normal_Profile(MN_info,H)

        normdf1 = Fun.localdata_norm(global_info,df1.copy())
        pos_local1,num_local1,maha_last1 = Fun.local_data_distribution(global_info,normdf1)

    
        #get K*
        K_local1 = Fun.calculate_K(normdf1,sample_for_k,global_info,pos_local1,num_local1)

        pos_global = pos_local1
        num_global = num_local1

        NP_df = sort_byNum(pos_global,num_global)
        K_ = max(1,(int)(K_local1))
        print(K_)
        #print(NP_df)
        global_info_l.append(global_info)
        NP_df_l.append(NP_df)
        K_l.append(K_)
    out_K = (int)(np.mean(K_l))
    if out_K >4:
        out_K = 4
    return  global_info_l, NP_df_l, out_K


    #return global_info, NP_df , K_


def online_detetion(NP_l,K,Global_info_l,s1):
    normal_c = 0
    normal_e = 0
    counts =[]
    sum_e = 0 
    for i in range(len(NP_l)):
        NP_ = NP_l[i]
        K_ = K
        Global_info_ = Global_info_l[i]
        data_is_normal, cells_num,maha_now,pos_now, count,sum = Fun.onePoint_detect_allL1(
                                                         NP_['pos'].tolist(),NP_['num'].tolist(),K_,
                                                         Global_info_,s1['Temperature'],
                                                         s1['Humidity'],s1['Voltage'],1) 
        #result.append(data_is_normal)
        if data_is_normal:
            normal_c +=1
        else:
            normal_e +=1
        counts.append(count)
        sum_e += sum
    result = sum_e/len(NP_l)
    if  result> K:
        return True
    else:
        return False

def getHrange(h_min,h_max):

    if h_min - 1*(h_max-h_min) >0:
        hmin_ = h_min - 1*(h_max-h_min)
    else:
        hmin_ = h_min
    hmax_ = h_max + 1*(h_max-h_min)
    hrange = [val/100 for val in range((int)(hmin_*100),(int)(hmax_*100))]  # HGDM_
    return hrange

def run(exp):
    exp_str = 'E'+str(exp)  #实验名 E0、E1、E2
    #output performance

    Output_DF = Structure.Output_DF.copy() 
    Output_DF.to_csv('HGDBE'+exp_str+'.csv',mode='a')
    
    _PARAMS_PATH = "params.yaml"  #读取参数文件
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)

    repeat_time = modelParams['CommonParams']['repeat_time']  #重复实验次数
    Filled_DF_Type = ['Temperature','Humidity','Voltage']     #intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str] # anomaly parameters
    attributes = modelParams['CommonParams']['attributes']  # 属性维度
    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate'])  #插入异常比例
    #datasize = modelParams['CommonParams']['datasize']  #总数据量
    datasize = 8000   # 2000 train 6000test
    # MN
    #bufferSize = modelParams['SNParams']['buffer_size']  #4.23 缓存大小
    bufferSize = 2000 #预先训练数据量
    store_pro = modelParams['SNParams']['store_pro'] #节点存储数据概率
    sampleforK = (int)(modelParams['SNParams']['sample_size_rate']*bufferSize) #更新时采样数
    anomaly_num = (int)((datasize - bufferSize)*anomalyRate) #异常数量
    H_l,H_h = Fun.get_HG_H(attributes,bufferSize) #H 边长值
    H_range = [val/100 for val in range((int)(H_l*100-2),(int)(H_h*100+2))]  #H边长范围
    anomalyType = modelParams['anomaly_type']   #异常名，异常类型词典 
    datefile1 = datafile_dic['data1']   #'datasets/node43.csv'
    try:
        dataframe1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    times = 0           
    while times < repeat_time: #重复实验数
        for typeName in anomalyType: #异常类型  #anomaly type normal outlier constant noise
            for subtype in anomalyType[typeName]: #0：T 1：H 2：V
                type =  anomalyType[typeName][subtype] # 异常类型 list
                timestamp = time.time() #程序运行时间
                update_times = 0 #更新参数
                data1 = dataframe1[0:datasize].copy()  # 复制原数据
                outlier_pos = [] # 插入异常
                data1,outlier_pos =  insert_anomaly(data1, bufferSize, anomaly_num, typeName, type, delta_mean = 0.5, delta_std_times = 1.5)
                data1.to_csv("HGDB_E withnoise.csv") # 存储插入异常的数据
                queue = Queue.MNQueue(bufferSize)  #初始化填充
                for i in range(bufferSize):
                    series = data1.iloc[[i]]
                    queue.enqueue(series)  
               
                if queue.is_trigger_update():  #预先训练
                    Global_info_e , NP_df_e , K = update_(queue.df, H_range, sampleforK)
                    queue.clear_count()
                    update_times +=1
                
                #记录异常
                error_l = []
                error_continue_count = 0
                #ount = 0
                #开始检测
                for row in range(bufferSize,datasize):
                    series = data1.loc[row,:]
                    #print(series)
                    is_normal = online_detetion(NP_df_e,K,Global_info_e,series)    
                    if is_normal:
                        p = np.random.rand()  ##enqueue
                        #print(p)              #产生一个概率数，
                        if p <= store_pro:    #概率存储
                            queue.enqueue(series)
                        error_continue_count = 0
                    else:
                        print(row,'detected',is_normal)
                        error_l.append(row)
                        error_continue_count +=1
                        

                    #检测更新
                    if queue.is_trigger_update() or  error_continue_count > 5:
                        Global_info_e , NP_df_e , K = update_(queue.df, H_range, sampleforK)
                        queue.clear_count()
                        update_times +=1
                        error_continue_count = 0

                tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos,error_l,datasize-bufferSize)
                s1 = pd.Series([exp_str,i, typeName,type, tn,fn,fp,tp,acc,fpr,tpr,p,f1,update_times,time.time()-timestamp],
                            index= Structure.Output_DF_Type)
                Output_DF = Structure.Output_DF.copy() 
                Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
                Output_DF.to_csv('HGDBE'+exp_str+'.csv',header=0,mode='a')
                #one test
        times+=1


if __name__ == '__main__':## 其他
    for i in range(3):
        print(i)
        run(i) 