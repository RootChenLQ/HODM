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
def update(df1,begin,end,H_range,sample_for_k):
    global_info_l = []
    NP_df_l = [] 
    K_l = []
    local_info1 = Fun.get_localMN_info(data1[begin:end])  
    #local info
    local_info1 = Fun.get_localMN_info(data1[begin:end])
   
    #整合三个df，获取global data
    MN_info = Structure.MN_selfconfigure_DF.copy()
    MN_info = MN_info.append(local_info1,ignore_index=True,sort=False)
    
    #print(MN_info)
    for H in H_range:
        print(H)
        #getlocal pos
        #print(H_range)
        global_info = Fun.get_Normal_Profile(MN_info,H)
        #print(global_info)
        #normlaize local data.  get local pos array num array
        normdf1 = Fun.localdata_norm(global_info,data1[begin:end].copy())
        pos_local1,num_local1,maha_last1 = Fun.local_data_distribution(global_info,normdf1)
        #print(pos_local1)
        #print(num_local1)
        #normdf2 = Fun.localdata_norm(global_info,data2[begin:end].copy())
        #pos_local2,num_local2,maha_last2 = Fun.local_data_distribution(global_info,normdf2)
        #print(pos_local2)
        #print(num_local2)
        #normdf3 = Fun.localdata_norm(global_info,data3[begin:end].copy())
        #pos_local3,num_local3,maha_last3 = Fun.local_data_distribution(global_info,normdf3)
        #print(pos_local3)
        #print(num_local3)
    
        #get K*
        K_local1 = Fun.calculate_K(normdf1,sample_for_k,global_info,pos_local1,num_local1)
        #K_local2 = Fun.calculate_K(normdf2,sample_for_k,global_info,pos_local2,num_local2)
        #K_local3 = Fun.calculate_K(normdf3,sample_for_k,global_info,pos_local3,num_local3)
        #print(K_local1,K_local2,K_local3)
        # get global pos_array, global K
        pos_global = pos_local1
        num_global = num_local1
        #pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local2,num_local2)
        #pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local3,num_local3)
        #print(pos_global)
        #print(num_global)
        NP_df = sort_byNum(pos_global,num_global)
        K_ = max(1,(int)(K_local1))
        print(K_)
        #print(NP_df)
        global_info_l.append(global_info)
        NP_df_l.append(NP_df)
        K_l.append(K_)
    return  global_info_l, NP_df_l, K_l



    #return global_info, NP_df , K_
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
        #getlocal pos
        #print(H_range)
        global_info = Fun.get_Normal_Profile(MN_info,H)
        #print(global_info)
        #normlaize local data.  get local pos array num array
        normdf1 = Fun.localdata_norm(global_info,data1.copy())
        pos_local1,num_local1,maha_last1 = Fun.local_data_distribution(global_info,normdf1)
        #print(pos_local1)
        #print(num_local1)
        #normdf2 = Fun.localdata_norm(global_info,data2[begin:end].copy())
        #pos_local2,num_local2,maha_last2 = Fun.local_data_distribution(global_info,normdf2)
        #print(pos_local2)
        #print(num_local2)
        #normdf3 = Fun.localdata_norm(global_info,data3[begin:end].copy())
        #pos_local3,num_local3,maha_last3 = Fun.local_data_distribution(global_info,normdf3)
        #print(pos_local3)
        #print(num_local3)
    
        #get K*
        K_local1 = Fun.calculate_K(normdf1,sample_for_k,global_info,pos_local1,num_local1)
        #K_local2 = Fun.calculate_K(normdf2,sample_for_k,global_info,pos_local2,num_local2)
        #K_local3 = Fun.calculate_K(normdf3,sample_for_k,global_info,pos_local3,num_local3)
        #print(K_local1,K_local2,K_local3)
        # get global pos_array, global K
        pos_global = pos_local1
        num_global = num_local1
        #pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local2,num_local2)
        #pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local3,num_local3)
        #print(pos_global)
        #print(num_global)
        NP_df = sort_byNum(pos_global,num_global)
        K_ = max(1,(int)(K_local1))
        print(K_)
        #print(NP_df)
        global_info_l.append(global_info)
        NP_df_l.append(NP_df)
        K_l.append(K_)
    out_K = (int)(np.mean(K_l))
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
    '''
    print(K_l)
    K = max(1,(int)(np.mean(K_l)))
    print(sum_e/len(K_l))
    print(K)
    '''
    result = sum_e/len(NP_l)
    if  result> K:
        return True
    else:
        return False
    '''
    print('K',K_l)
    print('counts',counts)
    print('True:False',normal_c ,':',normal_e)
    if normal_c >=2:
        return True
    else:
        return False
    '''
    #return data_is_normal

def getHrange(h_min,h_max):

    if h_min - 1*(h_max-h_min) >0:
        hmin_ = h_min - 1*(h_max-h_min)
    else:
        hmin_ = h_min
    hmax_ = h_max + 1*(h_max-h_min)
    hrange = [val/100 for val in range((int)(hmin_*100),(int)(hmax_*100))]  # HGDM_
    return hrange
if __name__ == '__main__':## 其他
    times = 0
    #load parameters
    df = pd.DataFrame(columns = save_type) 
    _PARAMS_PATH = "params.yaml"
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    
    while times < 1:
        Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记

#intel 1
        
        datefile1 = 'datasets/E0/node43.csv'
        datefile2 = 'datasets/E0/node44.csv'
        datefile3 = 'datasets/E0/node45.csv'

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

        bufferSize = modelParams['SNParams']['buffer_size']  #4.23
        bufferSize = 2000
        attributes = modelParams['CommonParams']['attributes']
        pos_buffer_size = modelParams['CommonParams']['pos_buffer_size']
        q = modelParams['CHParams']['member_size'] #三个节点
        sink = Node.CH_Node(0,q,attributes,modelParams['SNParams']['buffer_size'])
        sqrt_thres = modelParams['SNParams']['sqrt_thres']
        store_pro = modelParams['SNParams']['store_pro']
        sampleforK = (int)(modelParams['SNParams']['sample_size_rate']*bufferSize)
        H_l,H_h = Fun.get_HG_H(attributes,bufferSize)

        H_range = [val/100 for val in range((int)(H_l*100),(int)(H_h*100))]  # HGDM_

        #datalen = modelParams['CommonParams']['datasize']
        datalen = 8000
        data1 = data1[0:datalen].copy()
#         print(len(data1))
        '''
        data2 = data2[0:datalen].copy()
        #         print(len(data2))
        data3 = data3[0:datalen].copy()
        '''
        #不插入异常
        outlier_pos1 = []
        #data1,outlier_pos1,noise_data = Fun.insert_noise_error(data1,data2,data3,300,5,'Intel')
        start = bufferSize
        #data1.to_csv("withoutnoise.csv")
        #####
        # 插入异常的数据
        size = (datalen - bufferSize)*0.05
        error_type = 'constant'
        type_l = [0]

        #data1,outlier_pos1  = insert_outlier_error(data1,[0], start,300)
        data1,outlier_pos1 =  insert_anomaly(data1, start, size, error_type, type_l, delta_mean = 2, delta_std_times = 1.5)
        data1.to_csv("HGDB_E withnoise.csv")
        data1['Temperature'].plot()
        outlier_pos2 = []
        outlier_pos3 = []
#         bufferSize = 300
        #初始化填充
        queue = Queue.MNQueue(bufferSize)
        for i in range(bufferSize):
            series = data1.iloc[[i]]
            queue.enqueue(series)  
       # lls llse max min 
       #pretrain 300
        #begin = 0 
        #Global_info_list , NP_df_list , K_list = update(data1,begin,bufferSize,H_range,sampleforK)
        if queue.is_trigger_update():

            Global_info_e , NP_df_e , K = update_(queue.df, H_range, sampleforK)
            #update_times +=1
            queue.clear_count()
            
       
        recorddf1 = Structure.QueueBuffer_DF.copy()    # record normal data
        #recorddf2 = Structure.QueueBuffer_DF.copy()   
        #recorddf3 = Structure.QueueBuffer_DF.copy()  

        error_l = []

        count = 0
        for row in range(bufferSize,datalen):
            ## online detection node8
            print(row)
            series = data1.loc[row,:]
            if row in outlier_pos1:
                print('Dubug')
            print(series)
            is_normal = online_detetion(NP_df_e,K,Global_info_e,series)
            count +=1
            ##enqueue
            print(is_normal)
            #NP_df,NP_local = increament_learning2(NP_df,NP_local,pos,K)
            #increament_learning3(NP_df,NP_local,pos,K,attributes,Global_info.loc[0,'B'])



            if is_normal:
                p = np.random.rand()  
                print(p)                       #产生一个概率数，
                if p <= store_pro: 
                    queue.enqueue(series)
            else:
                error_l.append(row)
            
            #存储
            #概率存储
           
            #坚持更新
            if queue.is_trigger_update():

                Global_info , NP_df , K = update_(queue.df, H_range, sampleforK)
                queue.clear_count()
            '''
            if count%bufferSize ==0:
                count = 0 
                Global_info_list , NP_df_list , K_list = update(data1,row - bufferSize,row,H_range,sampleforK)
            '''
        print(len(error_l))
        print(error_l)
        print(len(outlier_pos1))
        print(outlier_pos1)
        #Fun.compute_same_num(outlier_pos1,error_l,datalen-bufferSize)
        tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(outlier_pos1,error_l,datalen-bufferSize)
       #one test
       
        times+=1