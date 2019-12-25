#coding:utf-8
import Fun
import Node
import Structure
import pandas as pd
import numpy as np
import random
from Tools.commonFun import sort_byNum,getMaxChebyshevDistance_index
from Tools.InsertNoise import *
random.seed = 0
import time
import yaml
import Queue
#using incremental learning, basic edition from HODM.py
#记录多次实验的性能

def statistic_analysis(df,series,size,precision_dic):
    #df recent data buffer of size n
    precision_l = [precision_dic['temperature'],precision_dic['humidity'],precision_dic['voltage']]
    df1 = df.tail(size).copy()
    mean = df1.mean()
    std = df1.std()
    error_l = []

    for i in range(df.shape[1]):
        t = series[i]
        # 精度判断
        delta = abs(t- mean[i])
        if delta < precision_l[i]:
            error_l.append(0)
        else:
            # 统计判断
            if abs(delta) < 3*std[i]:
                error_l.append(0)
            else:
                error_l.append(1)
    return error_l
    
def update(bufferQue,H,sample_for_k):
    MN_info = Structure.MN_selfconfigure_DF.copy()
    for i in range(len(bufferQue)):
        local_info =  Fun.get_localMN_info(bufferQue[i].df)
        MN_info = MN_info.append(local_info,ignore_index=True,sort=False)
    #local info
    #local_info1 = Fun.get_localMN_info(data1[begin:end])
    #local_info2 = Fun.get_localMN_info(data2[begin:end])
    #local_info3 = Fun.get_localMN_info(data3[begin:end])
    #整合三个df，获取global data
    #MN_info = Structure.MN_selfconfigure_DF.copy()
    #MN_info = MN_info.append(local_info1,ignore_index=True,sort=False)
    #MN_info = MN_info.append(local_info2,ignore_index=True,sort=False)
    #MN_info = MN_info.append(local_info3,ignore_index=True,sort=False)
    #print(MN_info)
    #getlocal pos
    #print(H_range)
    global_info = Fun.get_Normal_Profile(MN_info,H)
    #print(global_info)
    #normlaize local data.  get local pos array num array
    pos_global,num_global,K = [],[],[]
    for i in range(len(bufferQue)):
        normDF = Fun.localdata_norm(global_info,bufferQue[i].df.copy())
        #get local pos num
        pos_local,num_local,maha_last = Fun.local_data_distribution(global_info,normDF)
        #get local K
        K_local = Fun.calculate_K(normDF,sample_for_k,global_info,pos_local,num_local)
        K.append(K_local)
        pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local,num_local)

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
    #K_local1 = Fun.calculate_K(normdf1,sample_for_k,global_info,pos_local1,num_local1)
    #K_local2 = Fun.calculate_K(normdf2,sample_for_k,global_info,pos_local2,num_local2)
    #K_local3 = Fun.calculate_K(normdf3,sample_for_k,global_info,pos_local3,num_local3)
    #print(K_local1,K_local2,K_local3)
    # get global pos_array, global K
    #pos_global = pos_local1
    #num_global = num_local1
    #pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local2,num_local2)
    #pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local3,num_local3)
    #print(pos_global)
    #print(num_global)
    NP_df = sort_byNum(pos_global,num_global)
    #K_ = (int)((K_local1+K_local2+K_local3)/3)
    K_ = np.floor(np.mean(K))
    #print(NP_df)
    return global_info, NP_df , K_
'''
def update(df1,df2,df3,begin,end,H,sample_for_k):
    #local info
    local_info1 = Fun.get_localMN_info(data1[begin:end])
    local_info2 = Fun.get_localMN_info(data2[begin:end])
    local_info3 = Fun.get_localMN_info(data3[begin:end])
    #整合三个df，获取global data
    MN_info = Structure.MN_selfconfigure_DF.copy()
    MN_info = MN_info.append(local_info1,ignore_index=True,sort=False)
    MN_info = MN_info.append(local_info2,ignore_index=True,sort=False)
    MN_info = MN_info.append(local_info3,ignore_index=True,sort=False)
    print(MN_info)
    #getlocal pos
    print(H_range)
    global_info = Fun.get_Normal_Profile(MN_info,H)
    print(global_info)
    #normlaize local data.  get local pos array num array
    normdf1 = Fun.localdata_norm(global_info,data1[begin:end].copy())
    pos_local1,num_local1,maha_last1 = Fun.local_data_distribution(global_info,normdf1)
    #print(pos_local1)
    #print(num_local1)
    normdf2 = Fun.localdata_norm(global_info,data2[begin:end].copy())
    pos_local2,num_local2,maha_last2 = Fun.local_data_distribution(global_info,normdf2)
    #print(pos_local2)
    #print(num_local2)
    normdf3 = Fun.localdata_norm(global_info,data3[begin:end].copy())
    pos_local3,num_local3,maha_last3 = Fun.local_data_distribution(global_info,normdf3)
    #print(pos_local3)
    #print(num_local3)
   
    #get K*
    K_local1 = Fun.calculate_K(normdf1,sample_for_k,global_info,pos_local1,num_local1)
    K_local2 = Fun.calculate_K(normdf2,sample_for_k,global_info,pos_local2,num_local2)
    K_local3 = Fun.calculate_K(normdf3,sample_for_k,global_info,pos_local3,num_local3)
    print(K_local1,K_local2,K_local3)
    # get global pos_array, global K
    pos_global = pos_local1
    num_global = num_local1
    pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local2,num_local2)
    pos_global,num_global = Fun.merge_data_distribution(pos_global,num_global,pos_local3,num_local3)
    print(pos_global)
    print(num_global)
    NP_df = sort_byNum(pos_global,num_global)
    K_ = (int)((K_local1+K_local2+K_local3)/3)
    print(NP_df)
    return global_info, NP_df , K_
'''
def online_detetion(NP_,K_,Global_info_,s1,row):
    data_is_normal, cells_num,maha_now,pos_now,count = Fun.onePoint_detect(
                                                         NP_['pos'].tolist(),NP_['num'].tolist(),K_,
                                                         Global_info_,s1['Temperature'],
                                                         s1['Humidity'],s1['Voltage'],row) 
    return data_is_normal , pos_now ,count

def increament_learning(NP_,pos,count,K,newpos_l,prop=1):
    #仅增加坐标
    #count  neighbor data 
    # 修改num量 以领域数据量为标准 包括自身网格 统计验证 
    p = np.random.rand()  
    #print(pos)                       #产生一个概率数，
    if p <= prop:  
        #change NP+ analyze the pos in NP
        if pos in NP_['pos'].tolist():
            #num = NP_.loc[NP_['pos']==pos,'num']
            num = max(1, min(np.ceil(count),int(K/2)))
            #num = K
            NP_.loc[NP_['pos'] == pos,'num']+=num
            '''
            if ((NP_.loc[NP_['pos']==pos,'num']<K).all()):
                print('in')
                print(NP_)   
            '''
        else:# pos 不在list中
            #print('not in')
            #print(NP_)
            # if there are more than K data in L1-op

            #num = max(1,min(np.ceil(count),K-1))
            #num = max(1,np.random.randint(K/2,K))
            
            if count>0:
                num = max(1, min(np.ceil(count),int(K/2)))
            #else:
            #    num = 0
            
            #if count>0:
                s = pd.Series({'pos':pos, 'num':num})
            #else:  # may error position
            #    s = pd.Series({'pos':pos, 'num':1})
                NP_ = NP_.append(s,ignore_index=True)

                newpos_l.append(pos)

            #print(NP_)
    return NP_,newpos_l
    

def increament_learning1(NP_,NP_local,pos,K,prop=1):
    #动态变动NP数据,插入最新的数据，删除数据量较少的数据
    # NP_  pos 按照num大小排列
    prob = np.random.rand()                          #产生一个概率数，
    if prob <= prop:  
        print(pos)
        #change NP+ analyze the pos in NP
        if pos in NP_['pos'].tolist():
            #num = NP_.loc[NP_['pos']==pos,'num']
            if ((NP_.loc[NP_['pos']==pos,'num']<K).all()):
                print('in')
                print(NP_)
                NP_.loc[NP_['pos']==pos,'num']+=1
                #最小值减一
                if NP_.iloc[-1]['pos'] == pos:# 如果最后一行的数据是pos
                    NP_.iloc[-2]['num']-=1
                    if NP_.iloc[-2]['num'] ==0:
                        NP_.drop([len(NP_)-2],inplace=True)
                else: #最后一行不是pos
                    NP_.iloc[-1]['num']-=1
                    if NP_.iloc[-1]['num'] ==0:
                        NP_.drop([len(NP_)-1],inplace=True)
            #排列NP
            NP_ = NP_.sort_values(['num'], ascending=[False])
            NP_ = NP_.reset_index(drop=True)
            print(NP_)
        else:# pos 不在list中
            print('not in')
            print(NP_)
            #最小值减一 在末尾加1
            NP_.iloc[-1]['num'] -= 1
            if NP_.iloc[-1]['num'] ==0:
                    NP_.drop([len(NP_)-1],inplace=True)
            s = pd.Series({'pos':pos, 'num':1})
            NP_.append(s,ignore_index=True)
            print(NP_)
         
def increament_learning2(NP_old,NP_local,pos,K,prop=1):
    #动态变动NP数据,插入最新的数据，随机删除原始NP中的点，不调整K值
    # NP_  pos 按照num大小排列
    prob = np.random.rand()                          #产生一个概率数，
    if prob <= prop:   #概率存储
        #print(pos)
        #change NP+ analyze the pos in NP
        if pos in NP_local['pos'].tolist():  #如果pos 在NP_local中
            #num = NP_.loc[NP_['pos']==pos,'num']
            #print('in')
            pos_index = NP_local[(NP_local.pos==pos)].index.tolist()

            if (NP_local.loc[pos_index[0],'num']<K):
                #print('insert pos')
                #print(NP_local)
                NP_local.loc[pos_index[0],'num']+=1

                #随机选择，原始数组中的pos 减一
                index = np.random.randint(0,len(NP_old)-1)
                changePos = NP_old.loc[index,'pos']
                #print('index',index)
                #print('changePos',changePos)
                #NP_old 更新 
                if NP_old.loc[index,'num'] == 1:
                    NP_old.drop([index],inplace=True)  # 删除旧数据
                    #NP_old = NP_old.reset_index(drop=True)
                else:
                    NP_old.loc[index,'num'] -= 1
                #NP 更新
                #找到pos 对应的index
                index_ = NP_local[(NP_local.pos==changePos)].index.tolist()
                #print('index_',index_)
                if (NP_local.loc[index_[0],'num']==1):
                    NP_local.drop([index_[0]],inplace=True)
                else:
                    NP_local.loc[index_[0],'num'] -= 1
                #排列NP
                NP_local = NP_local.sort_values(['num'], ascending=[False])
                NP_local = NP_local.reset_index(drop=True)
                NP_old = NP_old.reset_index(drop=True)
            #print(NP_local)
        else:# pos 不在list中
            #print('not in')
            #print('NP_old',NP_old)
            #print('NP_local',NP_local)
            #NP_local 添加【pos ,1]
            s = pd.Series({'pos':pos, 'num':1})
            NP_local = NP_local.append(s,ignore_index=True)
            #NP_local , NP random num-1
            #随机选择，原始数组中的pos 减一
            
            index = np.random.randint(0,len(NP_old)-1)
            changePos = NP_old.loc[index,'pos']
            #print(NP_old)
            #print('index',index)
            #print('changePos',changePos)
            #NP_old 更新 
            if NP_old.loc[index,'num'] == 1:
                
                NP_old.drop([index],inplace=True)
                
            else:
                NP_old.loc[index,'num'] -= 1
            #NP 更新
            #找到pos 对应的index
            index_ = NP_local[(NP_local.pos==changePos)].index.tolist()
            if (NP_local.loc[index_[0],'num']==1):
                NP_local.drop([index_[0]],inplace=True)
            else:
                NP_local.loc[index_[0],'num'] -= 1
            #排列NP
            NP_local = NP_local.sort_values(['num'], ascending=[False])
            NP_local = NP_local.reset_index(drop=True)
            NP_old = NP_old.reset_index(drop=True)
    return NP_old,NP_local

def increament_learning3(NP_old,NP_local,pos,K,attributes,bits,prop=1):
    #删除曼哈顿距离最远的坐标
    #
    print('increament_learning3')
    index = getMaxChebyshevDistance_index(NP_old['pos'].tolist(),pos,attributes,bits)

    print(index)
    # 重新计算K值
    pass

def increament_learning4(NP_old,NP_local,pos,K,count):
    #动态变动NP数据,插入最新的数据，随机删除原始NP中的点，不调整K值
    # NP_  pos 按照num大小排列
    prob = np.random.rand()                          #产生一个概率数，
    if prob <= 1:   #概率存储




        #print(pos)
        #change NP+ analyze the pos in NP
        if pos in NP_local['pos'].tolist():  #如果pos 在NP_local中
            #num = NP_.loc[NP_['pos']==pos,'num']
            #print('in')
            pos_index = NP_local[(NP_local.pos==pos)].index.tolist()

            if (NP_local.loc[pos_index[0],'num']<K):
                #num size < K
                #print('insert pos')
                #print(NP_local)
                NP_local.loc[pos_index[0],'num']+=1

                #随机选择，原始数组中的pos 减一
                index = np.random.randint(0,len(NP_old)-1)
                changePos = NP_old.loc[index,'pos']
                #print('index',index)
                #print('changePos',changePos)
                #NP_old 更新 
                if NP_old.loc[index,'num'] == 1:
                    NP_old.drop([index],inplace=True)  # 删除旧数据
                    #NP_old = NP_old.reset_index(drop=True)
                else:
                    NP_old.loc[index,'num'] -= 1
                #NP 更新
                #找到pos 对应的index
                index_ = NP_local[(NP_local.pos==changePos)].index.tolist()
                #print('index_',index_)
                if (NP_local.loc[index_[0],'num']==1):
                    NP_local.drop([index_[0]],inplace=True)
                else:
                    NP_local.loc[index_[0],'num'] -= 1
                #排列NP
                NP_local = NP_local.sort_values(['num'], ascending=[False])
                NP_local = NP_local.reset_index(drop=True)
                NP_old = NP_old.reset_index(drop=True)
            #print(NP_local)
        else:# pos 不在list中
            #print('not in')
            #print('NP_old',NP_old)
            #print('NP_local',NP_local)
            #NP_local 添加【pos ,1]
            s = pd.Series({'pos':pos, 'num':1})
            NP_local = NP_local.append(s,ignore_index=True)
            #NP_local , NP random num-1
            #随机选择，原始数组中的pos 减一
            
            index = np.random.randint(0,len(NP_old)-1)
            changePos = NP_old.loc[index,'pos']
            #print(NP_old)
            #print('index',index)
            #print('changePos',changePos)
            #NP_old 更新 
            if NP_old.loc[index,'num'] == 1:
                
                NP_old.drop([index],inplace=True)
                
            else:
                NP_old.loc[index,'num'] -= 1
            #NP 更新
            #找到pos 对应的index
            index_ = NP_local[(NP_local.pos==changePos)].index.tolist()
            if (NP_local.loc[index_[0],'num']==1):
                NP_local.drop([index_[0]],inplace=True)
            else:
                NP_local.loc[index_[0],'num'] -= 1
            #排列NP
            NP_local = NP_local.sort_values(['num'], ascending=[False])
            NP_local = NP_local.reset_index(drop=True)
            NP_old = NP_old.reset_index(drop=True)
    return NP_old,NP_local

def run(exp):
    #params.yaml output data
    #选择实验数据
    exp_str = 'E'+str(exp)
    #加载实验参数
    times = 0
    update_times = 0
    #load parameters 记录实验结果
    Output_DF = Structure.Output_DF.copy() 

    _PARAMS_PATH = "params.yaml"
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    repeat_time = modelParams['CommonParams']['repeat_time']
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datafile_dic =  modelParams['datafile'][exp_str]
    # parameter set
    # Presition:
    presition_dic = modelParams['Presition']
    #common
    attributes = modelParams['CommonParams']['attributes']  # 3
    anomalyRate = (float) (modelParams['CommonParams']['anomalyRate']) 

    #pos_buffer_size = modelParams['CommonParams']['pos_buffer_size']  #
    datasize = modelParams['CommonParams']['datasize']
    #datasize = 6000
    statistic_analysis_data_size = modelParams['CommonParams']['statistic_analysis_data_size']   
    # MN
    bufferSize = modelParams['SNParams']['buffer_size']  #4.23 
    #sqrt_thres = modelParams['SNParams']['sqrt_thres']
    store_pro = modelParams['SNParams']['store_pro']
    sampleforK = (int)(modelParams['SNParams']['sample_size_rate']*bufferSize)
    #Sink 
    member_size = modelParams['CHParams']['member_size'] #三个节点
    #sink = Node.CH_Node(0,member_size,attributes,bufferSize)
    #H_range = [val/100 for val in range((int)(H_l*100),(int)(H_h*100))]  # HGDM_
    H_l,H_h = Fun.get_HG_H(attributes,bufferSize)
    H = H_l


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
    while times < repeat_time:

        for typeName in anomalyType:
            for subtype in anomalyType[typeName]:
                type =  anomalyType[typeName][subtype]
                #print(type)
                #type = [0]
                timestamp = time.time()
                #拷贝数据集
                data1 = dataframe1[0:datasize].copy()
                data2 = dataframe2[0:datasize].copy()
                data3 = dataframe3[0:datasize].copy()
                update_times = 0
                ###################################
                #插入异常
                outlier_pos1 = []
                start = bufferSize
                anomaly_num = (int)((datasize - bufferSize)*anomalyRate)
                #选择插入异常类型
                #type = [0,1,2]
                #data1,outlier_pos1,noise_data = Fun.insert_noise_error(data1,data2,data3,1000,2,'Intel')
                #data1,outlier_pos1,noise_data = Fun.insert_noise(data1,bufferSize,int(datasize*0.05),3,2)
                #data1,outlier_pos1  = insert_outlier_error(data1,type, start,300)
                #data1,outlier_pos1  = insert_constant_error(data1,type, start,5)
                
                data1,outlier_pos1  = insert_noise_error(data1,type,start,anomaly_num,0.5,1.5)

                '''output noie data'''
                #data1.to_csv("withoutnoise.csv")
                data1.to_csv("datasets/withnoise.csv")
                #data1['Temperature'].plot()
                outlier_pos2 = []
                outlier_pos3 = []

                dataArr = [data1.copy(),data2.copy(),data3.copy()]
                label_l = [outlier_pos1,outlier_pos2,outlier_pos3]
                #pretrain 300
                #buffer 存储 正确数据，提出明显异常
                queue1 = Queue.MNQueue(bufferSize)
                queue2 = Queue.MNQueue(bufferSize)
                queue3 = Queue.MNQueue(bufferSize)
                bufferQue = [queue1, queue2, queue3]
        
                for i in range(bufferSize):
                    for index in range(member_size):
                        series = dataArr[index].iloc[[i]]
                        bufferQue[index].enqueue(series)  
                
                #begin = 0 
                if bufferQue[0].is_trigger_update():
                    Global_info , NP_df , K = update(bufferQue,H,sampleforK)
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
                new_pos = []
                # start online operation
                #data_size = 2000
                for row in range(bufferSize,datasize):
                    ## online detection node8
                    
                    for i in range(member_size):
                        #print(i,' ',row)
                        #if row > 537 and i == 0:
                        #    print('test')
                        ############
                        series = dataArr[i].loc[row,:]
                        #print(series)
                        is_normal,pos,count = online_detetion(NP_local[i],K,Global_info,series,row)
                        ##enqueue
                        #print(is_normal)
                        #NP_df,NP_local = increament_learning2(NP_df,NP_local,pos,K)
                        #increament_learning3(NP_df,NP_local,pos,K,attributes,Global_info.loc[0,'B']
                        
                        NP_local[i],new_pos = increament_learning(NP_local[i],pos,count,K,new_pos,prop=store_pro)
                        
                        #NP_df,NP_local = increament_learning4(NP_df,NP_local,pos,K,count)
                        #if is_normal:
                        if is_normal:
                            bufferQue[i].enqueue(series)
                            #NP_df,NP_local = increament_learning2(NP_df,NP_local,pos,K)
                            #NP_df,NP_local = increament_learning2(NP_df,NP_local,pos,K)
                            pass
                        else:
                            result_l = statistic_analysis(bufferQue[i].df,series,statistic_analysis_data_size,presition_dic)
                            normal_l = True
                            for j in range(len(result_l)):
                                if result_l[j] ==1:
                                    normal_l = False
                                    #print(j,'may exits error')
                            if normal_l:
                                bufferQue[i].enqueue(series)
                            else:
                                #print(result_l)
                                error_l[i].append(row)
                        
                    if bufferQue[i].is_trigger_update():
                        Global_info , NP_df , K = update(bufferQue,H,sampleforK)
                        update_times +=1
                        #print(K) 
                        for j in range(member_size):
                            bufferQue[j].clear_count()
                        #K = 3 
                        NP_local = [NP_df.copy(),NP_df.copy(),NP_df.copy()]
                    
                for i in range(member_size):
                    #print(error_l[i])
                    tn,fn,fp,tp, acc,fpr,tpr,p,f1 = Fun.compute_performent(label_l[i],error_l[i],datasize-bufferSize)
                    #['Exp','ID','anomalyType','TN','FN','FP','TP','ACC','FPR','TPR','P','F1','Update_times','runtime']
                    s1 = pd.Series([exp_str,i, typeName,type, tn,fn,fp,tp,acc,fpr,tpr,p,f1,update_times,time.time()-timestamp],
                            index= Structure.Output_DF_Type)
                    Output_DF = Output_DF.append(s1,ignore_index = True,sort=False)
            #one test
                #print('update_times',update_times)
        times+=1
    Output_DF.to_csv(exp_str+'.csv')

if __name__ == '__main__':## 其他
    for i in range(3):
        run(i)
    
    