#coding:utf-8
import Structure
import Fun
import Queue
import pandas as pd
import numpy as np
import random
import math
import time
class CH_Node():
    def __init__(self,ID,MH_size,attributes,bufferlen):
#         print('CH_Node %d creating' %ID)
        self.ID = ID       
        self.MN_size = MH_size    #簇内节点个数
        #超网格格式更新参数
        self.isUpdate = False     #是否更新
        self.Call_UpdateTimes = 0 ##如果call_Update_times >= MN_size, 则跟新
        self.MN_info = Structure.MN_selfconfigure_DF.copy() #记录簇内节点数据信息
        self.CH_info = Structure.CH_configure_DF.copy() #记录全局数据信息  #包含HyperGrid信息
        self.pos_all =[]   #记录节点pos数据
        self.num_all = []  #记录num数
        #更新K值
        self.k_list = []  #记录簇内节点发来的K值，求平均为阈值
        self.K_ = 0       # 全簇的阈值
        self.echo = 0                      #记录hypergrid_info 更新版本。同步簇内版本号
        self.eventVal = 0                  #标记网络事件
        self.is_first_receive_pos = True   #标记簇头节点第一次接收到pos信息   
        self.Recv_pkt_num = 0        #发包数
        self.Send_pkt_num = 0        #收包数
        self.attributes = attributes  #for H
        self.bufferlen = bufferlen
        ######收发包量
        
    #########################更新网络结构模型#######################################
    #簇头节点接收更新数据点。
    def get_eventVal(self):                #返回事件值
        return self.eventVal

    def receive_call_update(self):  #接收到更新请求  5-29
        self.Recv_pkt_num +=1   

    def receive_call_update_data(self,mn_df):#收到簇内节点的MH_info，记录，统计申请更新个数,判断是否更新，
        #接收来自子节点的 申请更新数据包：包含各属性数值和LLS，各属性平方和LLSS、各属性的最大值MAX、最小值MIN
        self.Call_UpdateTimes +=1
        self.MN_info = self.MN_info.append(mn_df,ignore_index=True,sort=False)
#         print('sink mn_info',self.MN_info)
        if self.Call_UpdateTimes >= self.MN_size:
            self.send_NP_info()
            self.Call_UpdateTimes = 0
        else:
            self.eventVal = 0
#             print('Call_UpdateTimes',self.Call_UpdateTimes)
        #Call_UpdateTimes ++
        #判断是否收到所有簇的请求更新
        self.Recv_pkt_num += 1        #发包数
#       print('sink receive_call_update Recv_pkt +1 num= %d' %self.Recv_pkt_num)
        
    def sink_mn_call_update(self):
        self.Send_pkt_num +=1

    #发送汇总的NP信息
    def send_NP_info(self):   #发送HG_info  #发送簇内的统计信息（平均值，方差等）：用于标准化
        #发送HG信息，包含Max，Min，mean,std,
#         print('send_NP_info')
        ###########
#         print(self.MN_info)

        H_l,H_h = Fun.get_HG_H(self.attributes,self.bufferlen)
        self.CH_info = Fun.get_Normal_Profile(self.MN_info,H_h) #待定义
#         print(self.CH_info)
        #self.CH_info['H']
        self.MN_info = Structure.MN_selfconfigure_DF.copy()#记录簇内节点数据信息 #清空MN_info 用于下次更新  
        self.eventVal = 0x1222 #直接使用
        #print('self.CH_info',self.CH_info)
        self.Send_pkt_num += 1        #收包数
#         print('sink send_NP_info Send_pkt +1 num = %d'%self.Send_pkt_num)
        
    def receive_pos_k(self,pos_l,num_l,k): #收到簇内节点发来的pos num，及K* 信息，用于计算全簇的pos num K信息
        #接收到簇内节点的K值，及pos,num。信息
        #汇总pos,num，信息
#         print('ID %d receive_pos_k '%self.ID)
        
        self.Recv_pkt_num += 1        #发包数
#         print('sink receive_pos_k Recv_pkt +1 num= %d' %self.Recv_pkt_num)
        if self.is_first_receive_pos:
            self.pos_all = pos_l
            self.num_all= num_l
            self.k_list = []
            self.k_list.append(k)
            self.is_first_receive_pos = False
        else:
            self.pos_all,self.num_all = Fun.merge_data_distribution(self.pos_all,self.num_all,pos_l,num_l)
            self.k_list.append(k)
            if len(self.k_list) == self.MN_size:
                self.send_Pos_k_info()
#         print('pos_all',self.pos_all)
#         print('num_all',self.num_all)
        
    def send_Pos_k_info(self):   #发送pos_all num_all k
        #计算K
        #返回pos num k 信息
        #print( 'send_Pos_k_info')
        self.echo +=1
#         print('Now HyperGrid Versions is NO:%d'%self.echo)
        self.K_ = np.floor(np.mean(self.k_list))
        self.is_first_receive_pos = True
        self.Send_pkt_num += 1        #收包数
#         print('sink send_Pos_k_info send_pkt +1 num=%d '%self.Send_pkt_num)
        
        
    
    ##########################更新网络结构模型结束#######################################
    
    
    ##########################子节点异常事件判断#########################################
        
    def analyse_warning(self,volt): #收到簇内节点的对异常的判断
        pass
    def analyse_abnormal(self,volt): #收到簇内节点的对异常的判断
        pass
    def send_anaylyse(self,id_,val_): #发送节点的分析结果。
        self.Send_pkt_num += 1        #收包数
#         print('sink send_anaylyse send_pkt +1 num=%d '%self.Send_pkt_num)
        
    def receive_query_data(self,id_,df):
        self.Recv_pkt_num += 1        #发包数
#         print('sink receive_query_data Recv_pkt +1 num= %d' %self.Recv_pkt_num)
        
    def send_estimate(self):
        self.Send_pkt_num += 1        #收包数
#         print('sink send_estimate send_pkt +1 num=%d '%self.Send_pkt_num)
        
        
class MN_Node():
    def __init__(self,ID,memory_size,sqrt_thres = 0.5,save_prop=0.8,sampleforK = 50,pos_array_size = 10,attributes = 3):
#         print('MN_Node %d creating' %ID)
        #节点ID
        self.ID = ID       
        #模拟内存缓冲队列，先进先出
        self.queue = Queue.MNQueue(memory_size) 
        self.pos_queue = Queue.posQueue(pos_array_size,attributes)
        #事件变量，模拟无线传感器网路中的事件
        self.eventVal = 0            #为模拟簇头与簇内节点之间的交互，特加入event变量用于联系。
        #状态值设定，由于初始布置节点时，数据无预先，假设初始采集数据点全为正常。        后续添加 系统矫正
        #####预训练、检测阶段
        self.in_preTraining = True    #判断是否在预训练阶段
        
        #更新hyper grid结构时使用
        #统计本次数据的最大、最小值，累加和等。  发送给簇头
        self.MN_info = Structure.MN_selfconfigure_DF.copy()  #本地数据信息
        #接收到簇头的信息。
        self.CH_info = Structure.CH_configure_DF.copy()      #全局数据信息 
        #根据簇头发来的全簇的信息，映射数据点，完成本地K*的计算。
        self.pos_local = []            #本地pos 记录列表   
        self.num_local = []            #本地num 记录列表
        self.K_local = 0               #计算的本地K值
        #检验数据异常时使用
        self.pos_all = []              #簇内pos 记录列表
        self.num_all = []              #簇内num 记录列表
        self.K_ = 0                    #K的阈值 
        #可能边缘数据缓存数据信息， 增加系统的鲁棒性。     
        
        #warning 信息。预判断数据的趋势。
        self.is_in_warning = False            #进入示警阶段。
#         self.data_warning = np.array([0,0,0]) #记录示警的三维数据
        self.pos_warning = 0                  #警报的起始位置
        self.times_warning = 0                #警报验证的时间
        self.life_warning = 0                 #对应的loop信息，loop->0，则检验数量是否大于阈值
        self.records_warning = np.array([0,0])#警报分析后，记录警报发起点，持续时间。
        '''5.10 修改'''
        thres_times = 3
        check_times = 5
#         self.warning_thres = 3                #警告检测过程中，异常出现的次数。 para 相对于异常事件 check时间可以久些。
#         self.warning_check_times = 5          #警告检测的次数
        self.warning_thres = thres_times                #警告检测过程中，异常出现的次数。 para 相对于异常事件 check时间可以久些。
        self.warning_check_times = check_times          #警告检测的次数
 
        '''5.10 修改为' 10次'''
        #曼哈顿距离计算 记录当前点的曼哈顿距离异常
        self.maha_dis = np.zeros((1,3))   # 为方便maha_dis的叠加（vstack）， self.maha_dis  先置为[0,0,0]
        self.maha_last = np.array([])       # 存储上一个pos 信息
        self.maha_now = np.array([])        # 存储目前的pos 信息
        
        #可容忍的maha距离值内判断突变。1__  5.5
        self.accepted_maha_error_thres = np.array([1,1,1]) 
        self.maha_bytimes = np.array([])     #5.5 由于突然降温，降湿导致的，maha_last基准变化、失效。此变量用于记录每一次的记录值
        self.is_normal_change = True            #5.5 分析判断正常的突变
        '''5.10 修改'''
        self.belive_in_normal_change =0
#         self.belive_in_normal_change_thres =  3
        '''5.10 修改为'''
        self.belive_in_normal_change_thres =  thres_times
        
        self.pos_now = 0                    # 记录当前的pos 信息
        self.data_is_mahaError = False      # 记录此刻是否存在的maha异常
        self.data_is_normal = False         # 记录当前数据点检测结果，和maha_error配合使用，协调节点。
        
        
        #异常数据缓存区 
        self.in_abnormal_event = False           #记录是否为异常事件检测阶段
        self.data_abnormal = np.array([0,0,0])   #记录异常的数据信息   #无实际作用
        self.pos_abnormal = np.array([])         #异常记录的异常位置
        self.times_abnormal = 0                  #异常验证的次数。 避免单值异常，反复系统反复查询验证。
        self.life_abnormal = 0                   #对应的loop信息，loop->0，则检验数量是否大于阈值
        self.records_abnormal = np.array([0,0])  #异常分析后，记录异常发起点，持续时间。
        '''5.10 修改'''
#         self.abnorma_thres = 3                   #异常检测过程中，异常出现的次数。 para 相对于异常事件 check时间可以久些。
#         self.abnorma_check_times = 5             #异常检测的次数
        #缓存异常数据
        '''5.10 修改为'''
        self.abnorma_thres = thres_times                    #异常检测过程中，异常出现的次数。 para 相对于异常事件 check时间可以久些。
        self.abnorma_check_times = check_times             #异常检测的次数
        
        #记录异常的信息 point 用于验证
        self.detected_error = []        #记录异常信息，用于仿真。
        self.detected_warning = []      #记录警告信息，用于仿真。
        #簇内同步
        self.echo = 0 #记录hypergrid_info 更新版本  #可以防止数据的不同步，新加入的节点可以同步echo信息
        
        #后续性能对比记录数据
        self.Max_Storage = 0         #记录内存的使用情况，后期比较性能  new
        self.storage_bysample = []             #记录内存的使用情况，后期比较性能
        
        self.fixed_storage = 39+3 #bytes　4.23 +3字节 自适应mahadis_error

        #收发包量
        self.Recv_pkt_num = 0        #发包数
        self.Send_pkt_num = 0        #收包数
           
        #记录查看的超立方体的数量 用于统计DRR
        self.detect_cells = 0
        self.vol_total = 0    #每个次检测网格数于总网格数 的比例的类和
        self.detect_times = 0 #记录检测数量
        
        #记录是否需要更新
        self.needUpdate = False
        
        #记录模型更新的次数
        self.updateTimes = 0 
#         self.echo = 0 
#         self.train_var = []  #LLSE 简化版 
#         self.train_mean = [] #近似协方差计算使用
        # 4.23号添加
        self.maha_diff_thres = np.array([1,1,1])
        
        #4.25 初始化 mahadis 阈值的判断
        self.sqrt_thres = sqrt_thres
        
        #存储概率
        self.prop_threshold = save_prop
        #训练样本数
        self.sample_for_k = math.ceil(memory_size*0.3)
        
        
    ###    
    #计算本轮的数据大小 undone 
    def record_storage_use(self):  
        temp_storage = self.fixed_storage      #固定属性为39Byte
#         print('fixed_storage = %d'%self.fixed_storage)
        temp_storage +=len(self.queue.df)*3*4  #queue 大小记录，每一条记录三个属性， 每个属性4byte
#         print('queue storage = %d'%(len(self.queue.df)*3*4))
        if not self.in_preTraining:
            temp_storage+= 34                  #CH_info
#             print('CH_info storage = %d'%34)
        temp_storage += len(self.pos_all)*(2+2)   #pos_all pos_num
#         print('pos_all storage = %d'%(len(self.pos_all)*(2+2)))
        #undone
        temp_storage += len(self.maha_dis)*3*1  #警告或者异常时，mahadis存储大小
#         print('maha_dis storage = %d'%(len(self.maha_dis)*3*1))  #problem
        
        
#         temp_storage += len(self.records_warning)*2*(4+1)  #警告记录大小
#         temp_storage += len(self.records_abnormal)*2*(4+1) #异常记录大小 
        #判断本轮是否在更新
        if self.needUpdate:
            temp_storage += 51            #训练时，计算本地信息。MN_Info
            temp_storage += 1             #K*
            temp_storage += len(self.pos_local)*(2+2) # pos_local ,pos_num
        self.storage_bysample.append(temp_storage)
#         print('temp_storage',temp_storage)
#         print('Max_Storage',self.Max_Storage)
        if temp_storage > self.Max_Storage:
            self.Max_Storage = temp_storage
#         print('Max_Storage: ',self.Max_Storage)
#   检测异常状态
    def check_condition(self):
        if self.is_in_warning and self.in_abnormal_event:
            print('In Error conditions.......')
            print('In Error conditions.......')
            print('In Error conditions.......')
    #加载数据
    def load_dataset(self,loadDF,datatype):
        self.LoadDF = loadDF.copy()
        self.datapoint = 0
        self.dataType = datatype
    #预训练
    def preTrain(self,datasize):
        for rows in range(datasize):
#             print(rows)
#             print(self.LoadDF.iloc[[rows]])
            if self.datapoint<len(self.LoadDF):
                self.sample_a_data(self.datapoint,self.LoadDF.iloc[[self.datapoint]])
                self.datapoint += 1 
                self.record_storage_use() #记录此刻的内存使用
            else:
                print('Exceed the bound of dataDF')
    #返回事件值
    def get_eventVal(self):                #返回事件值
        return self.eventVal
    
    def get_ID_event_Val(self):
        return self.ID<<4 | self.eventVal

    #请求高兴
    def call_update(self):           #内存满，申请更新#MN sends a massege to HN,[max,min,lls,llss,lln]
        #compute_MH_info
#         print('ID= %d call_update'%self.ID)
        self.needUpdate = True #4.15
        self.MN_info = Fun.get_localMN_info(self.queue.df)
        self.Send_pkt_num += 1        #收包数
#         print('%d call_update send_pkt+1 num = %d' %(self.ID,self.Send_pkt_num))
        #发送 send
#         print('Node %d'%self.ID,self.MN_info)
        self.eventVal = 0x1  #sink节点0000、本节点ID self.ID、事件对应的标志位置1 请求更新0001
#         self.eventVal = 0<<8|self.ID<<4|0x1 #sink节点0000、本节点ID self.ID、事件对应的标志位置1 请求更新0001

   
        
    def receive_NP_update(self,np_):  # np  normal profile
    #接收到更新数据包事件
#         CH_configure_type = ['NT_g','NH_g','NV_g','mean_T','mean_H','mean_V','std_T','std_H',,'std_V',
#         'TemperatureMaxNorm_g','TemperatureMinNorm_g','HumidityMaxNorm_g','HumidityMinNorm_g',
#          VoltageMaxNorm_g','VoltageMinNorm_g']
        #存储CH——info 信息
        self.CH_info = np_
        self.in_preTraining = False
        self.Recv_pkt_num += 1        #发包数
#         print('%d receive_NP_update recv +1 num=%d'%(self.ID,self.Recv_pkt_num))
#         thres = 0.5
        if self.dataType =='Intel':
            if self.CH_info.at[0,'std_T'] >self.sqrt_thres:   #4.23 添加 maha_dis 自适应性
                self.maha_diff_thres[0] = 1
            else:
                self.maha_diff_thres[0] = 2
            if self.CH_info.at[0,'std_H'] >self.sqrt_thres:
                self.maha_diff_thres[1] = 1
            else:
                self.maha_diff_thres[1] = 2
            if self.CH_info.at[0,'std_V'] >self.sqrt_thres:
                self.maha_diff_thres[2] = 1
            else:
                self.maha_diff_thres[2] = 2
#             self.maha_diff_thres = np.array([2,2,2])
        elif self.dataType == 'Sensorscope':
            if self.CH_info.at[0,'std_T'] >self.sqrt_thres:   #4.23 添加 maha_dis 自适应性
                self.maha_diff_thres[0] = 1
            else:
                self.maha_diff_thres[0] = 2
            if self.CH_info.at[0,'std_H'] >self.sqrt_thres:
                self.maha_diff_thres[1] = 1
            else:
                self.maha_diff_thres[1] = 2
            if self.CH_info.at[0,'std_V'] >self.sqrt_thres:
                self.maha_diff_thres[2] = 1
            else:
                self.maha_diff_thres[2] = 2
        elif self.dataType == 'Xi-eye':
            if self.CH_info.at[0,'std_T'] >self.sqrt_thres:   #4.23 添加 maha_dis 自适应性
                self.maha_diff_thres[0] = 1
            else:
                self.maha_diff_thres[0] = 2
            if self.CH_info.at[0,'std_H'] >self.sqrt_thres:
                self.maha_diff_thres[1] = 1
            else:
                self.maha_diff_thres[1] = 2
            if self.CH_info.at[0,'std_V'] >500:
                self.maha_diff_thres[2] = 1
            else:
                self.maha_diff_thres[2] = 2
#             self.maha_diff_thres = np.array([3,3,3])
        else:
            if self.CH_info.at[0,'std_T'] >self.sqrt_thres:   #4.23 添加 maha_dis 自适应性
                self.maha_diff_thres[0] = 1
            else:
                self.maha_diff_thres[0] = 2
            if self.CH_info.at[0,'std_H'] >self.sqrt_thres:
                self.maha_diff_thres[1] = 1
            else:
                self.maha_diff_thres[1] = 2
            if self.CH_info.at[0,'std_V'] >self.sqrt_thres:
                self.maha_diff_thres[2] = 1
            else:
                self.maha_diff_thres[2] = 2
        self.accepted_maha_error_thres = self.maha_diff_thres+2 #5_5针对突变性异常的鲁棒性。2__
        print(self.CH_info)
        print('maha_diff_thres',self.maha_diff_thres)
    def produce_SUM(self):
    # 统计queue 里面数据的 poslist、numlist
        #根据queue中记录的数据 和 recieve_NP_update 事件收到的全局信息计算pos num值
        #self.local_poslist            #本地pos 记录列表
        #self.local_numlist            #本地num 记录列表
#         print(self.queue.df)
#         print('ID= %d produce_SUM'%self.ID)
        #本地数据标准化 
#         print('self.CH_info.H',self.CH_info['H_MAX'])
        temp_df = self.queue.df.copy()     # 使用切片 为复制的数据值
        temp_df = Fun.localdata_norm(self.CH_info,temp_df)
        #print(self.queue.df)
#         print(self.queue.df)
        #统计分布
        #self.pos_local,self.num_local = Fun.local_data_distribution(self.CH_info,temp_df)
        ##记录训练集最后一个坐标信息。
#         self.pos_local,self.num_local,self.maha_last = Fun.local_data_distribution(self.CH_info,self.queue.df)
        #修改queue存储异常，当前queue中dataframe存储为原始数据，不会随标准化同时变化
        self.pos_local,self.num_local,self.maha_last = Fun.local_data_distribution(self.CH_info,temp_df)
        self.Send_pkt_num += 1        #收包数
#         print('%d produce_SUM  send_pkt+1 %d'%(self.ID,self.Send_pkt_num))
        
        
#         print(self.pos_local)
#         print(self.num_local)
        #return pos_local,num_local
        
    def compute_local_k(self):
        #节点发送K*数据
        #选择queue中 s个数据，做邻域数据计算
        #self.local_poslist            #本地pos 记录列表
        #self.local_numlist            #本地num 记录列表
#         print('ID= %d call_update_k'%self.ID)
        #self.K_local = Fun.calculate_K(self.queue.df,online_structure.samples_for_k,self.CH_info,self.pos_local,self.num_local)
        #小规模测试时使用
        temp_df = self.queue.df.copy()     # 使用切片 为复制的数据值
        temp_df = Fun.localdata_norm(self.CH_info,temp_df)  #本地dataframe为原始数据，需要标准化后使用4.9
        self.K_local = Fun.calculate_K(temp_df,self.sample_for_k,self.CH_info,self.pos_local,self.num_local)
    
    def receive_Pos_K_update(self,pos_l,num_l,k,echo):  #接收更新K
        self.echo = echo
#         print('node %d HyperGrid Edition No:%d'%(self.ID,self.echo))
        #收到簇头节点的K值数据。更新
        self.pos_all = pos_l.copy()           #簇内pos 记录列表
        self.num_all = num_l.copy() 
        self.K_ = k
        self.updateTimes +=1
#         print('updateTimes +1 num = %d'%self.updateTimes)
#         print('ID= %d recieve_Pos_K_update,K_ = %f ' %(self.ID,self.K_))
#         print(' ')
        #更新的最后一步
        self.data_exceed_3std_times = 0
        self.queue.clear_count()   #放到结构更新完成在clear 
        self.Recv_pkt_num += 1        #发包数
#         print('%d recieve_Pos_K_update recv +1 num=%d'%(self.ID,self.Recv_pkt_num))
        self.needUpdate = False  #4.15更新状态清空
        self.queue.clear_count()
        self.pos_queue.clear_count()

    ##########################更新网络结构模型结束#######################################   
    def sample_next_data(self):
        if self.datapoint<len(self.LoadDF):
            self.sample_a_data(self.datapoint,self.LoadDF.iloc[[self.datapoint]])
            self.datapoint += 1 
        else:
            print('Exceed the bound of dataDF') 
    
    def is_normalbyHGDB(self,rows,series):   #判断数据异常函数
        s1 = series.loc[rows,:]
        if len(s1) > 1:
            #返回pos_now 信息，用于后续的计算
            self.data_is_normal, cells_num,self.maha_now,self.pos_now, count = Fun.onePoint_detect(
                                                         self.pos_all,self.num_all,self.K_,
                                                         self.CH_info,s1['Temperature'],
                                                         s1['Humidity'],s1['Voltage'])  #返回异常检测的值
            self.detect_cells += cells_num        #检查一个数据，需要访问超立方体的个数
            self.vol_total += cells_num /self.CH_info.at[0,'Vol'] #记录每一次检测网格数量占全部网格数的累和 4.15 6.5
            self.detect_times +=1
            #add 191112  record latest 10 pos data.
            self.pos_queue.enqueue(self.maha_now)
            return self.data_is_normal
        else:
            print('is_normal:  series is empty')
            return False 
    
    
    
    
    ###########################################################################################
    ###主要函数
    def sample_a_data(self,rows,series): #3.14  series dataframe 
        #series = df.reset_index(drop=True)
        #         print(series)
        
        #计算曼哈顿距离
        if self.in_preTraining:             #print('ID= %d in_preTraining',self.ID)
            prob = np.random.rand()         #产生一个概率数,判断是否存储
            #if prob <= online_structure.prop_threshold:   #测试阶段概率为1
            if prob <= 1:   #测试阶段概率为1
                self.queue.enqueue(series)              #数据添加到数组中
                  ###########判断是否满内存 4.11修改  4.13注销
                if self.queue.is_trigger_update(): 
                    #更新的判断依据，队列覆盖满，
                                             ####problem 1
                    #self.call_update()   #生成本地 MN数据
                   # self.queue.clear_count()                          #清除内存计数，用于下次的覆盖  4.14
                    self.needUpdate = True  #5.31

        else:       #一般采样时期
            #判断异常
            #3problem2
            self.is_normalbyHGDB(rows,series) #调用is_normal函数生成一些本地变量
            self.data_is_mahaError,maha_differ = self.is_maha_error() # 使用pos_queue

            '''5.9 非manha error 添加至存储'''
            if self.data_is_mahaError: # 曼哈顿信息记录到节点内部存储中   跟新maha_last
                #                 print('data is mahaerror',self.data_is_mahaError)
                #                 print('last pos',self.maha_last  )
                #                 print('now pos',self.maha_now)
                self.detected_error.append(rows)  #5.4 修改
                self.maha_bytimes = self.maha_now
            else:
#                 self.queue.enqueue(series)                            #queue 变化
                self.maha_last = self.maha_now
                self.maha_bytimes = self.maha_now
                ''' 5.9'''
                prob = np.random.rand()                          #产生一个概率数，
                if prob <= self.prop_threshold:               # #判断是否存储 测试阶段概率为1
                    self.queue.enqueue(series)                            #queue 变化
                    if self.queue.is_trigger_update():                    #判断是否满内存  4.14
                        self.call_update()                                #生成本地 MN数据 4.14
                        self.queue.clear_count()                          #清除内存计数，用于下次的覆盖  4.14
                ''' 5.9'''
#             print('self.data_is_mahaError:',self.data_is_mahaError)
#             print('self.data_is_normal',self.data_is_normal)
            ############main judge##############
            #根据self.data_is_mahaError，self.data_is_normal关系，调节节点行为。
            if self.data_is_mahaError:
                #1示警#########
                if self.data_is_normal:    #曼哈顿距离异常，数据分布正常  ->  数据趋势变化点。（预示警）。开启曼哈顿距离记录观测
                    #判断是否已经进入示警阶段：
                    #print('fun 1: .warning...')
                    if self.is_in_warning:                #已经进入示警阶段
                        #print('already in warning')
                        pass
                    else:                                 #未进入示警阶段，则开启。 warning_start
                        #print('now in warning')
                        if not self.in_abnormal_event:    # 正常->示警
                            self.maha_dis = np.zeros((1,3))              
                            self.is_in_warning = True                      #进入异常阶段
                            self.pos_warning  = self.datapoint 
                            self.maha_dis = np.vstack((self.maha_dis,  maha_differ))  #记录maha_differ 信息
                            self.maha_dis = np.delete(self.maha_dis,0,0)   #删除第一行数据。
                            self.times_warning = 1                         #累计异常的次数   
                            self.detected_warning = np.append(self.detected_warning,self.datapoint) #添加当前警告数据点
                            self.life_warning = self.warning_check_times   #5 para
                            #print(self.maha_dis)
                        else: #已经进入异常状态，记录警告数据。 #
                            self.detected_warning = np.append(self.detected_warning,self.datapoint)
                
                else:           #曼哈顿距离异常，数据分布异常  ->  异常事件、单值异常。 措施：存储到异常数组中
                    #                     print('fun 2: .abnormal... ')
                    #                     print('dicover data is abnormal') #数据异常，maha距离异常
                 
                    #进入数据异常监测
                    if self.is_in_warning:                #之前已进入示警阶段。转变状态  示警->异常
                        #关闭示警阶段，进入异常阶段。
                        self.is_in_warning = False        #警告和异常状态只有一个为True
                        self.in_abnormal_event = True
                        #存储警报的数据
                        #将警告数据存储。self.records_warning = np.array([0,0])
                        temp_nparr = np.array([self.pos_warning,self.times_warning-1])  #将起始警告起始时刻，和警告次数存储到数组中
                        #删除最后一个警告值，该值将存在异常数组中
                        self.detected_warning = np.delete(self.detected_warning,-1)
                        self.records_warning =  np.vstack((self.records_warning,temp_nparr))  #第一行[0,0] 为初始数据 
                        #                         print('warning to abnormal ....saving data....')
                        #                         print('records_warning',self.records_warning)
                        
                        #     #记录异常信息     
                        #异常开始的工作
                        self.maha_dis = np.zeros((1,3))                             #将maha_dis 数组置空，重新记录
                        self.maha_dis = np.vstack((self.maha_dis,  maha_differ))      #记录maha_differ 信息
                        self.maha_dis = np.delete(self.maha_dis,0,0)                  #删除第一行数据。
                        #                         print('new maha_dis',self.maha_dis)
                        #######
                        self.pos_abnormal = np.append(self.pos_abnormal,self.datapoint)#记录当前数据时刻，point值
                        
                        #                         self.data_abnormal = np.array([0,0,0])   #记录异常的数据信息 
                        temp_s1 = series.loc[rows,:]
                        temp_nparr = np.array([temp_s1['Temperature'],temp_s1['Humidity'],temp_s1['Voltage']])
                        self.data_abnormal = np.vstack((self.data_abnormal,temp_nparr))
                        self.data_abnormal = np.delete(self.data_abnormal,0,0)                  #删除第一行数据。
                        #记录异常的值
                        self.times_abnormal = 1                         #异常验证的次数。 避免单值异常，反复系统反复查询验证。
                        self.life_abnormal = self.abnorma_check_times   #异常检测的次数
                        #                         print('pos_abnormal start')
                        #                                                print('data_abnormal',self.data_abnormal)
                        
                    else:    #进来的条件  #状态调整为异常。（正常->异常、异常->异常）  warning 和 abnormal 不可以同时为True
                        #判断进入异常的前一状态。 确定异常是否为第一次进入
                        if self.in_abnormal_event:    #判断是否为异常状态进入
                            #                             print('already in abnormal event')
                            #                             print('abnormal-> abnormal')
                            #将异常数据继续缓存到数组中
                            temp_s1 = series.loc[rows,:]
                            temp_nparr = np.array([temp_s1['Temperature'],temp_s1['Humidity'],temp_s1['Voltage']])
                            self.data_abnormal = np.vstack((self.data_abnormal,temp_nparr))
                            #                             print('data_abnormal',self.data_abnormal)
                            # 记录异常值 +1  4.28
                            
                        else:                            #前一状态为正常，目前是刚进入异常状态
                            #                             print('normal -> abnormal')
                            self.in_abnormal_event = True
                            self.maha_dis = np.zeros((1,3))                             #将maha_dis 数组置空，重新记录
                            self.maha_dis = np.vstack((self.maha_dis,maha_differ))      #记录maha_differ 信息
                            self.maha_dis = np.delete(self.maha_dis,0,0)                  #删除第一行数据。
                            #                                    print('new maha_dis',self.maha_dis)
                            #######
                            self.pos_abnormal = np.append(self.pos_abnormal,self.datapoint)#记录当前数据时刻，point值

                            #                         self.data_abnormal = np.array([0,0,0])   #记录异常的数据信息 
                            temp_s1 = series.loc[rows,:]
                            temp_nparr = np.array([temp_s1['Temperature'],temp_s1['Humidity'],temp_s1['Voltage']])
                            self.data_abnormal = np.vstack((self.data_abnormal,temp_nparr))
                            self.data_abnormal = np.delete(self.data_abnormal,0,0)                  #删除第一行数据。
                            #记录异常的值
                            self.times_abnormal = 1                         #异常验证的次数。 避免单值异常，反复系统反复查询验证。
                            self.life_abnormal = self.abnorma_check_times   #异常检测的次数
                            #                             print('data_abnormal',self.data_abnormal)
                            
                ##########
                
            else:
                #3数据正常#########
                if self.data_is_normal:    #曼哈顿距离正常，数据分布正常  ->  数据正常   措施缓存到全局数组中
                    #                     print('fun 3: data insert into the pos array')
                    prob = np.random.rand()                          #产生一个概率数，
                    if prob <= self.prop_threshold:               # #判断是否存储 测试阶段概率为1
                        #判断pos_now  是否已经存储在本地pos_all中，是则添加该点的数量
                        if self.pos_now in self.pos_all:   
                        #                         print('data exit in the pos_all')
                        #                         print('before num in array', self.num_all[self.pos_all.index(self.pos_now)])
                            self.num_all[self.pos_all.index(self.pos_now)] += 1   #对应pos位置的数量+1
                        #                         print('after add 1, num in array', self.num_all[self.pos_all.index(self.pos_now)])
                        #否则，添加pos信息，num 信息
                        else:  
                        #                         print('data exit not in the pos_all')
                            self.pos_all.append(self.pos_now)   #将数据点添加到NP文件中
                            self.num_all.append(1)              #数量添加1
                        #                         print('after add 1, num in array', self.num_all[self.pos_all.index(self.pos_now)])
                        ###########
                    #4可能边界值#########
                else:                      #曼哈顿距离正常，数据分布异常 ->  数据边界变化  措施缓存到本地数组中。
                    #先存储到pos_all中，
                    #                     print('fun 4: neighbor error')
                    #假设为本地正常数据
                    #判断pos_now  是否已经存储在本地pos_all中，是则添加该点的数量
                    prob = np.random.rand()                          #产生一个概率数，
                    if prob <= self.prop_threshold:               # #判断是否存储 测试阶段概率为1
                        if self.pos_now in self.pos_all:   
                            #                         print('data exit in the pos_all')
                            #                         print('before num in array', self.num_all[self.pos_all.index(self.pos_now)])
                            self.num_all[self.pos_all.index(self.pos_now)] += 1   #对应pos位置的数量+1
                            #                         print('after add 1, num in array', self.num_all[self.pos_all.index(self.pos_now)])
                            #否则，添加pos信息，num 信息
                        else:  
                            #                         print('data exit not in the pos_all')
                            self.pos_all.append(self.pos_now)   #将数据点添加到NP文件中
                            self.num_all.append(1)              #数量添加1
                            #                         print('after add 1, num in array', self.num_all[self.pos_all.index(self.pos_now)])
                    

#         ###########判断是否满内存 4.11修改  4.13注销
#         self.is_queue_full  =  self.queue.is_trigger_update()
#         if self.data_exceed_3std:
#             self.data_exceed_3std_times+=1
#         if self.is_queue_full: # or (self.data_is_normal and (self.data_exceed_3std_times > self.data_exceed_3std_times_thres)): 
#             #更新的判断依据，队列覆盖满，数据标准化超过三
#                                      ####problem 1
#             self.call_update()   #生成本地 MN数据
#             #展示内存队列
#             #self.queue.showQueue()
#             #清除内存计数，用于下次的覆盖
# #                      self.queue.clear_count()   放到结构更新完成在clear 
#         else:
#             self.eventVal = 0

            
    
 

   
    
    ##########################子节点异常事件判断#########################################
    
    # #曼哈顿距离计算 记录当前点的曼哈顿距离异常
    # self.maha_dis = np.array([0,0,0])
    # self.maha_last = np.array([])  # 存储上一个pos 信息
    # self.maha_now = np.array([])   # 存储目前的pos 信息
    # self.is_mahaError = False     # 记录此刻是否存在的maha异常
    
    def is_maha_error(self):        #new 4.3 如果未进入示警阶段(in_warning_proccess)，无需记录.反之记录maha差分值
        # pos_
        #判断pos_last,pos_now的信息
        #判断self.pos_last,self.pos_now 的关系
#         print('is_maha_error')
#         print('node ID = %d'%self.ID)
#         print('self.maha_now',self.maha_now)
#         print('self.maha_last',self.maha_last)
        maha_dis = 0
        is_mahaError = False
        temp_dis = np.array([0,0,0])
        for i in range(len(self.maha_last)):
            temp_dis[i] = self.maha_now[i]-self.maha_last[i]   
#             if np.abs(self.maha_now[i]-self.maha_last[i])<=2:  #4.15 放宽曼哈顿距离的阈值，减少正常数据的示警。可根据不同的属性，调整该值
            if np.abs(self.maha_now[i]-self.maha_last[i])<=self.maha_diff_thres[i]:    #4.23 增加自适应性 undone
                pass#
            else:
                is_mahaError = True
        
#         self.maha_bytimes       #5.5
        
        #在警报阶段
        if self.is_in_warning:        #进入警报阶段. 第一个数据在sample_a_data函数汇中添加并记录初始时刻的信息self.datapoint
            self.maha_dis = np.vstack((self.maha_dis,temp_dis))
#             print(self.maha_dis)
            if is_mahaError:                       #每次判断，统计数量  
                self.times_warning += 1           #预先+1.若发现状态转移， 则记录时减1 
                #添加当前的datapoint信息，可能存在与异常值的重合
                self.detected_warning = np.append(self.detected_warning,self.datapoint) #添加当前警告数据点
        elif self.in_abnormal_event:               #进入异常阶段
            self.maha_dis = np.vstack((self.maha_dis,temp_dis))
#             print(self.maha_dis)
            if is_mahaError:               #每次判断，统计数量
                 self.times_abnormal += 1  #异常事件优先级最高，不会被打断        
        return is_mahaError,temp_dis

    

    def node_internal_processing(self): #节点内部处理函数，警报还有异常。   正常，异常，警告三个状态只能一者为True
        self.check_condition()
        temp_dis = np.array([0,0,0])
#         print('node %d in internal_processing'%self.ID)
        #1、判断警告数组是否为空 ，是否为warning阶段  ##偏移
        if self.is_in_warning:     #判断是够正在异常中。
#             print('node_internal_processing warning event')
             #5.5 避免正常的突发性事件的发生，导致模型偏移
#              if self.life_warning >= self.warning_check_times-1: #说明第一次进入
            if self.life_warning >= 1: #每一次进入
                #计算每个值与前一个值的maha距离。
#                 self.is_normal_change = True
                for i in range(len(self.maha_bytimes)):
                    #undone
                    temp_dis[i] = self.maha_now[i]-self.maha_bytimes[i]   
                    if np.abs(temp_dis[i]) <= self.maha_diff_thres[i]:    #5_6 增加自适应性 undone
                        temp_dis[i] = 0
                        #pass#
                    elif np.abs(temp_dis[i]) <= self.accepted_maha_error_thres[i]:      #介于正常阈值与突变阈值之间
                        temp_dis[i] = 1
                        pass #保留变化值
                    else: #大于弹性阈值，则数据异常
                        self.is_normal_change  = False
                        

                if self.is_normal_change:  #增加模型对正常突变值的鲁棒性
                    #计算向量。
                    if self.life_abnormal == self.warning_check_times: # 第一次进入
                        self.normal_change_vector = temp_dis
                    else: #第二、三...次进入
                        #判断 趋势向量是否相同
                        if (temp_dis==0).all():  #正常的单值突变后，数据趋于平缓
                            self.belive_in_normal_change+=1
                        elif (self.normal_change_vector == temp_dis).all():  #数据仍然有变化趋势
                            self.belive_in_normal_change+=1 #置信值+1
                        else: 
                            self.normal_change_vector = temp_dis
                else: #不是突发的改变
                    #pass
                    self.belive_in_normal_change = 0 #    置信值置0
             #1.1 警告数组置信轮数-
            self.life_warning -= 1
            #1.2 是否存在置信轮数为0的值  query_warning()  并且self.is_in_warning = False。示警阶段结束
            if self.life_warning <= 0:
                #1.3 判断近期曼哈顿差分是否大于阈值，示警。 火灾
                #分析self.maha_dis 数据
#                 print('warning times',self.times_warning)
                is_need_query  = self.analyse_maha_dis() #maha 差分数组 分析函数
                self.is_in_warning = False
                if is_need_query:     #查询异常
#                     print('发起警告查询')
                    self.query_warning(self.pos_now)
                else: #不需要查询 5_6
                    self.maha_last = self.maha_now

        #2、判断是否为abnormal阶段
        elif self.in_abnormal_event:  
#             print('node_internal_processing abnormal event')

             #5.5 避免正常的突发性事件的发生，导致模型偏移
            if self.life_abnormal >= 1: #说明第一次进入
                #计算每个值与前一个值的maha距离。
#                 self.is_normal_change = True  #大于异常值后变，不做累计置信值
                for i in range(len(self.maha_bytimes)):
                    #undone
                    temp_dis[i] = self.maha_now[i]-self.maha_bytimes[i]   
    #             if np.abs(self.maha_now[i]-self.maha_last[i])<=2:  #4.15 放宽曼哈顿距离的阈值，减少正常数据的示警。可根据不同的属性，调整该值
                    if np.abs(temp_dis[i]) <= self.maha_diff_thres[i]:    #5_6 增加自适应性 undone
                        temp_dis[i] = 0
                        #pass#
                    elif np.abs(temp_dis[i]) <= self.accepted_maha_error_thres[i]:      #介于正常阈值与突变阈值之间
                        temp_dis[i] = 1
                        pass #保留变化值
                    else: #大于弹性阈值，则数据异常
                        self.is_normal_change  = False
                if self.is_normal_change:
                    #计算向量。
                    if self.life_abnormal == self.abnorma_check_times: # 第一次进入
                        self.normal_change_vector = temp_dis
                    else: #第二、三...次进入
                        #判断 趋势向量是否相同
                        if (temp_dis==0).all():  #正常的单值突变后，数据趋于平缓
                            self.belive_in_normal_change+=1
                        elif (self.normal_change_vector == temp_dis).all():  #数据仍然有变化趋势
                            self.belive_in_normal_change+=1 #置信值+1
                        else: 
                            self.normal_change_vector = temp_dis
                else: #不是突发的改变
                    #pass
                    self.belive_in_normal_change = 0 #    置信值置0
                    
            #2.1、异常数组置信轮数-1
            self.life_abnormal -= 1
            #2.2、是否存在置信轮数为0的值  query_abnormal()
            if self.life_abnormal <= 0:
#                 print('Error times',self.times_abnormal)
                is_need_query  = self.analyse_maha_dis() #maha 差分数组 分析函数
                self.in_abnormal_event = False 
                #2.3、判断异常值累计数量。
                if is_need_query:     #查询异常
#                     print('发起异常查询')
                    self.query_abnormal(self.pos_now) 
        #正常状态
        else:
#             print('No abnormal happens')
            pass
        
        #直接在循环中判断 undone maha_now 出现负数坐标值，则更新self.data_is_normal, cells_num,self.maha_now
        #3判断内存是否填充满 或者Normol profile属性的标准化后,绝对最大值大于3
        #3.1 若需要更新，则发起call_update。计算本地存储的数据信息
        #3.2 不需要，则pass
#         if (self.data_exceed_3std and self.is_normal) or self.queue.is_trigger_update():
#         if self.queue.is_trigger_update():  # sample中存在判断
#             print('calling update...........')
#             self.call_update()
        # 4.11
#         self.needUpdate = False 4.15
        need_update = False
        if self.data_is_normal:
            for i in range(len(self.maha_now)):
                if self.maha_now[i]<=0:   #小于负数 更新
                    need_update = True
        if self.needUpdate or need_update:
            self.needUpdate = True
        
        #事件判断

        self.record_storage_use()        #记录此刻的内存使用
        
    def analyse_maha_dis(self):       #maha_dis 分析函数
        is_need_query = False
        
        if self.is_in_warning:
#             print('in analysing process:analyse warning maha_dis  ')
#             print(self.maha_dis)
            if (self.belive_in_normal_change < self.belive_in_normal_change_thres  and   #5.6
                    self.times_warning > self.warning_thres):    #para            #警告检测过程中，异常出现的次数
                is_need_query = True
                
            #将警告数据存储。self.records_warning = np.array([0,0])
            temp_nparr = np.array([self.pos_warning,self.times_warning])  #将起始警告起始时刻，和警告次数存储到数组中
            self.records_warning =  np.vstack((self.records_warning,temp_nparr))  #第一行[0,0]未初始数据
#             print('records_warning', self.records_warning)
#             self.is_in_warning = False
            
        elif self.in_abnormal_event:
#             print('in analysing process:analyse abnormal maha_dis ')
#             print(self.maha_dis)
            if (self.belive_in_normal_change < self.belive_in_normal_change_thres  and 
                    self.times_abnormal > self.abnorma_thres):    #para            #异常检测过程中，异常出现的次数
                is_need_query = True
            temp_nparr = np.array([self.pos_abnormal[-1],self.times_abnormal])  #将起始警告起始时刻，和警告次数存储到数组中    
            self.records_abnormal = np.vstack((self.records_abnormal,temp_nparr)) #存储异常发生时刻，并且一顿事件内发生的次数
#             print('records_abnormal', self.records_abnormal)    
#             self.in_abnormal_event = False    
        return is_need_query
        # self.pos_warning 初始异常记录点
        #self.warning_records = np.array([0,0]) 
        
#         return is_need_query
    
   
    def query_warning(self,pos):      #发生异常,向簇内节点发起警告查询。   发包在send_query_data中累计
        self.eventVal = 0x4           #发送警告查询请求
        
#         self.Send_pkt_num += 1        #发包数
#         print('%d query_warning Send_pkt +1 num =%d '%(self.ID,self.Send_pkt_num))
#         print('query_warning')
    def query_abnormal(self,pos):      #发生异常,向簇内节点发起查询。 发包在send_query_data中累计
        #
        self.eventVal = 0x5           #发送异常查询请求
        #获取缓存的近期十个数据，作为LLSE集合。
#         self.Send_pkt_num += 1        #发包数
#         print('%d query_abnormal Send_pkt +1 num=%d '%(self.ID,self.Send_pkt_num))
#         print('query_abnormal')
        
#     def receive_warning_analyse(self,pos):    #收到来此其他节点的异常点判断。记录数据点与最近几个数据点的，样本差
#         self.eventVal = 0xc           #发送异常查询请求
#         self.Recv_pkt_num += 1        #收包数
#         print('%d receive_warning_analyse Recv_pkt +1 num =%d' %(self.ID,self.Recv_pkt_num))
# #         print('receive_warning_analyse')
#     def receive_abnormal_analyse(self,pos):      #发生异常,向簇内节点发起查询。
#         #处理data_abnormal 数据缓存
#         self.eventVal = 0xd           #发送异常查询请求
#         self.Recv_pkt_num += 1        #收包数
#         print('ID%d receive_abnormal_analyse Recv_pkt +1 num = %d'%(self.ID,self.Recv_pkt_num))
        
    def send_query_data(self):  #异常发生后，发送ID 、前一超网格模型的均值作为 LLSE简化计算。
        #undone
        
        temp_df = self.queue.df[len(self.queue.df)-10:len(self.queue.df)].copy()
        temp_df  = temp_df.reset_index(drop=True)
#         print(temp_df)
       
        self.Send_pkt_num += 1        #发包数
#         print('%d send_query_data Send_pkt +1 num=%d'%(self.ID,self.Send_pkt_num))
        return self.ID,temp_df
    
    def receive_query_data(self,id_,df):  #异常发生后，发送ID 、前一超网格模型的均值作为 LLSE简化计算。
#         meant_self = self.MN_info.at[0,'LLS_T']/self.MN_info.at[0,'LN_T']
#         meanh_self = self.MN_info.at[0,'LLS_H']/self.MN_info.at[0,'LN_H']
#         meanv_self = self.MN_info.at[0,'LLS_V']/self.MN_info.at[0,'LN_V']
#         self.train_mean
#         estimate_val = t_mean + ((t_mean**2+meant_self**2)/2-t_mean*meant_self)/self.CH_info.at[0,'']
#         return self.ID,self.MN_info.at[0,'LLS_T']/self.MN_info.at[0,'LN_T']
        #undone 
        #提取本地的最近10个数据
        
        df_local = self.queue.df[len(self.queue.df)-10:len(self.queue.df)].copy()
        df_local = df_local.reset_index(drop=True)
#         print('df_local',df_local)
        #求测试样本和本地样本的均值
        mean1 = df.mean()
        mean2 = df_local.mean()
        #求本地样本的方差
        var2 = df_local.var()
        #求协方差
        cov_t = df_local['Temperature'].cov(df['Temperature'])
        cov_h = df_local['Humidity'].cov(df['Humidity'])
        cov_v = df_local['Voltage'].cov(df['Voltage'])
        #求估计值,使用当前值
        
        if self.datapoint >= len(self.LoadDF):
            point = self.datapoint -1
        else:
            point = self.datapoint
        
        
        if var2['Temperature'] ==0:
            estimate_t = mean1['Temperature']
        else:
            
            estimate_t = mean1['Temperature'] + cov_t/var2['Temperature'] * (
                self.LoadDF.at[point,'Temperature'] - mean2['Temperature'] )

        if var2['Humidity'] ==0:
            estimate_h = mean1['Humidity']
        else:
            
            estimate_h = mean1['Humidity'] + cov_h/var2['Humidity'] * (
                self.LoadDF.at[point,'Humidity'] - mean2['Humidity'] )

#         print(mean1['Voltage'])
#         print(cov_v)
#         print(var2['Voltage'])
#         print( self.LoadDF.at[self.datapoint,'Voltage'])
#         print(mean2['Voltage'])
        if var2['Voltage'] ==0:  #电压值保持不变
            estimate_v = mean1['Voltage']  #使用原来的均值替代
        else:
            estimate_v = mean1['Voltage'] + cov_v/var2['Voltage'] * (
                self.LoadDF.at[point,'Voltage'] - mean2['Voltage'] )
            
#         print('estimate_t,estimate_h,estimate_v')
#         print(estimate_t,estimate_h,estimate_v)
        self.estimate_data = [estimate_t,estimate_h,estimate_v]
        self.estimate_id = id_
        self.Recv_pkt_num += 1        #收包数
#         print('node %d receive_query_data num = %d'%(self.ID,self.Recv_pkt_num))
    #发送估计值到异常节点 undone
    def send_estimate(self):
#         print('nodeid',self.ID)
#         print('send_estimate')
#         print(self.estimate_id,self.estimate_data)
        self.Send_pkt_num += 1        #收包数
#         print('%d send_estimate Send_pkt +1 num = %d'%(self.ID,self.Send_pkt_num))
        return self.estimate_id,self.estimate_data
    def receive_estimate(self,id_,estimate_data):
        #undone
        self.eventVal = 0x0
        #将估计数据标准化，并映射
#         print('ID %d receive_estimate'%self.ID)
        norm_t,norm_h,norm_v = Fun.onedata_norm(
            self.CH_info,estimate_data[0],estimate_data[1],estimate_data[2])
        pos_estimate,maha_estimate = Fun.get_mapPos(
            self.CH_info,norm_t,norm_h,norm_v)    #获取映射点信息
#         self.maha_last
#         print('maha_estimate',maha_estimate)
#         print('self.maha_last',self.maha_last)
        is_mahaError1 = False
#         temp_dis = np.array([0,0,0])
        for i in range(len(self.maha_last)):
#             temp_dis[i] = maha_estimate[i]-self.maha_last[i]
            if np.abs(maha_estimate[i]-self.maha_last[i])<=1:
                pass
            else:
                is_mahaError1 = True
                
        ###与最近一个值比较maha距离
        is_mahaError2 = False
#         temp_dis = np.array([0,0,0])
        for i in range(len(self.maha_now)):
#             temp_dis[i] = maha_estimate[i]-self.maha_now[i]
            if np.abs(maha_estimate[i]-self.maha_now[i])<=1:
                pass
            else:
                is_mahaError2 = True
        
#         if is_mahaError: #估计值和当前maha_last,距离很大。说明当前值，异常
#             self.maha_last = maha_estimate
#             print('change self.maha_last to ',self.maha_last)
#         else: # 出现异常。
#             pass
        if is_mahaError1:  # last 错误
            if is_mahaError2:# now 错误
                self.maha_last = maha_estimate
            else: #now 正确
                self.maha_last = self.maha_now
        else: #last 正确  都不变
            if is_mahaError2: #now 错误
                pass
            else:   #now 正确
                self.maha_last = self.maha_now
        #计算估计值与maha_last的曼哈顿距离。如果所有其他节点的估计值到其的 曼哈顿距离大于阈值，则maha_last 错误。修正maha_last
        #...
        self.Recv_pkt_num += 3        #收包数 其他节点 和sink 节点。
#         print('node %d receive_estimate +3 num = %d'%(self.ID,self.Recv_pkt_num))
    def send_update_request(self):
        self.Send_pkt_num += 1
        self.needUpdate = True #5.30
    def receive_call_update(self):
        self.Recv_pkt_num +=1
#         if self.ID == self.testID:
#             print('receive_call_update +1 = ',self.Recv_pkt_num)
        print(' ID %d: receive_call_update recv_pkt_num= %d'%(self.ID,self.Recv_pkt_num ))
        self.call_update()
 

if __name__ == "__main__":
    random.seed = 0