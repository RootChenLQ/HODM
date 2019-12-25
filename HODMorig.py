#coding:utf-8
import Fun
import Node
import pandas as pd
import numpy as np
import random
random.seed = 0
import time
import yaml
#记录多次实验的性能
# save_type = ['ID','ACC','FPR','TPR','P']
# save_type = ['ID','ACC','FPR','TPR','P','Recv_pkt','Send_pkt','Update_times','Rate_of_DetectedArea','Max_Storage']
save_type = ['ID','TN','FN','FP','TP','ACC','FPR','TPR','P','Recv_pkt','Send_pkt','Update_times','Rate_of_DetectedArea','Max_Storage']

df = pd.DataFrame(columns = save_type) 
if __name__ == '__main__':## 其他
    times = 0
    #load parameters
    _PARAMS_PATH = "params.yaml"
    with open(_PARAMS_PATH, "r") as f:
        modelParams = yaml.safe_load(f)
    '''
  CHParams:
    member_size: 3
  SNParams:
    buffer_size: 300
    store_pro: 0.8
    sample_size: 50
    sqrt_thres: 0.5
  CommonParams:
    datasize: 10000'''

    while times < 1:
        #导入数据
        # Filled_DF_Type = ['Temperature','Humidity','Light','Voltage'] #四维
        Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记

#intel 1
        
        datefile1 = 'datasets/node43op.csv'
        datefile2 = 'datasets/node44op.csv'
        datefile3 = 'datasets/node45op.csv'

# intel 2
#         datefile1 = 'Intel_node18_10000.txt'
#         datefile2 = 'Intel_node21_10000.txt'
#         datefile3 = 'Intel_node23_10000.txt'
# intel read
        try:
            ##读取txt 文件
#             print('data read success')
            data1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
        except IOError:
            print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
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


        
        datalen = modelParams['CommonParams']['datasize']
        
        datalen = 4000

        data1 = data1[0:datalen].copy()
#         print(len(data1))
        data2 = data2[0:datalen].copy()
#         print(len(data2))
        data3 = data3[0:datalen].copy()
#         print(len(data3))
#         print(data3)
        #不插入异常
        outlier_pos1 = []
        #data1,outlier_pos1,noise_data = Fun.insert_noise_error(data1,data2,data3,300,5,'Intel')
        outlier_pos2 = []
        outlier_pos3 = []
        #创建节点
#         bufferSize = 300
        bufferSize = modelParams['SNParams']['buffer_size']  #4.23
        bufferSize = 2000
        attributes = modelParams['CommonParams']['attributes']
        pos_buffer_size = modelParams['CommonParams']['pos_buffer_size']
        q = modelParams['CHParams']['member_size'] #三个节点
        sink = Node.CH_Node(0,q,attributes,modelParams['SNParams']['buffer_size'])
        
        sqrt_thres = modelParams['SNParams']['sqrt_thres']
        store_pro = modelParams['SNParams']['store_pro']
        sampleforK = (int)(modelParams['SNParams']['sample_size_rate']*bufferSize)
        #加载数据
        mn1 = Node.MN_Node(8,bufferSize,sqrt_thres,store_pro,sampleforK,pos_buffer_size,attributes) #sqrt_thres = 0.5,save_prop=0.8,sampleforK = 50
        mn2 = Node.MN_Node(9,bufferSize,sqrt_thres,store_pro,sampleforK,pos_buffer_size,attributes)
        mn3 = Node.MN_Node(10,bufferSize,sqrt_thres,store_pro,sampleforK,pos_buffer_size,attributes)
        #加载数据
        data_len = len(data1)
        mn1.load_dataset(data1,'Intel')
        mn2.load_dataset(data2,'Intel')
        mn3.load_dataset(data3,'Intel')
        #模型预训练,直接采样至节点buffer满，计算本地self.MN_info
        mn1.preTrain(bufferSize)
        mn2.preTrain(bufferSize)
        mn3.preTrain(bufferSize)
        #发起更新 sample 采样函数中，队列满则发起call_update
        mn1.send_update_request() #send 请求搞笑的数据包
        mn2.send_update_request()
        mn3.send_update_request()
        #sink 节点接收到   簇内节点的本地信息包。汇总，统计出最大值、最小值、平均值，及一些结构参数
        sink.sink_mn_call_update()
        sink.sink_mn_call_update()
        sink.sink_mn_call_update()
        
        mn1.receive_call_update() #产生 MN_info
        mn2.receive_call_update()
        mn3.receive_call_update()
        
        
        sink.receive_call_update_data(mn1.MN_info)
        sink.receive_call_update_data(mn2.MN_info)
        sink.receive_call_update_data(mn3.MN_info)
        #簇内节点接收到簇头的全局信息包
        mn1.receive_NP_update(sink.CH_info)
        mn2.receive_NP_update(sink.CH_info)
        mn3.receive_NP_update(sink.CH_info)
        #簇内节点内部计算 pos num 的数量
        mn1.produce_SUM()
        mn2.produce_SUM()
        mn3.produce_SUM()
        #本地计算 K*值，利用本地计算的pos_local ,num_local 
        mn1.compute_local_k()                
        mn2.compute_local_k()
        mn3.compute_local_k()
        #簇头节点接收到簇内节点发送来的pos num k*信息并整合
        sink.receive_pos_k(mn1.pos_local,mn1.num_local,mn1.K_local)
        sink.receive_pos_k(mn2.pos_local,mn2.num_local,mn2.K_local)
        sink.receive_pos_k(mn3.pos_local,mn3.num_local,mn3.K_local)
        #簇头发送pos_all ,num_all ,K 信息给节点
        #sink.send_Pos_k_info()  # 接收次数>=3 触发sink.send_Pos_k_info()函数
        #簇内节点接收到pos_all ,num_all ,K 信息
        mn1.receive_Pos_K_update(sink.pos_all,sink.num_all,sink.K_,sink.echo)
        mn2.receive_Pos_K_update(sink.pos_all,sink.num_all,sink.K_,sink.echo)
        mn3.receive_Pos_K_update(sink.pos_all,sink.num_all,sink.K_,sink.echo)

        #更新后，将队列的计数清空 receive_Pos_K_update 已经添加clear
#         mn1.queue.clear_count()
#         mn2.queue.clear_count()
#         mn3.queue.clear_count()
        #将事件置0，进入检测阶段
        mn1.eventVal = 0x0
        mn2.eventVal = 0x0
        mn3.eventVal = 0x0
        sink.eventVal = 0x0
#         mn1.maha_laat = [1,2,3]
    #     wsn_event = 0
        
        outlier_pos_2 = []
        outlier_pos_3 = []
        
        ###开始检测
        step = bufferSize
#         step = 100 #4.23
        runtime_bySamples = np.array([])
        data_size = datalen - bufferSize   #测试集的数量 4.23
#         data_size = datalen - 100
        while step < datalen:
#         while step <310:
            #每个采集数据
            print(step)
            startTime = time.time()  #记录开始时间

            if step%1000 ==0:
                print('sample data no:',step)
            mn1.sample_next_data()   # 包含异常检测，曼哈顿距离计算，
            mn2.sample_next_data()
            mn3.sample_next_data()
            #获取节点的事件变量，协调网络事件


            #sink 根据事件变量，调整。更新、异常事件处理
            mn1.node_internal_processing()
            mn2.node_internal_processing()
            mn3.node_internal_processing()
            
            
#             if mn1.eventVal == 0x4: #警告次数大于阈值
                #undone1
            
            
            ##警告纠正 异常纠正
            if mn1.eventVal == 0x5  or mn1.eventVal == 0x4:  #发生异常
                #发起询问
#                 print(step)
                ab_id,data_df = mn1.send_query_data()
                #其他节点接收查询，并估计
                mn2.receive_query_data(ab_id,data_df)
                mn3.receive_query_data(ab_id,data_df)
                sink.receive_query_data(ab_id,data_df) #为空
                #其他节点发送估计结果
                est_id1,est_data1 = mn2.send_estimate()
                est_id2,est_data2 = mn3.send_estimate()
                sink.send_estimate() #为空
                #estdata求平均值
                est_data = []
                for i in range(len(est_data1)):
                    est_data.append((est_data1[i]+est_data2[i])/2)
                #undone
                #处理估计结果
                mn1.receive_estimate(est_id1,est_data)
#                 mn1.receive_estimate(est_id1,est_data1)
#                 mn1.receive_estimate(est_id2,est_data2)
            if mn2.eventVal == 0x5 or mn2.eventVal == 0x4:
#                 print(step)
#                 print('mn2 query abnormals')
                ab_id,data_df = mn2.send_query_data()
                #其他节点接收查询，并估计
                mn1.receive_query_data(ab_id,data_df)
                mn3.receive_query_data(ab_id,data_df)
                sink.receive_query_data(ab_id,data_df) #为空
                #其他节点发送估计结果
                est_id1,est_data1 = mn1.send_estimate()
                est_id2,est_data2 = mn3.send_estimate()
                sink.send_estimate() #为空
                #undone
                #处理估计结果
                est_data = []
                for i in range(len(est_data1)):
                    est_data.append((est_data1[i]+est_data2[i])/2)
                #undone
                #处理估计结果
                mn2.receive_estimate(est_id1,est_data)
#                 mn2.receive_estimate(est_id1,est_data1)
#                 mn2.receive_estimate(est_id2,est_data2)
            if mn3.eventVal == 0x5 or mn3.eventVal == 0x4:
#                 print(step)
                ab_id,data_df = mn3.send_query_data()
                #其他节点接收查询，并估计
                mn1.receive_query_data(ab_id,data_df)
                mn2.receive_query_data(ab_id,data_df)
                sink.receive_query_data(ab_id,data_df) #为空
                #其他节点发送估计结果
                est_id1,est_data1 = mn1.send_estimate()
                est_id2,est_data2 = mn2.send_estimate()
                sink.send_estimate() #为空
                #undone
                #处理估计结果
                est_data = []
                for i in range(len(est_data1)):
                    est_data.append((est_data1[i]+est_data2[i])/2)
                #undone
                #处理估计结果
                mn3.receive_estimate(est_id1,est_data)
#                 mn3.receive_estimate(est_id1,est_data1)
#                 mn3.receive_estimate(est_id2,est_data2)
                
            #判断更新
#             if mn1.queue.is_trigger_update() or mn2.queue.is_trigger_update() or mn3.queue.is_trigger_update() :
            if mn1.needUpdate or mn2.needUpdate or mn3.needUpdate:
                print('Update....+1')
#                 if not mn1.needUpdate:
#                     mn1.call_update()
#                 if not mn2.needUpdate:
#                     mn2.call_update()
#                 if not mn3.needUpdate:
#                     mn3.call_update()
                if mn1.needUpdate:    #请求更新
                    mn1.send_update_request()
                    sink.receive_call_update()
                if mn2.needUpdate:
                    mn2.send_update_request()
                    sink.receive_call_update()
                if mn3.needUpdate:
                    mn3.send_update_request()
                    sink.receive_call_update()
                #簇内节点接收簇头的发送的同步更新数据包
                sink.sink_mn_call_update()  #sink  节点接收到更新请求
                mn1.receive_call_update()
                mn2.receive_call_update()
                mn3.receive_call_update()
                #sink 节点接收到   簇内节点的本地信息包。汇总，统计出最大值、最小值、平均值，及一些结构参数
                sink.receive_call_update_data(mn1.MN_info)
                sink.receive_call_update_data(mn2.MN_info)                
                sink.receive_call_update_data(mn3.MN_info)
                #簇内节点接收到簇头的全局信息包
                mn1.receive_NP_update(sink.CH_info)
                mn2.receive_NP_update(sink.CH_info)
                mn3.receive_NP_update(sink.CH_info)
                #簇内节点内部计算 pos num 的数量
                mn1.produce_SUM()
                mn2.produce_SUM()
                mn3.produce_SUM()
                #本地计算 K*值，利用本地计算的pos_local ,num_local 
                mn1.compute_local_k()                
                mn2.compute_local_k()
                mn3.compute_local_k()
                #簇头节点接收到簇内节点发送来的pos num k*信息并整合
                sink.receive_pos_k(mn1.pos_local,mn1.num_local,mn1.K_local)
                sink.receive_pos_k(mn2.pos_local,mn2.num_local,mn2.K_local)
                sink.receive_pos_k(mn3.pos_local,mn3.num_local,mn3.K_local)
                #簇头发送pos_all ,num_all ,K 信息给节点
                #sink.send_Pos_k_info()  # 接收次数>=3 触发sink.send_Pos_k_info()函数
                #簇内节点接收到pos_all ,num_all ,K 信息
                mn1.receive_Pos_K_update(sink.pos_all,sink.num_all,sink.K_,sink.echo)
                mn2.receive_Pos_K_update(sink.pos_all,sink.num_all,sink.K_,sink.echo)
                mn3.receive_Pos_K_update(sink.pos_all,sink.num_all,sink.K_,sink.echo)
                
                ##接收更新，清除队列计数
                mn1.queue.clear_count()
                mn2.queue.clear_count()
                mn3.queue.clear_count()
                
                
            endTime = time.time()
            runtime_bySamples = np.append(runtime_bySamples,endTime-startTime) 
            step+=1

        #保存结果 
        #8号节点


        #整合警告和异常的数据集
        detect_all1 = np.hstack((mn1.detected_error,mn1.detected_warning))
        detect_all1 = np.unique(detect_all1) # 去除异常重复记录值
        #添加发包量、收包量、示警数、异常检测率。
        print('node8')
        tn,fn,fp,tp,acc,fpr,tpr,p = Fun.compute_same_num(outlier_pos1,detect_all1,data_size)#1存在异常 
        #acc,fpr,tpr,p = online_func_paper.compute_same_num_no_error(detect_all,data_size) #2无异常
        s1 = pd.Series([mn1.ID, tn,fn,fp,tp,acc,fpr,tpr,p,mn1.Recv_pkt_num,mn1.Send_pkt_num,mn1.updateTimes,
                        #mn1.vol_total/mn1.detect_times,mn1.Max_Storage],
                        mn1.detect_cells/(mn1.detect_times*27),mn1.Max_Storage],  #8.27
                       index = save_type)
        df = df.append(s1,ignore_index = True,sort=False)
        
#         if acc < 0.9:

#             file_name = str(times)+'.csv'
# #             file = open(file_name,'w')          
#             np.savetxt('outlierpos'+file_name,outlier_pos1)
    
#             np.savetxt('outlierdata'+file_name,noise_data)
            
#             np.savetxt('insertoutlier data'+file_name,data1)
        
#             np.savetxt('detected_error'+file_name,mn1.detected_error)
        
#             np.savetxt('detected_warning'+file_name,mn1.detected_warning)
            
#             np.savetxt('records_warning'+file_name,mn1.records_warning)
            
#             np.savetxt('records_abnormal'+file_name,mn1.records_abnormal)
#             file.writelines('outlier_pos\n')
#             for i in range(len(outlier_pos1)):
#                 file.write(str(i)+': '+str(outlier_pos1[i])+' ')
#             file.writelines('\n')
#             file.writelines('\n')
            
# #             noise_data 4.30 undone
#             for i in range(len(noise_data)):
#                 file.write(str(i)+': '+str(noise_data[i])+' ')
#             file.writelines('\n')
#             file.writelines('\n')
            
# #             print(outlier_pos)
# #             print('mn1 queue size',mn1.queue.count)
#             file.writelines('detected_error')
#             for i in range(len(mn1.detected_error)):
#                 file.write(str(i)+': '+str(mn1.detected_error[i])+' ')
#             file.writelines('len of detected_error')
#             file.writelines(str(len(mn1.detected_error)))
#             file.writelines('\n')
#             file.writelines('\n')   
                
                
# #             print('detected_error',mn1.detected_error)
#             file.writelines('detected_warning')
#             for i in range(len(mn1.detected_warning)):
#                 file.write(str(i)+': '+str(mn1.detected_warning[i])+' ')
#             file.writelines('\n')  
#             file.writelines('len of detected_warning\n')
#             file.writelines(str(len(mn1.detected_warning)))
#             file.writelines('\n')
#             file.writelines('\n')  
    
    
#             file.writelines('records_warning\n')
#             for i in range(len(mn1.records_warning)):
#                 file.write(str(mn1.records_warning[i])+' ')
#             file.writelines('\n')  
#             file.writelines('len of records_warning\n')
#             file.writelines(str(len(mn1.records_warning)))
#             file.writelines('\n')
#             file.writelines('\n')     
            
#             file.writelines('records_abnormal\n')
#             for i in range(len(mn1.records_abnormal)):
#                 file.write(str(mn1.records_abnormal[i])+' ')
#             file.writelines('\n')  
#             file.writelines('len of records_abnormal\n')
#             file.writelines(str(len(mn1.records_abnormal)))
#             file.writelines('\n')
#             file.writelines('\n')  
#             #
            
#             file.close()

        
        #9号节点
#         print('mn1 queue size',mn2.queue.count)
#         print('detected_error',mn2.detected_error)
#         print('detected_warning',mn2.detected_warning)
#         print('len  detected_error',len(mn2.detected_error))
#         print('len  detected_warning',len(mn2.detected_warning))
#         print('detected_warning',mn2.records_warning)
#         print(len(mn2.records_warning))
#         print('detected_error',mn2.records_abnormal)
#         print(len(mn2.records_abnormal))

        #整合警告和异常的数据集
        print('node9')
        detect_all2 = np.hstack((mn2.detected_error,mn2.detected_warning))
        detect_all2 = np.unique(detect_all2) # 去除异常重复记录值
#         print(len(detect_all))
        #比对数据检测正常情况
  
        ##
#         print('train times %d abnormal'%times)
        tn,fn,fp,tp,acc,fpr,tpr,p = Fun.compute_same_num(outlier_pos_2,detect_all2,data_size)#1存在异常
#         acc,fpr,tpr,p = online_func_paper.compute_same_num_no_error(detect_all,data_size) #2无异常
        s1 = pd.Series([mn2.ID, tn,fn,fp,tp,acc,fpr,tpr,p,mn2.Recv_pkt_num,mn2.Send_pkt_num,mn2.updateTimes,
                        #mn2.vol_total/mn2.detect_times,mn2.Max_Storage],
                        mn2.detect_cells/(mn2.detect_times*27),mn2.Max_Storage], #8.27
                       index= save_type)
        df = df.append(s1,ignore_index = True,sort=False)
        #
        #10号节点
#         print('mn3 queue size',mn3.queue.count)
#         print('detected_error',mn3.detected_error)
#         print('detected_warning',mn3.detected_warning)
#         print('len  detected_error',len(mn3.detected_error))
#         print('len  detected_warning',len(mn3.detected_warning))
#         print('detected_warning',mn3.records_warning)
#         print(len(mn3.records_warning))
#         print('detected_error',mn3.records_abnormal)
#         print(len(mn3.records_abnormal))

        #整合警告和异常的数据集
        print('node10')
        detect_all3 = np.hstack((mn3.detected_error,mn3.detected_warning))
        detect_all3 = np.unique(detect_all3) # 去除异常重复记录值
#         print(len(detect_all))
        #比对数据检测正常情况
        # data_size = 5000
        # outlier_pos = []
        tn,fn,fp,tp, acc,fpr,tpr,p = Fun.compute_same_num(outlier_pos_3,detect_all3,data_size)  #1存在异常
#         acc,fpr,tpr,p = online_func_paper.compute_same_num_no_error(detect_all,data_size) #2无异常计算
        s1 = pd.Series([mn3.ID, tn,fn,fp,tp,acc,fpr,tpr,p,mn3.Recv_pkt_num,mn3.Send_pkt_num,mn3.updateTimes,
                        #mn3.vol_total/mn3.detect_times,mn3.Max_Storage],  #8.27
                        mn3.detect_cells/(mn3.detect_times*27),mn3.Max_Storage],
                       index= Output_DF_Type)
        df = df.append(s1,ignore_index = True,sort=False)
        times+=1