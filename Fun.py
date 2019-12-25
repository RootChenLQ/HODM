#coding:utf-8
import pandas as pd
import random
import numpy as np
import math
import Structure
#function8 Intel 数据集  节点本地数据集统计属性到list中 # new
#参数
#df：本次时间段内节点采集的数据  dataframe  ##intel_lab_type = ['DateTime','Epoch','ID','Temperature','Humidity','Voltage']
#返回值：簇内节点本地数据信息dataframe
##series = pd.Series([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0],index = MN_configure_type)
##MN_selfconfigure_DF.append(series,ignore_index=True )
#处理第一个节点的数据。温度，湿度，光照
#MN_configure_type = ['LLS_T','LLS_H','LLS_V','LLSS_T','LLSS_H','LLSS_V',
####'TemperatureMax','TemperatureMin','HumidityMax','HumidityMin','VoltageMax','VoltageMin','LN_T'','LN_H','LN_V']
def get_localMN_info(df):
    
    templist = []  #用于存储series的零时变量
    ##计算累加和
    templist.append(df['Temperature'].sum())#计算列的总和
    templist.append(df['Humidity'].sum())   #计算列的总和
    templist.append(df['Voltage'].sum())   #计算列的总和
    ##计算平方和
    templist.append(df['Temperature'].apply(lambda x: x*x).sum())   #计算列的总和
    templist.append(df['Humidity'].apply(lambda x: x*x).sum())   #计算列的总和
    templist.append(df['Voltage'].apply(lambda x: x*x).sum())   #计算列的总和
    #温度最大值、最小值
    templist.append(df['Temperature'].max())
    templist.append(df['Temperature'].min())
    #湿度最大值，最小值
    templist.append(df['Humidity'].max())
    templist.append(df['Humidity'].min())
    #电压最大值，最小值
    templist.append(df['Voltage'].max())
    templist.append(df['Voltage'].min())
    
    #记录各属性的数据量，非nan值
    templist.append(df['Temperature'].count())
    templist.append(df['Humidity'].count())
    templist.append(df['Voltage'].count())
    # 将数据
    temp_MN_info = Structure.MN_selfconfigure_DF.copy()#记录簇内节点数据信息
    temp_MN_info = temp_MN_info.append(pd.Series(templist,index = Structure.MN_configure_type),
                                       ignore_index=True,sort=False)
    return temp_MN_info

###function10Intel 数据集  整合簇内各节点属性，形成簇内整理配置参数###   new
#参数列表
#df :  整合好的dataframe文件，MN_configure_df. 多个节点的MH_info 汇总的函数
# MN_configure_type = ['LLS_T','LLS_H','LLS_V',
#                      'LLSS_T','LLSS_H','LLSS_V',
#                      'TemperatureMax','TemperatureMin',
#                      'HumidityMax','HumidityMin',
#                      'VoltageMax','VoltageMin',
#                      'LN_T','LN_H','LN_V']
 #定义簇头节点中，数据包发送给簇内节点的结构
# CH_configure_type = ['NT_g','NH_g','NV_g',
#                      'mean_T','mean_H','mean_V',
#                      'std_T','std_H','std_V',
#                      'TemperatureMaxNorm_g','TemperatureMinNorm_g',
#                      'HumidityMaxNorm_g','HumidityMinNorm_g',
#                     'VoltageMaxNorm_g','VoltageMinNorm_g',
#                     #'N',
#                      'H_MIN','H_MAX','B','C','D']  #HyperGrid网络信息
#返回NP_info
def get_Normal_Profile(df,H):################
    tempDF = Structure.CH_configure_DF.copy()  #簇头节点发送给子节点数据的数据包结构
    templist = []     #用于暂时存储簇头数据包的数据
    #1、计算各属性总数 NT_g、NH_g、NL_g、LN_V
    
    templist.append(df['LN_T'].sum()) #0 数量
    templist.append(df['LN_H'].sum()) #1
    templist.append(df['LN_V'].sum()) #2
    ##
    #2、各属性平均值
    templist.append((df['LLS_T'].sum())/templist[0]) #3  平均数
    templist.append((df['LLS_H'].sum())/templist[1]) #4
    templist.append((df['LLS_V'].sum())/templist[2]) #5
    #3、各属性方差
    temp_var_T = 0
    temp_var_H = 0
    temp_var_V = 0
    for i in range(len(df)):  #方差可以近似等于平方的期望-期望的平方
        temp_var_T += df.at[i,'LLSS_T']
        temp_var_H += df.at[i,'LLSS_H']
        temp_var_V += df.at[i,'LLSS_V']
    temp_var_T /= templist[0]
    temp_var_H /= templist[1]
    temp_var_V /= templist[2]
    temp_var_T -= templist[3] * templist[3] 
    temp_var_H -= templist[4] * templist[4]
    temp_var_V -= templist[5] * templist[5]

    templist.append(np.sqrt(temp_var_T))     #6 标准差
    templist.append(np.sqrt(temp_var_H))     #7
    templist.append(np.sqrt(temp_var_V))     #8

    #4、各属性 簇内最大值，最小值,先做均值为0，方差为1的预处理
    '''4.15改
    templist.append((df['TemperatureMax'].max()-templist[3])/templist[6])  #9 normalized val 
    templist.append((df['TemperatureMin'].min()-templist[3])/templist[6])  #10
    templist.append((df['HumidityMax'].max()-templist[4])/templist[7])     #11
    templist.append((df['HumidityMin'].min()-templist[4])/templist[7])     #12
    templist.append((df['VoltageMax'].max()-templist[5])/templist[8])      #13
    templist.append((df['VoltageMin'].min()-templist[5])/templist[8])      #14
    '''
    temp_max_norm = (df['TemperatureMax'].max()-templist[3])/templist[6]
    temp_min_norm = (df['TemperatureMin'].min()-templist[3])/templist[6]
    humi_max_norm = (df['HumidityMax'].max()-templist[4])/templist[7]
    humi_min_norm = (df['HumidityMin'].min()-templist[4])/templist[7]
    volt_max_norm = (df['VoltageMax'].max()-templist[5])/templist[8]
    volt_min_norm = (df['VoltageMin'].min()-templist[5])/templist[8]
#     templist.append = undone
    
#     side_t = math.ceil((ch_info.at[0,'TemperatureMaxNorm_g'] - ch_info.at[0,'TemperatureMinNorm_g'] )/ch_info.at[0,'H_MAX'])
#     side_h =  math.ceil((ch_info.at[0,'HumidityMaxNorm_g'] - ch_info.at[0,'HumidityMinNorm_g'] )/ch_info.at[0,'H_MAX'])
#     # 3月10号修改
#     #side_l =  math.ceil((gf.at[0,'LightMaxNorm_g'] - gf.at[0,'LightMinNorm_g'] )/hg.at[0,'H_MAX']) #
#     side_v =  math.ceil((ch_info.at[0,'VoltageMaxNorm_g'] - ch_info.at[0,'VoltageMinNorm_g'] )/ch_info.at[0,'H_MAX'])
    
    #templist.append(n*online_structure.dimensions)                       
    #H_l,H_h = get_HG_H(Structure.dimensions,Structure.point_size)
    H_h = H
#     templist.append(H_l)  #15 hyper grid h值
    templist.append(H_h)  #9
    side_t = math.ceil((temp_max_norm-temp_min_norm)/H_h)
    side_h = math.ceil((humi_max_norm-humi_min_norm)/H_h)
    side_v = math.ceil((volt_max_norm-volt_min_norm)/H_h)
    vol = side_t*side_h * side_v
    #print('side t= %d h= %d v= %d' %(side_t,side_h,side_v))
    #print('vol',vol)
    
    templist.append(vol)  #10
    #B,C = hypergrid_structure_B_C(gf,H_h)
    #求解 B、C 
    max_ = 0
    min_ = 0 
    #获取最大值 比较最大最小值
    if temp_max_norm > humi_max_norm:
        max_ = temp_max_norm
    else:
        max_ = humi_max_norm
    if max_ < volt_max_norm:
        max_ =  volt_max_norm
    #获取最小值
    if temp_min_norm < humi_min_norm:
        min_ = temp_min_norm
    else:
        min_ = humi_min_norm
    if min_ > volt_min_norm:
        min_  = volt_min_norm
    #c = 最小值向上取整
    C = math.ceil(np.abs(min_)+5)  #由于检测异常时，需要搜索领域， 增加鲁棒性 
    B = 0
    val = (max_+C)/H_h  
    while(2**B<val):
        B+=1
#     if b < 6:
#         b = 6
    templist.append(B)      #11
    templist.append(C)      #12
#     D = 2*templist[16]*np.sqrt(online_structure.dimensions)
#     #D = hypergrid_structure_D(online_structure.dimensions,H_h)    #19
#     templist.append(D)       #19
  
    #5 将列表添加到dataframe中
    tempDF = tempDF.append(pd.Series(templist,index = Structure.CH_configure_type),
                           ignore_index=True,sort=False)
    #print(tempDF)
    return tempDF

###function11 Intel 数据集  本地数据集规范函数###  即子节点内部数据处理函数#
#参数 
#gf :golbal file簇内属性数据。由CH_compute_globalfile产生
#CH_configure_type = ['NT_g','NH_g','NL_g','mean_T','mean_H','mean_L','std_T','std_H','std_L','TemperatureMaxNorm_g','TemperatureMinNorm_g',
#                     'HumidityMaxNorm_g','HumidityMinNorm_g',
#                     'LightMaxNorm_g','LightMinNorm_g']
#df: 本地数据 
#intel_lab_type = ['Date','Time','Epoch','ID','Temperature','Humidity','Light','Voltage']
#返回: 数据集服从 0均值，单位方差分布
def localdata_norm(gf,df): #new   传入的数据也会改变
    #本地数据做正态化 
    #温度、 #湿度       #光照  规范0均值，单位方差分布
    temp_df = df
#     df['Temperature'] = df['Temperature'].apply(lambda x:(x-gf['mean_T'])/gf['std_T'])   #方差公式存在问题
#     df['Humidity'] = df['Humidity'].apply(lambda x:(x-gf['mean_H'])/gf['std_H'])   #方差公式存在问题
#     df['Voltage'] = df['Voltage'].apply(lambda x:(x-gf['mean_V'])/gf['std_V'])   #方差公式存在问题
#     return df 
    temp_df['Temperature'] = temp_df['Temperature'].apply(lambda x:(x-gf['mean_T'])/gf['std_T'])   #方差公式存在问题
    temp_df['Humidity'] = temp_df['Humidity'].apply(lambda x:(x-gf['mean_H'])/gf['std_H'])   #方差公式存在问题
    temp_df['Voltage'] = temp_df['Voltage'].apply(lambda x:(x-gf['mean_V'])/gf['std_V'])   #方差公式存在问题

    return temp_df


def onedata_norm(np_,temp,humi,volt):    ##new  标准化一条数据
    #本地数据做正态化
    #温度、 #湿度 #电压  规范0均值，单位方差分布
    t = (temp - np_['mean_T'])/np_['std_T']
    h = (humi - np_['mean_H'])/np_['std_H']
    v = (volt - np_['mean_V'])/np_['std_V']
    return t[0],h[0],v[0] 

# 测试1 ：存在负数值   --2月26
#
###function12 计算样本检测区域d的值 
#gf :global file 记录节点的采样数据 dataframe结构
#返回簇内采样数据的总数
def clusterSample_num(gf):
    return gf.at[0,'NT_g']

###function13  计算超网格结构
#1、h ：表示超立方体的宽度
# h 属于 [ ( ((3/8)^q) * (6*q / (n*sum(R(fi))) ) )^(1/(q+2)),  
#                    ( ((1/2)^q) * (6*q/(n*sum(R(fi)))) )^(1/(q+2))]
#多变量正太分布N（mu,sigma） mu和sigma都是向量
## R(fi) = 1/( (2^q+1) * pi^(q/2))
#temp = (6*(pi**0.5q)*(2**q+1))/n
# h* = ((Z1/Z2)*temp)**(1/q+2)
#2、q ：数据的纬度 温度、湿度、光照   q=3 
#函数参数
#q 数据空间维度
#N 簇内总采样样本数量，单次采样的总和  修改3.5。改为定长存储
##new
def get_HG_H(q,n):   # , n = size_of_storage
    #print('hypergrid_structure building...')
    temp = 6*( np.pi**(0.5*q) )*(2**(q+1))/(q*n)
    max_H = ( (0.5**q)*temp ) **(1/(q+2))
    min_H = ( ((3/8)**q)*temp ) **(1/(q+2))
#     print('获取h*范围[%f,%f]' %(min_H,max_H))
    return min_H,max_H

#b次方 pow(a,b) a^b
#或者 a**b
#hypergrid_structure_H(4,1200)
#((3/8)**(4)*4*(np.pi**2)/25)**(1/6)
#((1/2)**(4)*4*(np.pi**2)/25)**(1/6)
#hypergrid_structure_H(4,1200)
#hypergrid_structure_H(3,300)  获取h*范围[0.500230,0.594474]
#get_HG_H(4,300) 获取h*范围[0.561165,0.679803]

###function18  计算样本检测区域分布情况  #已修改为 电压 new 
#参数：
#df 本地数据dataframe ：本地数据正态化的  #intel_lab_type = ['Date','Time','Epoch','ID','Temperature','Humidity','Light','Voltage']
'''
CH_configure_type = ['NT_g','NH_g','NV_g',
                    'mean_T','mean_H','mean_V',
                    'std_T','std_H','std_V',
                    'TemperatureMaxNorm_g','TemperatureMinNorm_g',
                     'HumidityMaxNorm_g','HumidityMinNorm_g',
                     'VoltageMaxNorm_g','VoltageMinNorm_g'
                     #'N' 和上述变量重复
                     ,'H_MIN','H_MAX','B','C','D']
'''
#返回本地数据pos_list,num_list,maha_pos 训练数据最后一个映射坐标
def local_data_distribution(np_df,data_df):  #df 标准化后的数据(需要先标准化) 
    list_pos = [] #存储数据转化为超网栅结构中的坐标值。b 为每一维存储的比特位数。
    data_list = [] #存储不重复的数据
    num_list = [] #存储数据的重复个数
    pos = 0      #用于存储点的位置
   
    maha_pos_last = np.array([0,0,0])    #记录最后一个数据的曼哈顿坐标，用于下一轮的记录曼哈顿距离  4.2改
    #逐行查询df，将df中数据映射到hypergrid中。
    bits = int(np_df.at[0,'B'])
    i = 0
    for i in data_df.index.values:   #按照index索引依次提取数据
        #templist = []
        #templist.append(i)  #添加数据的index
        pos = 0
        #按照每维度信息 b比特存储
        #温度df.at[i,'Temperature']
        #print('温度：',df.at[i,'Temperature'] )
        ######判断是否为空 nan ,如果检测属性是nan值则该区域填充111... 全1
        if data_df.at[i,'Temperature'] != np.nan:
            ###h 修改
            pos_t = math.floor((data_df.at[i,'Temperature'] + np_df.at[0,'C'])/np_df.at[0,'H_MAX']) 
            maha_pos_last[0] = pos_t
            #print('temp  pos',pos_t)
            pos = pos_t<<2*bits | pos
        else:
            print('存在nan，位置',i)
            pos = (2**bits-1)<<2*bits | pos
            
        if data_df.at[i,'Humidity'] != np.nan:
            
            pos_h = math.floor((data_df.at[i,'Humidity'] + np_df.at[0,'C'])/np_df.at[0,'H_MAX'])
            maha_pos_last[1] = pos_h
            #print('humi pos',pos_h)
            pos = pos_h<<1*bits | pos
        else:
            print('存在nan，位置',i)
            pos = (2**bits-1)<<bits | pos
            
        if  data_df.at[i,'Voltage']  != np.nan:
            ###h 修改
            pos_v = math.floor((data_df.at[i,'Voltage'] + np_df.at[0,'C'])/np_df.at[0,'H_MAX'])
            maha_pos_last[2] = pos_v
            #print('light pos',pos_l)
            pos = pos_v | pos
        else:
            print('存在nan，位置',i)
            pos = (2**bits-1) | pos
        list_pos.append(pos)
        
    
    #去重复，统计重复的个数
    list_pos.sort()
    #生成两个列表 data[]  num[]
    count = 0
    data = -1
    #记录采集样本的数值及其数量
    for i in range(len(list_pos)):
        if i == 0:
            data = list_pos[i]
            data_list.append(data)
            count = 1
        else:
            if data != list_pos[i]:
                num_list.append(count)
                data =  list_pos[i]
                data_list.append(data)
                count = 1
            else:
                count += 1
        if i == len(list_pos)-1:
            num_list.append(count)
    #返回
    return data_list,num_list,maha_pos_last
 
###function19  联合两对数据及数量列表  new 
#参数：
#data_list 非重复数据 集合
#num_list 对应数据的数量
#返回：数据列表、数量列表
''''''    
def merge_data_distribution(data_list1,num_list1,data_list2,num_list2):
    if(len(data_list1)==0):
        data_l = data_list2
        sum_l = num_list2
    else:
        data_l = []  #记录总的非重复数据
        sum_l = []  #记录个数据的个数
        l1_i = 0
        l2_i = 0
        while  l1_i < len(data_list1) or l2_i < len(data_list2):
            #判断数组是否已经达到边界
            if l1_i>=len(data_list1):   #data_list1 已经加载完全。将data_list2的剩余数据加入
                
                while l2_i < len(num_list2):
                    data_l.append(data_list2[l2_i])
                    sum_l.append(num_list2[l2_i])
                    l2_i+=1
            elif l2_i>= len(num_list2): #data_list2 已经加载完全。将data_list1的剩余数据加入
                
                while l1_i < len(num_list1):
                    data_l.append(data_list1[l1_i])
                    sum_l.append(num_list1[l1_i])
                    l1_i+=1
                    
            else:  # data_list1、datalist2都没加载完
                #两个数组还未到达边界
                if data_list1[l1_i] > data_list2[l2_i]:   # datalist1的数据大于fatalist2的数据
                    data_l.append(data_list2[l2_i])
                    sum_l.append(num_list2[l2_i])
                    l2_i +=1
                    
                elif data_list1[l1_i] < data_list2[l2_i]: # datalist1的数据小于fatalist2的数据
                    data_l.append(data_list1[l1_i])
                    sum_l.append(num_list1[l1_i])
                    l1_i +=1
                else:    # datalist1的数据等于fatalist2的数据
                    data_l.append(data_list1[l1_i])
                    sum_l.append(num_list1[l1_i]+num_list2[l2_i])
                    l1_i +=1
                    l2_i +=1
    return data_l,sum_l
###function19 计算样本检测区域d的  new 
#子节点计算K*值
#参数：
#df 本地正态化的数据集 ：df
#size、percentage：本地数据计算k值的百分比采样
#ch_info: global file 记录簇内信息
#np_val ：记录本地数据的超网格内分布
#np_num：记录本地分布对应的数据量
#def hypergrid_structure_K_MN(df,percentage,gf,hg,np_val,np_num):   #s = 50  本地存储数据300 （*1/6）
def calculate_K(df,size,ch_info,np_val,np_num):   #s = 50  本地存储数据300 （*1/6）
    #df 标准化化后的
    
    #逐行查询df，将df中数据映射到hypergrid中。
    bits = int(ch_info.at[0,'B'])
    C = ch_info.at[0,'C']
#     print('bits',bits)
    #1、计算比例rate = |NP|/Volume of hg
    #4.15改
    '''
    side_t = math.ceil((ch_info.at[0,'TemperatureMaxNorm_g'] - ch_info.at[0,'TemperatureMinNorm_g'] )/ch_info.at[0,'H_MAX'])
    side_h =  math.ceil((ch_info.at[0,'HumidityMaxNorm_g'] - ch_info.at[0,'HumidityMinNorm_g'] )/ch_info.at[0,'H_MAX'])
    # 3月10号修改
    #side_l =  math.ceil((gf.at[0,'LightMaxNorm_g'] - gf.at[0,'LightMinNorm_g'] )/hg.at[0,'H_MAX']) #
    side_v =  math.ceil((ch_info.at[0,'VoltageMaxNorm_g'] - ch_info.at[0,'VoltageMinNorm_g'] )/ch_info.at[0,'H_MAX'])
    rate = len(np_val)/(side_t*side_h*side_v) 
    '''
    rate = len(np_val)/ch_info.at[0,'Vol']
#     print('vol',side_t*side_h*side_v)
    #print('side_l',side_l)
    #3月十号修改
   # rate = len(np_val)/(side_t*side_h*side_l) 
    #print('np_val ',np_val)
   
    #print('rate',rate)
    #2、提取百分之=percentage 的数据
    #df = df.sample(frac=percentage)      ####修改为个数
    df = df.sample(n=size)      ####修改为个数
    #print('df ',df)
    #df = df.reset_index(drop=True)
    #print('resample dataframe:',df)
    sample_size = len(df)  #采样的个数
    all_num = 0               #记录包含覆盖区域在内的数据量
    e_t = 0
    e_h = 0
    # 3月10号修改
    #e_l = 0
    e_v = 0
    last_pos = -1
    last_sum = 0
    for i in df.index.values:
        pos = 0 
        #按照每维度信息 b比特存储
        #计算pos信息
        pos_t = math.floor((df.at[i,'Temperature'] + C)/ch_info.at[0,'H_MAX']) 
        pos = pos_t<<2*bits | pos
        pos_h = math.floor((df.at[i,'Humidity'] + C)/ch_info.at[0,'H_MAX'])
        #print('humi pos',pos_h)
        pos = pos_h<<1*bits | pos
        pos_v = math.floor((df.at[i,'Voltage'] + C)/ch_info.at[0,'H_MAX'])
        pos = pos_v<<0*bits | pos
        #print('pos',pos)
        #遍历领域
        nparr = df.loc[i,:].values.tolist()
        L1_neighbor_pos = get_L1_Optimizing(nparr,C,bits,pos)
        for pos_ in L1_neighbor_pos:
            if pos_ in np_val:
                all_num += np_num[np_val.index(pos_)]
                last_sum +=np_num[np_val.index(pos_)]

        '''
        if df.at[i,'Temperature']+ ch_info.at[0,'C']  - math.floor(df.at[i,'Temperature']+ ch_info.at[0,'C']) >0.5:
            temp_e_t = 1
        else:
            temp_e_t = -1
        if df.at[i,'Humidity']+ ch_info.at[0,'C']  - math.floor(df.at[i,'Humidity']+ ch_info.at[0,'C']) >0.5:
            temp_e_h = 1
        else:
            temp_e_h = -1
        # 3月十号修改
        if df.at[i,'Voltage']+ ch_info.at[0,'C']  - math.floor(df.at[i,'Voltage']+ ch_info.at[0,'C']) >0.5:
            temp_e_v = 1
        else:
            temp_e_v = -1
        if temp_e_t == e_t and temp_e_h ==e_h and  temp_e_v == e_v and pos == lastpos: #数据相同分布，计算临近分布，可直接使用上次的计算值
            all_num += last_sum
        else:   #不同分布
            # 三维 H、T、L
            #找到所属超立方体，添加该立方体的数据量
            last_sum = 0
            if pos in np_val:
                all_num += np_num[np_val.index(pos)]
                last_sum += np_num[np_val.index(pos)]
            #V
            new_pos = pos + temp_e_v
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            # T
            new_pos = pos + temp_e_t*(1<<2*bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            # T、V
            new_pos = new_pos + temp_e_v
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum += np_num[np_val.index(new_pos)] 
            #H
            new_pos = pos + temp_e_h*(1<<bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum += np_num[np_val.index(new_pos)]
            #H、V
            new_pos = new_pos + temp_e_v
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            # T、H
            new_pos = pos +  temp_e_t*(1<<2*bits) + temp_e_h*(1<<bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            #H T V
            new_pos =new_pos+temp_e_v
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            #print('add num ', last_sum)
            e_t = temp_e_t
            e_h = temp_e_h
            e_v = temp_e_v
            lastpos = pos 
            '''
    #all_sum 记录所有的领域总数据量
#     print(' calculate_K ')
#     print('vol = %d, %d, %d' %(side_t,side_h,side_v))
#     print('rate ',rate)
#     print('all_num',all_num)
#     print('sample_size',sample_size)
    temp_k  = rate*all_num/sample_size
#     print('k* = %f'%temp_k)
#     print(' ')
    return temp_k
#test
'''
#查找对应变量对应的属性，及另一个列表对应位置的值
l2 = [1,2,3,4]
l3= [3,4,5,6]
l3[l2.index(2)]

k = hypergrid_structure_K_MN(df18,0.5,CH_info,HG_info,p2,s2)
k
'''
###function20  计算簇的k平均值
#参数：
#簇内各节点的k值
#返回值：簇内k的平均值
def hypergrid_structure_K_CH(k_list):
    return np.mean(k_list)

###function21 Intel 数据集  时间窗口属性计算函数###  即子节点内部检测函数#
#参数 
#np_val  :簇内hypergrid 信息
#np_num ：对应超立方体内的数据量
#ct_df :CheckTable_type = ['Data_Index','HG_pos']  
#df: 原数据集：标准化的
#hg:  hypergrid 信息
#k :阈值
#outlier_insertpos_list 插入异常的列表
#返回：异常值的列表

def self_detect(np_val,np_num,ct_df,df,hg,k):   #检测是否存在nan值
    #提取查找表中的index值
    detect_error_l = []
    #print(np_val)
    #print(np_num)
    bits = int(hg.at[0,'B'])
  
    for i in ct_df.index.values:
        #print(np_val.index(ct_df.at[i,'HG_Pos']))
        #print(np_num[np_val.index(ct_df.at[i,'HG_Pos'])])
        if np_num[np_val.index(ct_df.at[i,'HG_Pos'])] > k: # 大于k个数据表示该数据点为正常
        #if np_num[np_val.index(ct_df.at[i,'HG_Pos'])] > 5: # 大于k个数据表示该数据点为正常
            #print('normal')
            pass
        else:    #所在超立方体内数据量小于k,则进一步检查领域
                  #遍历领域
            if df.at[ct_df.at[i,'Data_Pos'],'Temperature'] + hg.at[0,'C']  - math.floor(df.at[ct_df.at[i,'Data_Pos'],'Temperature'] + hg.at[0,'C']) >0.5:
                temp_e_t = 1
            else:
                temp_e_t = -1
            if df.at[ct_df.at[i,'Data_Pos'],'Humidity']+ hg.at[0,'C']  - math.floor(df.at[ct_df.at[i,'Data_Pos'],'Humidity']+ hg.at[0,'C']) >0.5:
                temp_e_h = 1
            else:
                temp_e_h = -1
            if df.at[ct_df.at[i,'Data_Pos'],'Light']+ hg.at[0,'C']  - math.floor(df.at[ct_df.at[i,'Data_Pos'],'Light']+ hg.at[0,'C']) >0.5:
                temp_e_l = 1
            else:
                temp_e_l = -1
            #不同分布
            # 三维 H、T、L
            #找到所属超立方体，添加该立方体的数据量
            all_num = 0
            pos = ct_df.at[i,'HG_Pos']
            if pos in np_val:
                all_num += np_num[np_val.index(pos)]
            #T
            new_pos = pos + temp_e_t*(1<<2*bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
            #H
            new_pos = pos + temp_e_h*(1<<bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
            #L
            new_pos = pos + temp_e_l
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
            # T、H
            new_pos = pos +  temp_e_t*(1<<2*bits) + temp_e_h*(1<<bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
            # T、L
            new_pos = pos +  temp_e_t*(1<<2*bits) + temp_e_l
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
            #H、L
            new_pos = pos +  temp_e_h*(1<<bits) + temp_e_l
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
            #H T L
            new_pos = pos +  temp_e_t*(1<<2*bits) + temp_e_t*(1<<bits)+temp_e_t
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
            #得到领域的统计总数：all_sum
            #print(all_num)
            if all_num < k:
            #if all_num < 5:
                detect_error_l.append(ct_df.at[i,'Data_Pos'])
    return detect_error_l
##test 
#errorlist = self_detect(p2,s2,ct_df18,df18,HG_info,k_mean)
###function22  计算两个列表总相同数据的个数
#参数 
#norm_l  :标准列表：代表异常插入列表
#test_l ：检测列表：代表检测出异常的汇总
#返回：ACC准确率 、FPR误报率
def compute_same_num_no_error(test_l,size):  #可以改进

    if len(test_l) != 0:                                    # 3.5修改
        #P: detect error
        #N: normal data
        FP = len(test_l)            
        TP = 0
        TN = size - FP            
        FN = 0
        
        print(' TN ',TN,' FN ',FN,' FP ',FP,' TP ',TP)
        if size != 0:
            ACC = (TP + TN)/size                   #准确率
        else:
            ACC = 0
            print('size = 0,分母为0')
        if TN+FP!=0:
            FPR = FP/(TN+FP)                       #假正例率 误判率
        else:
            FPR = 0
            print('TN+FP = 0,分母为0')
        if TP+FN !=0:
            TPR = TP/(TP+FN)                       #真正例率
        else:
            print('TP+FN = 0')
            TPR = 0
        if TP + FP!=0:
            P = TP/(TP + FP)                       #查准率
        else:
            print('TP + FP = 0')
            P =0
#         R = TP/(TP + FN)                       #召回率  TPR 相同
        print(' ACC(准确率): ',ACC,' FPR(假正例率): ',FPR,' TPR(真正例率): ',TPR,'P(召回率): ',P)
        return ACC,FPR,TPR,P
    else:
        print('detect_error_list is empty')
        return 1,0,1,1
def compute_performent(label_l,detected_l,size):  
    TP = 0
    for index in detected_l:
        if index in label_l:
            TP+=1
    FN = len(label_l)-TP
    FP = len(detected_l)-TP 
    TN = size - TP - FN - FP
    if size != 0:
        ACC = (TP + TN)/size                   #准确率
    else:
        ACC = 0
        print('size = 0,分母为0')
    if TN+FP!=0:
        FPR = FP/(TN+FP)                       #假正例率 误判率
    else:
        FPR = 0
        print('TN+FP = 0,分母为0')
    if TP+FN !=0:
        TPR = TP/(TP+FN)                       #真正例率 Recall definition
    else:
        print('TP+FN = 0')
        TPR = 0
    if TP + FP != 0:
        P = TP/(TP + FP)                       #查准率
    else:
        print('TP + FP = 0')
        P =0
    if P+TPR !=0:
        F1 = 2*P*TPR /(P+TPR)                #F-Score就是 Precision和 Recall的加权调和
    #         R = TP/(TP + FN)                       #召回率  TPR 相同
    else:
        F1 = 0
    print(' ACC(准确率): ',ACC,' FPR(假正例率): ',FPR,' TPR(真正例率 召回率): ',TPR,'P(查准率): ',P,"F1: ", F1)
    return TN,FN,FP,TP,ACC,FPR,TPR,P,F1
def compute_same_num(norm_l,test_l,size):  #可以改进
    # norm_l 标记数据集
    # test_l 检测数据集
    # size 数据大小
    if len(norm_l) !=0: #存在异常插入
        print('Detect error...')
        dt_num = 0
        for i in test_l:
            if i in norm_l:
                dt_num+=1   #检测出的错误数量  # len(test_l) - dt_num = 误判的量 NP
        print('same = ',dt_num)
        if len(test_l) != 0:                                    # 3.5修改
            '''
            TN = dt_num             #本来是false
            FN = len(test_l)-dt_num  #本来是true
            FP = len(norm_l) - dt_num #
            TP = size - len(norm_l) - len(test_l) + dt_num
            '''
            # P: error data   2019 12 19
            # N: correct data
            TP = dt_num  
            FP = len(test_l) - dt_num
            FN = len(norm_l) - dt_num
            TN = size - len(norm_l) - len(test_l) + dt_num

            print(' TN ',TN,' FN ',FN,' FP ',FP,' TP ',TP)
            if size != 0:
                ACC = (TP + TN)/size                   #准确率
            else:
                ACC = 0
                print('size = 0,分母为0')
            if TN+FP!=0:
                FPR = FP/(TN+FP)                       #假正例率 误判率
            else:
                FPR = 0
                print('TN+FP = 0,分母为0')
            if TP+FN !=0:
                TPR = TP/(TP+FN)                       #真正例率 Recall definition
            else:
                print('TP+FN = 0')
                TPR = 0
            if TP + FP != 0:
                P = TP/(TP + FP)                       #查准率
            else:
                print('TP + FP = 0')
                P =0
            if P+TPR !=0:
                F1 = 2*P*TPR /(P+TPR)                #F-Score就是 Precision和 Recall的加权调和
            #         R = TP/(TP + FN)                       #召回率  TPR 相同
            else:
                F1 = 0
            print(' ACC(准确率): ',ACC,' FPR(假正例率): ',FPR,' TPR(真正例率 召回率): ',TPR,'P(查准率): ',P,"F1: ", F1)
            return TN,FN,FP,TP,ACC,FPR,TPR,P,F1
        else: #detect nothing
            print('detect_error_list is empty')
            TP = 0  
            FP = 0
            FN = len(norm_l)
            TN = size - len(norm_l)

            return  TN,FN,FP,TP,ACC,FPR,TPR,P,F1
    else:  #无异常插入
        print('without  err...')
        if len(test_l) != 0:                                    # 3.5修改
            FP = len(test_l)             #误判
            TP = 0
            TN = size - FP            #本来是false
            FN =  0  #本来是true
            print(' TN ',TN,' FN ',FN,' FP ',FP,' TP ',TP)
            if size != 0:
                ACC = (TP + TN)/size                   #准确率
            else:
                ACC = 0
                print('size = 0,分母为0')
            if TN+FP!=0:
                FPR = FP/(TN+FP)                       #假正例率 误判率
            else:
                FPR = 0
                print('TN+FP = 0,分母为0')
            if TP+FN !=0:
                TPR = TP/(TP+FN)                       #真正例率
            else:
                print('TP+FN = 0')
                TPR = 0
            if TP + FP!=0:
                P = TP/(TP + FP)                       #查准率
            else:
                print('TP + FP = 0')
                P =0
            if P+TPR !=0:
                F1 = 2*P*TPR /(P+TPR)                #F-Score就是 Precision和 Recall的加权调和
            else:
                F1 = 0

            #         R = TP/(TP + FN)                       #召回率  TPR 相同
            print(' ACC(准确率): ',ACC,' FPR(假正例率): ',FPR,' TPR(真正例率 召回率): ',TPR,'P(查准率): ',P,"F1: ", F1)
            return TN,FN,FP,TP,ACC,FPR,TPR,P,F1
        else:
            print('detect_error_list is empty')
            return size,0,0,0,1,0,0,0,0
# ##function 24  单数据映射   ##new
# #参数，normal profile，数据series
# #指标
def getKappaScore(tp,tn,fp,fn):
    acc = (tp+tn)/(tp+tn+fp+fn)
    p_true = (tp+fn)*(tp+fp)/( (tp+tn+fp+fn) * (tp+tn+fp+fn) )
    p_false = (tn+fp)*(tn+fn)/( (tp+tn+fp+fn) * (tp+tn+fp+fn) )
    pc = p_true + p_false
    kappa = (acc-pc)/(1-pc)
    return kappa

def get_mapPos(np_,temp,humi,voltage):   # 此处数据已经经过预处理  3月14日改 已经标准化过。
    #print('get_mapPos')
    #print('np_',np_)
    pos = 0
    bits = int(np_.at[0,'B'])
    maha_dis = np.array([0,0,0])
    #print(bits)
    if temp != np.nan:
        ###h 修改
        pos_t = int(math.floor((temp + np_.at[0,'C'])/np_.at[0,'H_MAX']))
        maha_dis[0]=pos_t
        #print('temp  pos',pos_t)
        pos = pos_t<<2*bits | pos
    else:
        #print('存在nan，位置',i)
        pos = (2**bits-1)<<2*bits | pos
    #print('or temp',pos)
    #湿度df.at[i,'Humidity']
    #print('湿度：',df.at[i,'Humidity'])
    if humi != np.nan:
        pos_h = int(math.floor((humi + np_.at[0,'C'])/np_.at[0,'H_MAX']))
        pos = pos_h<<1*bits | pos
        maha_dis[1] = pos_h
    else:
        #print('存在nan，位置',i)
        pos = (2**bits-1)<<bits | pos
    if  voltage != np.nan:
        ###h 修改
        pos_v = int(math.floor((voltage + np_.at[0,'C'])/np_.at[0,'H_MAX']))
        #print('light pos',pos_l)
        pos = pos_v | pos
        maha_dis[2] = pos_v
        
    else:
        #print('存在nan，位置',i)
        pos = (2**bits-1) | pos
    return pos,maha_dis
#邻域计算方式

def get_delta_pos(e_t,e_h,e_v,map_):  # l1   去除本数据点
    return np.dot(np.array([
        [0,0,e_v],
        [0,e_h,0],
        [e_t,0,0],
        [0,e_h,e_v],
        [e_t,0,e_v],
        [e_t,e_h,0],
        [e_t,e_h,e_v]]),map_)


# 获取偏移检测向量 
#nparr = [T,H,V]

def get_test_direct(nparr,C):
    if nparr[0] + C - np.floor(nparr[0]+C) >0.5:
        e_t = 1
    else:
        e_t = -1
    if nparr[1] + C  - np.floor(nparr[1]+ C) >0.5:
        e_h = 1
    else:
        e_h = -1
    # 3月十号修改
    if nparr[2]+ C  - np.floor(nparr[2]+ C) >0.5:
        e_v = 1
    else:
        e_v = -1
    return e_t,e_h,e_v

def get_L1_simple(nparr,C,B,pos):     #返回简化的L1 领域信息
    if nparr[0] + C - np.floor(nparr[0]+C) >0.5:
        e_t = 1
    else:
        e_t = -1
    if nparr[1] + C  - np.floor(nparr[1]+ C) >0.5:
        e_h = 1
    else:
        e_h = -1
    # 3月十号修改
    if nparr[2]+ C  - np.floor(nparr[2]+ C) >0.5:
        e_v = 1
    else:
        e_v = -1
    map_ = np.array([2**(2*B),2**B,1])
    detla_pos = get_delta_pos(e_t,e_h,e_v,map_)
#     print(detla_pos)
    L1_simple_pos = detla_pos+pos
    return L1_simple_pos
def get_L1_Optimizing(nparr,C,B,pos):
    # 191218
    e = []  #e for delta pos
    for i in range(len(nparr)):
        if nparr[i] + C - np.floor(nparr[i]+C) >0.9:
            e.append([0,1])
        elif nparr[i] + C - np.floor(nparr[i]+C) <0.1:
            e.append([0,-1])
        else:
            e.append([0,1,-1])
            '''
    for i in range(len(nparr)):
        if isinstance(e[i],list):
            print(e[i])
        else:
            print(e[i])
            '''
    L1_optimizie_pos = []  
    for i in e[0]:
        for j in  e[1]:
            for z in e[2]:
                L1_optimizie_pos.append(pos+i*2**(2*B)+j*2**B+z)
    return L1_optimizie_pos
    #map_ = np.array([2**(2*B),2**B,1])
    #detla_pos = get_delta_pos(e_t,e_h,e_v,map_)
#     print(detla_pos)
    #L1_simple_pos = detla_pos+pos
    #return L1_simple_pos
    
def get_L1_remaining(pos_simple,B,pos): #返回L1领域剩余超立方体
    l1 = [-1,0,1]
    l2 = [-1,0,1]
    l3 = [-1,0,1]
    delta = []
    for x in l1:
        for y in l2:
            for z in l3:
    #                 print(x,y,z)
                if l1 == 0 and l2 == 0 and l3 ==0:  #一开始计算过
                    pass
                else:
                    delta.append(x*(2**(2*B))+y*(2**B)+z + pos)
#     print(delta)
#     print(pos_simple)
    pos_remain = []
    for i in range(len(delta)):
        if delta[i] not in pos_simple:
            pos_remain.append(delta[i]) 
#     print(pos_remain)
    return pos_remain
# #单个数据点检测函数， 4月2号改 new
def onePoint_detect(pos_all,num_all,k,np_,temp,humi,volt,row):
    #if row >537:
    #    print('debug')
    #检测单元格数量
    test_cell_num = 0 
    #数值
#     data_exceed_3std = False
    
    temp_norm,humi_norm,volt_norm = onedata_norm(np_,temp,humi,volt)       #提取温湿度信息
#     if np.abs(temp_norm) >= 3 or np.abs(humi_norm) >= 3 or np.abs(volt_norm) >= 3:
#         data_exceed_3std = True
#         print('discover normed data exceed 3',temp_norm,humi_norm,volt_norm)
    pos,maha_now = get_mapPos(np_,temp_norm,humi_norm,volt_norm)                    #获取映射点信息
    #统计邻域数据量
    sum_ = 0
    #num_in_hypergrid = 0    #记录本单元格数据量
    num_in_l1hyperGrid = 0  
#     print('pos',pos)
#     print('HyperGrid .........')
    #0、检测所在超立方体数据量
    '''
    if pos in pos_all:   #检测异常                         #判断数据所在单元格是否已经存在pos_all中
        num_in_hypergrid = num_all[pos_all.index(pos)]
        test_cell_num +=1                        #检测自身所在单元格
        if num_in_hypergrid > k:
#             print('in HyperGrid is normal num = %d' %num_in_hypergrid)
            return True,test_cell_num,maha_now,pos,num_in_hypergrid#,data_exceed_3std           #如果超过阈值，则返回True 退出
    sum_+= num_in_hypergrid                       #记录单元格数量
    '''
    #1、判断L1-简化邻域 1221 l1 优化领域包含自身
#     print('L1_simple .........')
    L1_simple_num = 0                                                     #保存简化领域的数据量
    arr = np.array([temp_norm,humi_norm,volt_norm])                       #将标准化的数据，整合和矩阵
#     print(arr)
    #L1_simple_area  = get_L1_simple(arr,np_.at[0,'C'],np_.at[0,'B'],pos)  #获取邻域的坐标值
    L1_simple_area = get_L1_Optimizing(arr,np_.at[0,'C'],np_.at[0,'B'],pos)
#     print('L1_simple_area',L1_simple_area)
#     print('L1_simple_area',L1_simple_area)
    for simple_pos in L1_simple_area:     #逐个取简化领域的数据点
        test_cell_num +=1  #检测单元格数+1
        if simple_pos in pos_all:         #判断是否在数组中
            L1_simple_num += num_all[pos_all.index(simple_pos)]  #累计l1-简化域的数据量           
            sum_ += num_all[pos_all.index(simple_pos)]           #累计当前遍历超立方体的数据量
            if simple_pos != pos:
                num_in_l1hyperGrid += num_all[pos_all.index(simple_pos)] 
            if sum_ > k:       #如果超过阈值，则返回True 退出
#                 print('in L1_simple_area is normal num = %d' %sum_)
#                 return True,test_cell_num,maha_now,pos,data_exceed_3std
                return True,test_cell_num,maha_now,pos,num_in_l1hyperGrid
            
#     print('L1_remaining .........')
    #3、数据量未超过阈值，判断L1-余下邻域
    '''191218 余下不检验
    L1_remaining_num = 0
    L1_remaining_area = get_L1_remaining(L1_simple_area,np_.at[0,'B'],pos)
#     print('L1_remaining_area',L1_remaining_area)
    
    for remain_pos in L1_remaining_area:
#         print('remain_pos',remain_pos)
        test_cell_num +=1                   #检测单元格数+1
        if remain_pos in pos_all:              #缺少判断
            L1_remaining_num +=  num_all[pos_all.index(remain_pos)]
            sum_ += num_all[pos_all.index(remain_pos)] 
            if sum_ > k:                    #如果超过阈值，则返回True 退出
#                 print('in L1_remaining_area is normal num = %d' %sum_)
                return True,test_cell_num,maha_now,pos,sum_#,data_exceed_3std   #如果超过阈值，则返回True 退出   
    #4end 数据量仍然小于K 则返回false
   '''
#     print('Sample an error data')
#     return False,test_cell_num,maha_now,pos,data_exceed_3std
    return False,test_cell_num,maha_now,pos,num_in_l1hyperGrid
def update_k(df,gf,hg,np_val,np_num,pos_local,index_list):
    ##MHs 先分布式计算k*，CH 在求平均
    e_t = -2
    e_h = -2
    e_l = -2
    print('pos_local',pos_local)
    print('index_list',index_list)
    bits = int(hg.at[0,'B'])
    #1、计算比例rate = |NP|/Volume of hg
    side_t = math.ceil((gf.at[0,'TemperatureMaxNorm_g'] - gf.at[0,'TemperatureMinNorm_g'] )/hg.at[0,'H_MAX'])
    side_h =  math.ceil((gf.at[0,'HumidityMaxNorm_g'] - gf.at[0,'HumidityMinNorm_g'] )/hg.at[0,'H_MAX'])
    side_l =  math.ceil((gf.at[0,'LightMaxNorm_g'] - gf.at[0,'LightMinNorm_g'] )/hg.at[0,'H_MAX'])
    #print('side_l',side_l)
    rate = len(np_val)/(side_t*side_h*side_l)
    print('rate',rate)
    all_num = 0
    last_sum = 0
    last_pos = -1
    for i  in range(len(pos_local)):
        pos = pos_local[i]
        #遍历领域
        if df.at[index_list[i],'Temperature']+ hg.at[0,'C']  - math.floor(df.at[index_list[i],'Temperature']+ hg.at[0,'C']) >0.5:
            temp_e_t = 1
        else:
            temp_e_t = -1
        if df.at[index_list[i],'Humidity']+ hg.at[0,'C']  - math.floor(df.at[index_list[i],'Humidity']+ hg.at[0,'C']) >0.5:
            temp_e_h = 1
        else:
            temp_e_h = -1
        if df.at[index_list[i],'Light']+ hg.at[0,'C']  - math.floor(df.at[index_list[i],'Light']+ hg.at[0,'C']) >0.5:
            temp_e_l = 1
        else:
            temp_e_l = -1
        ''' 
        print('temp_e_t',temp_e_t)
        print('temp_e_h',temp_e_h)
        print('temp_e_l',temp_e_l)
        '''   
        if temp_e_t == e_t and temp_e_h ==e_h and  temp_e_l == e_l and pos == last_pos: #数据相同分布，计算临近分布，可直接使用上次的计算值
            '''   
            print('same.....................')
            print('last_sum=',last_sum)
            print('.....................')
            '''   
            all_num += last_sum
        else:   #不同分布
             # 三维 H、T、L
            #print('different................')
            last_pos = pos
            #找到所属超立方体，添加该立方体的数据量
            if pos in np_val:
                all_num += np_num[np_val.index(pos)]
                last_sum +=np_num[np_val.index(pos)]
            #L
            new_pos = pos + temp_e_l
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            # T
            new_pos = pos + temp_e_t*(1<<2*bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            # T、L
            new_pos = new_pos + temp_e_l
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)] 
            #H
            new_pos = pos + temp_e_h*(1<<bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            #H、L
            new_pos = new_pos + temp_e_l
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            # T、H
            new_pos = pos +  temp_e_t*(1<<2*bits) + temp_e_h*(1<<bits)
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            #H T L
            new_pos =new_pos+temp_e_t
            if new_pos in np_val:
                all_num += np_num[np_val.index(new_pos)]
                last_sum +=np_num[np_val.index(new_pos)]
            #print('add num ', last_sum)
            e_t = temp_e_t
            e_h = temp_e_h
            e_l = temp_e_l
    print('rate =',rate)
    print('all_num =',all_num)
    print('index len = ',len(index_list))
    return rate*all_num/len(index_list)
#2019 12 24
def insert_outlier(df,begin,num,delta_mean,delta_std_times):
    tempdf = df.copy()
    insert_list = random.sample(range(begin,len(tempdf)),num) #
    mean = tempdf.mean() + delta_mean
    std = tempdf.std() * delta_std_times
    #按照不同的异常插入
    #产生异常数据集
    t_error = np.random.normal(mean[0],std[0], num)
    h_error = np.random.normal(mean[1],std[1], num)
    v_error = np.random.normal(mean[2],std[2], num)
    
    noise_data = np.vstack((t_error,h_error))
    noise_data = np.vstack((noise_data,v_error))
    noise_data = noise_data.T

        #添加异常
    for i in range(len(insert_list)):
        for j in range(tempdf.shape[1]):
            tempdf.iloc[insert_list[i],j] = noise_data[i,j]
    return tempdf,insert_list,noise_data

def insert_noise(df1,begin,num,delta_mean,delta_std_times):
    tempdf = df1.copy()
    insert_list = random.sample(range(begin,len(tempdf)),num) #
    mean = tempdf.mean() + delta_mean
    std = tempdf.std() * delta_std_times
    #按照不同的异常插入
    #产生异常数据集
    t_error = np.random.normal(mean[0],std[0], num)
    h_error = np.random.normal(mean[1],std[1], num)
    v_error = np.random.normal(mean[2],std[2], num)
    
    noise_data = np.vstack((t_error,h_error))
    noise_data = np.vstack((noise_data,v_error))
    noise_data = noise_data.T

        #添加异常
    for i in range(len(insert_list)):
        for j in range(tempdf.shape[1]):
            tempdf.iloc[insert_list[i],j] = noise_data[i,j]
    return tempdf,insert_list,noise_data





# 插入异常函数 4.13
def insert_noise_error(df1,df2,df3,datalen,num,type_='default'):
    tempdf = df1.copy()
#     print(tempdf)
    size = int(len(df1)/datalen)
#     print(size)
    insert_list = random.sample(range(1,size),num) #从dataframe数据中，提取异常插入的序列
#     insert_list = [0,1,2]
    insert_list.sort()
#     print(insert_list)
    #计算
    mean = np.zeros([num,3])
    var = np.zeros([num,3])
#     print('mean',mean)
#     print('var',var)
    noise_pos = np.array([])
    noise_data = np.array([0,0,0])
    t_error = np.array([])
    h_error = np.array([])
    v_error = np.array([])
    
    for i in range(len(insert_list)):
        point = insert_list[i]
        data_all = df1[point*datalen:(point+1)*datalen]
        data_all = data_all.append(df2[point*datalen:(point+1)*datalen],ignore_index=True)
        data_all = data_all.append(df3[point*datalen:(point+1)*datalen],ignore_index=True)
        if  type_ == 'Xi-eye':
            temp_noise_pos = random.sample(range(point*datalen,(point+1)*datalen),int(datalen/10))  #5.21修改
            # temp_noise_pos = random.sample(range(point*datalen,(point+1)*datalen),int(datalen/6)) 
        else:
            temp_noise_pos = random.sample(range(point*datalen,(point+1)*datalen),int(datalen/3)) 
        temp_noise_pos.sort()
        noise_pos = np.append(noise_pos,temp_noise_pos)

#         np.vstack((,))
#         data_all = np.vstack((data_all,df3[point*300:(point+1)*300]))
#         print(data_all)
        mean[i] = data_all.mean()
        var[i] = data_all.std()
        if type_ == 'Intel':

            ##E1
            mean[i] += 3
            var[i] *= 1.5
            #E2
            #mean[i] += 0.5
            #var[i] *= 1.5
#             mean[i] += 5
#             var[i] *= 1.5
        elif type_ == 'Sensorscope':
#             mean[i] += 0
#             var[i] *= 1.5
            #E2
#             mean[i] += 0.5
#             var[i] *= 1.5
            mean[i] += 5
            var[i] *= 1.5

        elif type_ == 'Xi-eye':
            ## 1
            print(mean[i])
            print(var[i])
            for j in range(len( mean[i])):
                if j==2:
                    print('Xi-eye')
                    mean[i][j] += 0
                    var[i][j] *=1.5
                else:
                    mean[i][j] += 0
                    var[i][j] *=1.5
            print(mean[i])
            print(var[i])
            print('')
            ## 2
#             print(mean[i])
#             print(var[i])
#             for j in range(len( mean[i])):
#                 if j==2:
#                     print('Xi-eye')
#                     mean[i][j] += 10
#                     var[i][j] *=1.5
#                 else:
#                     mean[i][j] += 0.5
#                     var[i][j] *=1.5
#             print(mean[i])
#             print(var[i])
#             print('')
            ## 3
#             print(mean[i])
#             print(var[i])
#             for j in range(len( mean[i])):
#                 if j==2:
#                     print('Xi-eye')
#                     mean[i][j] += 100
#                     var[i][j] *= 1.5
#                 else:
#                     mean[i][j] += 5
#                     var[i][j] *=1.5
#             print(mean[i])
#             print(var[i])
#             print('')
          ## 4
#             for j in range(len( mean[i])):
#                 if j==2:
#                     print('Xi-eye')
#                     mean[i][j] += 100
#                     var[i][j] *= 1.5
#                 else:
#                     mean[i][j] += 10
#                     var[i][j] *=1.5

#             print('')
        #else:
         #   mean[i] += 0
          #  var[i] *= 1.5
#         print('mean[i]',mean[i])
#         print('var[i]',var[i])

        #sensorscope  参数
#         mean[i] += 5
#         var[i] *= 1.5

        #
    
    
        #产生异常数据集
        t_error = np.random.normal(mean[i][0],var[i][0], int(datalen/3))
        h_error = np.random.normal(mean[i][1],var[i][1], int(datalen/3))
        v_error = np.random.normal(mean[i][2],var[i][2], int(datalen/3))
        
        
        temp_data = np.vstack((t_error,h_error))
        temp_data = np.vstack((temp_data,v_error))
        temp_data = temp_data.T
        noise_data = np.vstack((noise_data,temp_data))
        #添加异常
        for i in range(len(temp_noise_pos)):
            tempdf.at[temp_noise_pos[i],'Temperature'] = t_error[i]
            tempdf.at[temp_noise_pos[i],'Humidity'] = h_error[i]
            tempdf.at[temp_noise_pos[i],'Voltage'] = v_error[i]
        
    return tempdf,noise_pos,noise_data
#     print(mean)
#     print(var)
# #     mean = mean + 0.5
# #     var = var*1.5
#     print(mean)
#     print(var)
    
#     var = 
#离散异常事件仿真
'''
def simulate_file(df,interval,times):
    tempdf = df.copy()
#     print(tempdf)
    size = int(len(tempdf)/interval)
    print(size)
    insert_list = random.sample(range(1,size),times) #从dataframe数据中，提取异常插入的序列
    insert_list.sort()
    print(insert_list)
    for i in insert_list:
        print(tempdf[i:i+10])
        for j in range(interval):
            tempdf.at[i*interval+j,'Temperature'] = tempdf.at[i*interval+j,'Temperature'] + 0.01*(np.e**(j/100)-1)
        print(tempdf[i:i+10])
    return tempdf
def simulate_high_Humidity(df,interval,times):
    tempdf = df.copy()
#     print(tempdf)
    size = int(len(tempdf)/interval)
    print(size)
    insert_list = random.sample(range(1,size),times) #从dataframe数据中，提取异常插入的序列
    insert_list.sort()
    print(insert_list)
    for i in insert_list:
        print(tempdf[i:i+10])
        for j in range(interval):
            random_humidity = np.random.random(1)*10+90
            tempdf.at[i*interval+j,'Humidity'] = random_humidity[0]
        print(tempdf[i:i+10])
    return tempdf
def simulate_power_off(df,interval,times):
    tempdf = df.copy()
#     print(tempdf)
    size = int(len(tempdf)/interval)
    print(size)
    insert_list = random.sample(range(1,size),times) #从dataframe数据中，提取异常插入的序列
    insert_list.sort()
    print(insert_list)
    for i in insert_list:
        print(tempdf[i:i+10])
        for j in range(interval):
            random_humidity = np.random.random(1)*10+90
            tempdf.at[i*interval+j,'Voltage'] = 0
        print(tempdf[i:i+10])
    return tempdf
'''
#test
# df1 = simulate_power_off(data1[0:6000],100,2)
# df1['Voltage'].plot()
if __name__ == "__main__":
    nparr = [1.5,2.1,3]
    C = 4
    B = 3
    pos = 100
    get_L1_Optimizing(nparr,C,B,pos)