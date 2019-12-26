#coding:utf-8
import pandas as pd
import numpy as np
import random
# anomaly insert functions: 

#fun1 outlier + contexual anomaly
def insert_outlier_error(df,type_l,start,size):
    insert_list = random.sample(range(start,len(df)),size) #从dataframe数据中，提取异常插入的序列  
    if len(type_l) == 0:
        insert_list = []
    else:
        for val in insert_list:
            for index in type_l:
                if np.random.random()>0.2: #outlier
                    df.iloc[val,index] = df.iloc[val,index] + random.randint(20,30)
                else: #contexual outlier
                    df.iloc[val,index] = df.iloc[(val+1000)%len(df),index] 
        #df['Temperature'].plot()
        insert_list = np.sort(insert_list)
    return df,insert_list


#fun2 constant anomaly
#df, start, size, error_type, type_l, delta_mean = 2, delta_std_times = 1.5):
def insert_constant_error(df,type_l,start,size,period = 30):
    #size num = constant
    periodSize = int(len(df-start)/period)
    assert(periodSize > size), "datasize is to small"
    insert_index = random.sample(range(1,periodSize-1),size) #从dataframe数据中，提取异常插入的序列
    insert_list = []
   
    if len(type_l) == 0:
        insert_list = [] 
    else:
        for index in type_l:
            #insert_list = random.sample(range(start,len(df)-period),size) #从dataframe数据中，提取异常插入的序列   
            for val in insert_index:
                s_ = val*period + start
                df.iloc[s_:s_+period,index] = 100
                #insert_list.append([pos in range(s_:s_+period)])
                temp_l = [pos for pos in range(s_,s_+period)]
                insert_list += temp_l
            #insert_list.append(range(val,val+period)) 
        #df['Temperature'].plot()
        insert_list = np.sort(insert_list)
    return df,insert_list


#fun3 insert noise anomaly
def insert_noise_error(df,type_l,start,size,delta_mean,delta_std_times):
    tempdf = df.copy()
    insert_list = random.sample(range(start,len(tempdf)),size) #
    mean = tempdf.mean() + delta_mean
    std = tempdf.std() * delta_std_times
    #按照不同的异常插入
    #产生异常数据集
    if len(type_l) ==0:
        insert_list = []
    else:
        for val in type_l:
            error_l = np.random.normal(mean[val],std[val], size)
            for i in range(size): 
                tempdf.iloc[insert_list[i],val] = error_l[i]
    return tempdf,insert_list



#fun3 insert noise anomaly 
def insert_noise_error_all(df,type_l,start,size,delta_mean,delta_std_times):
    tempdf = df.copy()
    insert_list = random.sample(range(start,len(tempdf)),size) #
    mean = tempdf.mean() + delta_mean
    std = tempdf.std() * delta_std_times
    #按照不同的异常插入
    #产生异常数据集
    t_error = np.random.normal(mean[0],std[0], size)
    h_error = np.random.normal(mean[1],std[1], size)
    v_error = np.random.normal(mean[2],std[2], size)
    
    noise_data = np.vstack((t_error,h_error))
    noise_data = np.vstack((noise_data,v_error))
    noise_data = noise_data.T

        #添加异常
    for i in range(len(insert_list)):
        for j in range(tempdf.shape[1]):
            tempdf.iloc[insert_list[i],j] = noise_data[i,j]
    return tempdf,insert_list

def insert_anomaly(df, start, size, error_type, type_l, delta_mean = 2, delta_std_times = 1.5):
    '''
    df : original dataframe
    start: = databuffer
    size: anomaly datasize
    error_type: 
    delta_mean: default 2 for noise anomaly insert [mean+delta]
    delta_std_times: default 2 for noise anomaly [insert std*times]
    '''
    assert (error_type in ['outlier','constant','noise']), 'wrong error_type'
    insert_list = []
    if error_type == 'outlier':
        df,insert_list = insert_outlier_error(df,type_l,start,size)
    elif error_type == 'constant':
        df,insert_list = insert_constant_error(df,type_l,start,size) 
    elif error_type == 'noise':
        #insert_noise_error(df,type_l,start,size,delta_mean,delta_std_times)
        df,insert_list = insert_noise_error(df,type_l,start,size,delta_mean,delta_std_times)
    else:
        print('error error_type')
    return df,insert_list 


#fun1 noise 按照周期插入异常 no use
def insert_noise_error_org(df,noisePeriod,size):
    tempdf = df.copy()
    size_ = int(len(df)/noisePeriod)
    insert_list = random.sample(range(1,size_),size) #从dataframe数据中，提取异常插入的序列
    insert_list.sort()
    #计算
    mean = np.zeros([size,3])
    var = np.zeros([size,3])
    
    noise_pos = np.array([]) #record noise position
    noise_data = np.array([0,0,0])  # noise array
    t_error = np.array([])
    h_error = np.array([])
    v_error = np.array([])
    
    for i in range(len(insert_list)):
        point = insert_list[i]
        data_all = df[point*noisePeriod:(point+1)*noisePeriod]
        temp_noise_pos = random.sample(range(point*noisePeriod,(point+1)*noisePeriod),int(noisePeriod/3))  # sample noise position
  
        temp_noise_pos.sort()
        noise_pos = np.append(noise_pos,temp_noise_pos) #record noise position
        mean[i] = data_all.mean()  # mean and std
        var[i] = data_all.std()
        mean[i] += 0
        var[i] *= 5
        #produce noise arrays
        t_error = np.random.normal(mean[i][0],var[i][0], int(noisePeriod/3)) 
        h_error = np.random.normal(mean[i][1],var[i][1], int(noisePeriod/3))
        v_error = np.random.normal(mean[i][2],var[i][2], int(noisePeriod/3))
        
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
if __name__ == "__main__":
    #a = pd.DataFrame([[1,2,3],[4,5,6]],columns=['a','b','c'])
    pass