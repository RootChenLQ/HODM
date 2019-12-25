#引用库函数
import numpy as np
import matplotlib.pyplot as plt #引入绘图库
import pandas as pd  
# import warnings
# warnings.filterwarnings("ignore")  #忽略版本报错
#intel datasets
#补全数据集 缺失数据等于相邻值的平均
def fill_data(df):
    rows = 0
    filled_t = []
    filled_h = []
    filled_l = []
    filled_v = []
    epochs = []
    Epoch = 2
    Temperature = 4
    Humidity = 5
    Light = 6
    Voltage = 7
    #ini_epoch_ = df.at[0,'Epoch']
    #end_epoch_ = df.at[len(df)-1,'Epoch']
    epoch_ = df.iat[0,Epoch]
    while rows < len(df):
#         print(rows)
        if epoch_ > 7287:
            #print('test')
            pass
        if epoch_ > df.iat[rows,Epoch]:
            #if epoch_ - df.iat[rows,Epoch] < 10:
            pass   
            #rows+=1
            #exit error epoch
        elif epoch_ == df.iat[rows,Epoch]:
#             print('==')
            t = df.iat[rows,Temperature]
            h = df.iat[rows,Humidity]
            l = df.iat[rows,Light]
            v = df.iat[rows,Voltage]
            filled_t.append(t)
            filled_h.append(h)
            filled_l.append(l)
            filled_v.append(v)
            epochs.append(epoch_)
            epoch_+=1
           
        elif epoch_ < df.iat[rows,Epoch]:
#             print('<')
            if df.iat[rows,Epoch] - epoch_ < 30:
                while epoch_ <  df.iat[rows,Epoch]:
    #                 print(epoch_)
                    weight = df.iat[rows,Epoch] - epoch_
                    fill_t = (df.iat[rows,Temperature]+weight*filled_t[-1])/(1+weight)
                    fill_h = (df.iat[rows,Humidity]+weight*filled_h[-1])/(1+weight)
                    fill_l = (df.iat[rows,Light]+weight*filled_l[-1])/(1+weight)
                    fill_v = (df.iat[rows,Voltage]+weight*filled_v[-1])/(1+weight)
                    filled_t.append(fill_t)
                    filled_h.append(fill_h)

                    filled_l.append(fill_l)
                    filled_v.append(fill_v)
                    epochs.append(epoch_)
                    epoch_+=1
                #当epoch+1 = df.at[rows,'Epoch']
                #此时epoch 等于 df.at[rows,'Epoch']
                if epoch_ == df.iat[rows,Epoch]:
    #                 print('=')
                    t = df.iat[rows,Temperature]
                    h = df.iat[rows,Humidity]
                    l = df.iat[rows,Light]
                    v = df.iat[rows,Voltage]
                    filled_t.append(t)
                    filled_h.append(h)
                    filled_l.append(l)
                    filled_v.append(v)
                    '''
                    filled_t.append(df.iat[rows,Temperature])
                    filled_h.append(df.iat[rows,Humidity])
                    filled_l.append(df.iat[rows,Light])
                    filled_v.append(df.iat[rows,Voltage])'''
                    epochs.append(epoch_) 
                    
                    epoch_+=1
            else:
                pass
        rows+=1
    #返回温度、湿度、电压，构成的矩阵
    #################
    #将三个列表整合为一个矩阵
    #方法1 
    np_array0 = np.array(epochs)
    np_array1 = np.array(filled_t)
    np_array2 = np.array(filled_h)
    np_array3 = np.array(filled_l)
    np_array4 = np.array(filled_v)
    #数组按照垂直添加，
    np_array0 = np.vstack((np_array0,np_array1))   #水平添加的：使用np.hstack(a,b)
    np_array0 = np.vstack((np_array0,np_array2))
    np_array0 = np.vstack((np_array0,np_array3))
    np_array0 = np.vstack((np_array0,np_array4))
#     np_array1 = np.vstack((np_array1,np_array4))
    np_array0 = np_array0.T
#     print('np_array1',np_array1.shape)
#     print('epoch ranges:[%d,%d]'%(ini_epoch_,end_epoch_))
    return np_array0

if __name__ == "__main__":
    
    np.random.seed(0)
    titlesize = 20
    xylabelsize = 18 
    intel_lab_type = ['Date','Time','Epoch','ID','Temperature','Humidity','Light','Voltage']
    Epoch = 2
    Temperature = 4
    Humidity = 5
    Light = 6
    Voltage = 7

    #intel lab 解析的属性
#date:yyyy-mm-dd	time:hh:mm:ss.xxx	epoch:int	moteid:int	temperature:real	humidity:real	light:real	voltage:real

    datefile = 'datasets/intelLab_dataset_full.txt'
    try:
        ##读取txt 文件
        print('data read success')
        data = pd.read_csv(datefile,names = intel_lab_type,sep=' ')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')

    for i in range(1,55):
        ''' #original data divided by ID
        s = "node"+str(i)+"orig.csv"
        tempdata = data[data["ID"]==i].copy()
        tempdata.sort_values(by=['Date','Time'],inplace=True,ascending=[True,True])# sort data
        tempdata.to_csv(s)'''
       
        #insert data by
        s = "datasets/intel/node"+str(i)+"oped.csv"
        tempdata = data[data["ID"]==i].copy()
        tempdata.sort_values(by=['Date','Time'],inplace=True,ascending=[True,True])# sort data
        filled_np = fill_data(tempdata)
        outputda = pd.DataFrame(filled_np,columns=['Epoch','Temperature','Humidity','Light','Voltage'])
        outputda.to_csv(s)
    ''' 
    s = "node"+str(0)+"op.csv"
    tempdata = data[data["ID"]== 43].copy()
    tempdata.sort_values(by=['Date','Time'],inplace=True,ascending=[True,True])# sort data
    filled_np = fill_data(tempdata)
    outputda = pd.DataFrame(filled_np,columns=['Epoch','Temperature','Humidity','Light','Voltage'])
    outputda.to_csv(s)
     '''
    