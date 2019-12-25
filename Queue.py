#coding:utf-8
# 先进先出
import pandas as pd
import numpy as np
import Structure
import Fun
import math
from Tools.InsertNoise import insert_noise_error,insert_outlier_error
from Tools.DrawFun import draw3D
# 获取两个向量的余弦值
def getCos_val(vec1):
    vec2 = np.array([1,0,0])
    assert(len(vec1)==len(vec2)),"two vectors must have same shape"
    val1 = 0.0
    val2 = 0.0
    val3 = 0.0
    for i in range(len(vec1)):
        val1 += vec1[i]*vec2[i]
        val2 += vec1[i]*vec1[i]
        val3 += vec2[i]*vec2[i]
    cosval_ = val1/((val2*val3)**0.5)
    return cosval_
#获取向量到远点的距离
def getDistance(vec):
    temp = 0
    for i in range(len(vec)):
        temp += vec[i]**2
    return np.sqrt(temp)

#获取去掉两端最大最小值的 均值和方差
def getRobustMeanStd(arr):
    temp = np.sort(arr)
    #size = len(temp)
    robustarr = temp
    #ignore = int(size*0.05)
    #print(temp[3:-3])
    #assert(ignore<size-ignore),"Array is to short"
    #robustarr = temp[ignore:-ignore]
    #robustarr = temp
    return robustarr.mean(),robustarr.std()

class MNQueue():
    def __init__(self,size):
        self.size = size
        #self.front = -1
        #self.rear = -1
        self.count = 0
        #QueueBuffer_DF结构
        #['index','DateTime','Epoch','ID','Temperature','Humidity','Voltage']
        self.df = Structure.QueueBuffer_DF.copy()
        
    def enqueue(self,series): # 3.14修改，
        is_success = False 
        if len(self.df) == self.size:
            self.dequeue()
            self.enqueue(series)  #  ,index)
        else:
            self.df = self.df.append(series,ignore_index=True,sort=False)   #df.append(timetable_df.iloc[[0]],ignore_index=True)
            is_success = True
            #self.rear = (self.rear+1)% self.size
        if is_success:
            self.count += 1
            
    def dequeue(self):
        if len(self.df) == 0:
            #raise exception("queue is empty")
            print('queue is empty')
        else:
            #print('dequeue ',self.queue[0])
            #self.queue.pop(0)
            #index 删除第一个索引
            #self.index.pop(0)
            #dataframe删除第一条记录
            self.df.drop(index=0,inplace=True)
            #重置index 
            self.df = self.df.reset_index(drop=True)
            #self.front = (self.front+1) % self.size
            #self.count -= 1
    def isfull(self):
        return self.count == self.size
    #def isempty(self):
    #    return self.count == 0
    def showQueue(self):
        #print(self.queue)
        #print(self.index)
        print(self.df)
    def clear_count(self):
        self.count = 0 
    def is_trigger_update(self):
#         print('self.count %d,self.size %d'%(self.count,self.size))
        #if self.count == self.size:
#             print('is_trigger_update')
            #pass
        return self.count  == self.size

class posQueue(): #recorded pos array and dis sita. reload until model update
    def __init__(self,size,attributes):
        self.size = size
        self.attributes = attributes
        self.count = 1  #预先存储一个数组
        self.pos_arr = np.array([0,0,0])
        self.params = np.array([0,0]) # store distance and cos sita
        self.init = True
    def enqueue(self,arr): # 3.14修改，
        if arr.size ==  self.attributes:
            if self.isfull():
                #print('dequeue')
                self.dequeue()
            # add new line 
            self.pos_arr = np.vstack((self.pos_arr,arr))
            cosval = getCos_val(arr)
            dis = getDistance(arr)
            self.params = np.vstack((self.params,np.array([dis,cosval])))
            self.count += 1
            if(self.count>1 and self.init): #delete the inital [0,0,0]
                self.dequeue()
                self.init = False

    def dequeue(self):
        if self.isempty():
            #raise exception("queue is empty")
            print('queue is empty')
        else:
            self.pos_arr = np.delete(self.pos_arr,0,axis=0) #delete first line
            self.params = np.delete(self.params,0,axis=0)
            #self.count -= 1

    def isfull(self):
        return self.count == self.size

    def isempty(self):
        return self.count == 0

    def showQueue(self):
        #print(self.queue)
        #print(self.index)
        for i in range(self.count):
            print(self.pos_arr[i])
    def clear_count(self):
        self.count = 0 
        self.pos_arr = np.array([0,0,0])
        self.params = np.array([0,0]) # store distance and cos sita
        self.init = True
    def check_pos(self,pos):
        #get distance and sita 
        thres = 3
        testDis = getDistance(pos)
        testSita = getCos_val(pos)
        tempParams = np.copy(self.params)
        tempParams = np.vstack((tempParams,np.array([testDis,testSita])))
   
        #mean_d, std_d = getRobustMeanStd(self.params[:,0])  #不插入异常
        mean_d, std_d = getRobustMeanStd(tempParams[:,0])
        mean_sita, std_sita = getRobustMeanStd(tempParams[:,1]) #插入测试数据
        temp1 = abs(testDis - mean_d)
        result1 = True if temp1>thres*std_d else False   # list operation +三目运算
        '''
        if(std_d>0.1):
            result1 = True if temp1>thres*std_d else False   # list operation +三目运算
        else:
            result1 = False
        '''
        #print(result1)
        #noise analyze
        '''
        for i in range(len(result1)):
            if result1[i] == 0:
                zeros+=1
            else:
                ones+=1
        if zeros != 0:
            if ones/zeros > 1:
                print('exiting noise....')
            else:
                print('noise clear')
        '''

        #mean_sita, std_sita = getRobustMeanStd(self.params[:,1]) 不插入测试数据
        temp2 = abs(testSita - mean_sita)
        result2 = True if temp2>thres*std_sita else False   # list operation +三目运算
        '''
        if(std_sita > 0.1):
            result2 = True if temp2>thres*std_sita else False   # list operation +三目运算
        else:
            result2 = False
        '''
        #print(result2)
        '''
        if(result1 or result2):
            print('mean',mean_d,mean_sita)
            print('std',std_d,std_sita)
            print('real',testDis,testSita)'''
        return result1 or result2
    '''
    def is_trigger_update(self):
#         print('self.count %d,self.size %d'%(self.count,self.size))
        if self.count == self.size:
#             print('is_trigger_update')
            pass
        return self.count  == self.size
    '''
if __name__ == "__main__":
    '''
    from queue import Queue
    q = Queue(10)
    for i in range(100):
        if q.full():
            q.get()
        q.put(i)
    '''
    '''
    posqueue = posQueue(10,3)
    a = np.random.randint(8,10,(9,3))
    
    #a = np.vstack((a,np.random.randint(1,10,(1,3))))

        #cosVal = np.append(cosVal,getCos_val(a[i],vec))
    for i in range(len(a)):
        print(a[i])
        posqueue.enqueue(a[i])
    temp = np.random.randint(1,100,(1,3))
    print(temp)
    #posqueue.enqueue(temp[0])
    posqueue.check_pos(temp[0])
    
    '''
    from mpl_toolkits import mplot3d
    # matplotlib inline
    import matplotlib.pyplot as plt
    #获取数据
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datefile1 = 'datasets/node43op.csv'
    datefile2 = 'datasets/node44op.csv'
    datefile3 = 'datasets/node45op.csv'
    try:
        ##读取txt 文件
        data1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    queuelen = 40
    train_len = 100
    
    #插入异常
    data1 = data1[0:1000].copy()
    #data1,noise_pos,noise_data = insert_noise_error(data1,300,3)
    #data1,noise_pos = insert_outlier_error(data1,10)
    #label = np.zeros(len(data1))
    #for index in noise_pos:
    #    label[index] = 1
    #draw3D(data1,label)
    #print(noise_pos)
    #data1['Temperature'].plot()
    #data1.loc[500,'Temperature'] = 10
    
    mean = data1[0:train_len].mean()
    mean = mean.values
    print(mean)
    std = data1[0:train_len].std()
    std = std.values
    print(std)
    H_l,H_h = Fun.get_HG_H(3,train_len)
    queue1 = posQueue(queuelen,3)

    '''
    for i in range(train_len-queuelen,train_len):
        #print('mean',mean)
        #print('std',std)
        series = data1.iloc[[i]]
        #print(data1.iloc[[i]])   #self.LoadDF.iloc[[self.datapoint]]
        temp = series.values[0]
        #print(temp.shape)
        #print(mean.shape)
        temp2 = ((temp - mean)/std + 10) / H_h
        print(temp2)
       
       
        temp3 = np.floor(temp2)
        print(temp3)
        queue1.enqueue(temp3)
    '''


    count = 0
    for i in range(train_len,len(data1)):
        #normalize
        if(i%train_len == 0):
            print(i)
            #relean the params
            print('Update\n')
            mean = data1[i-train_len:i].mean()
            mean = mean.values
            print(mean)
            std = data1[i-train_len:i].std()
            std = std.values
        #re compute the stored pos
            queue1.clear_count()
            for j in range(i-queuelen,i):
                series = data1.iloc[[j]]
                temp = series.values[0]
                temp = ((temp - mean)/std + 10) / H_h
                pos = np.floor(temp)
                queue1.enqueue(pos)
        series = data1.iloc[[i]]
        temp = series.values[0]
        temp = ((temp - mean)/std + 10) / H_h
        pos = np.floor(temp)
        #enqueue
        result = queue1.check_pos(pos)
        if result:
            print('error:',data1.iloc[[i]])
            count+=1
        queue1.enqueue(pos)
       

    print(count)
