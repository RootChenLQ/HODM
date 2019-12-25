#coding:utf-8
#clq
import time
import numpy as np
#from sklearn.cluster import KMeans

'''
def timer(func):
    def wrapper(*args,**kwds):
        start = time.time()
        func(*args,**kwds)
        stop = time.time()
        print('消耗时间 %0.3f'%(stop-start))
    return wrapper

@timer
'''


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
    size = len(temp)
    ignore = int(size*0.25)
    #print(temp[3:-3])
    robustarr = temp[ignore:-ignore]
    return robustarr.mean(),robustarr.std()

def getLocal_label(history_vec):  #new data is insert the end of history_vec1
    #direct_vec = np.array([1,0,0])
    #test data
    cosval = np.array([])
    distance = np.array([])
    # distance between 
    for i in range(history_vec.shape[0]):
        cosval = np.append(cosval,getCos_val(history_vec[i]))
        distance = np.append(distance,getDistance(history_vec[i]))
    #cosVal = np.array([])  # load cosval for juding
    mean_d, std_d = getRobustMeanStd(distance)
    print(distance)
    #mean_d = distance.mean()
    #std_d = distance.std()
    temp = abs(distance - mean_d)
    result1 = [1 if item>2*std_d else 0 for item in temp]   # list operation +三目运算
    print(result1)
    ones = 0
    zeros = 0
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



    mean_cos = cosval.mean()   # 
    print(mean_cos)
    std_cos = cosval.std()
    #print('std',std)
    print(cosval)
    temp = abs(cosval - mean_cos)
    #print('temp',temp)
    result2 = [1 if item>2*std_cos else 0 for item in temp]   # list operation +三目运算
    print(result2)
    #counts zeros and ones 
    ones = 0
    zeros = 0
    for i in range(len(result2)):
        if result2[i]==0:
            zeros+=1
        else:
            ones+=1
    if zeros != 0:
        if ones/zeros > 1:
            print('exiting noise....')
        else:
            print('noise clear')
    #if result[-1] == 1:
    return result1,result2


if __name__ == "__main__":
    #vec = np.ones(3)
    #print(vec)
    '''
    a = np.array([1,1,1])
    for i in range(1,10):
        a = np.vstack((a,np.array([i%2+1,i%3+1,i+1])))
    '''
    a = np.random.randint(8,10,(9,3))
    for i in range(3):
        a = np.vstack((a,np.random.randint(1,100,(1,3))))
    #a = np.vstack((a,np.random.randint(1,10,(1,3))))
    for i in range(a.shape[0]):
        print(a[i])
        #cosVal = np.append(cosVal,getCos_val(a[i],vec))
    label = getLocal_label(a)
    print(label)