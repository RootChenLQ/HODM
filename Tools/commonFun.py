#coding:utf-8
import numpy as np 
import pandas as pd 
def sort_byNum(arr1,arr2):
    #use for array ascending =false
    # return df_
    type_ = ['pos','num']
    np_arr1 = np.array(arr1)
    np_arr2 = np.array(arr2)
    np_arr1 = np.vstack((np_arr1,np_arr2))
    np_arr1 = np_arr1.T
    #print(np_arr1)
    df_ = pd.DataFrame(np_arr1,columns=type_)
    df_ = df_.sort_values(['num'], ascending=[False])
    df_ = df_.reset_index(drop=True)
    #print(df_)
    return df_
def encoderPos(arr,bits):
    # arr = [x1,x2,x3,x4,...]
    #output x1 x2 x3 x4
    pos = 0
    for i in range(len(arr)):
        pos = arr[i]<<((len(arr)-1-i)*bits) | pos
    return pos
def decoderPos(pos,attributes,bits):
    #input x1 x2 x3 ...
    #ouput[x1,x2,x3,x4]
    arr = np.zeros(attributes)
    bits =(int)(bits)
    for i in range(0,attributes):
        divid = 1<<bits
        arr[i] = pos%divid
        pos = pos//divid
    arr = arr[::-1]
    
    
    return arr
def getChebyshevDistance(pos1,pos2,attributes,b): 
    pos_array1 = decoderPos(pos1,attributes,b)
    pos_array2 = decoderPos(pos2,attributes,b)
    ChebyshevDistance = abs(pos_array1[0]-pos_array2[0])
    for i in range(1,attributes):
        #pos_array1.append(pos1>>b)
        #pos1 = 
        temp = abs(pos_array1[i]-pos_array2[i])
        if temp > ChebyshevDistance:
            ChebyshevDistance = temp
    # pos1 = [x1,y1,z1] pos2 = [x2,y2,z2]
    return ChebyshevDistance
def getMaxChebyshevDistance_index(pos_arr,pos,attributes,b):
    # get the  index of max MaxChebyshevDistance_
    maxDis = 0
    maxIndex = 0
    for i in range(len(pos_arr)):
        dis = getChebyshevDistance(pos_arr[i],pos,attributes,b)
        if dis > maxDis :
            dis = maxDis
            maxIndex = i
    print('max_index',maxIndex)
    print('max_pos',pos_arr[maxIndex])
    return maxIndex

if __name__ == "__main__":
    '''
    arr1 = [8,9,10]
    arr2 = [4,11,6]
    sort_byNum(arr1,arr2)
    #decoderPos(33,2,4)
    dis = getChebyshevDistance(256+53,512+17,3,4)
    print(dis)
    pos_arr = []
    '''
    arr = [[1,2,3],[1,3,3],[4,7,1],[1,1,10]]
    pos = encoderPos([1,1,1],4)
    pos1 = encoderPos(arr[0],4)
    pos2 = encoderPos(arr[1],4)
    pos3 = encoderPos(arr[2],4)
    pos4 = encoderPos(arr[3],4)
    print(pos1)
    #index = decoderPos([pos],3,4)
    #print(index)
    pos_arr = [pos1,pos2,pos3,pos4]
    index = getMaxChebyshevDistance_index(pos_arr,pos,3,4)