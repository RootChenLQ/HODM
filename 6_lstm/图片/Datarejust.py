import numpy as np
from scipy.stats import wasserstein_distance

'''
divi  函数 分解 给预测样本和真实样本分组 以一个时间窗口分成batch_size个子序列
'''
def data_divi(actual_target, predicted_target, W):
    real_value=actual_target
    pre_value=predicted_target
    batch_size = int(len(real_value + 1) / W)
    pre_value_batch = np.zeros([batch_size,W])
    real_value_batch = np.zeros([batch_size,W])

    for i in range(0,batch_size):
        real_value_batch[i,:] = real_value[i*W:i*W+W]
        pre_value_batch[i,:] = pre_value[i*W:i*W+W]

    return real_value_batch,pre_value_batch,batch_size
'''
与data_divi 区别在于 divi：是用一个固定的窗口，每次都进入新的W个样本
divi_oneW   ： 每次进来一个样本，始终保持 W 个数据在窗口内
'''
def data_divi_oneW(actual_target,predicted_target,W):
    real_value = actual_target
    pre_value = predicted_target
    batch_size = (real_value.shape[0]+1)-(W-1)
    pre_value_batch = np.zeros([batch_size,W])
    real_value_batch = np.zeros([batch_size,W])

    for i in range(W,batch_size):
        real_value_batch[i,:] = real_value[i-W:i]    #这里缺少了W个数据，因为我们是从W开始保存
        pre_value_batch[i,:] = pre_value[i-W:i]


    return  real_value_batch,pre_value_batch,batch_size



'''
计算每个子序列中的直方图频率，采用字典的方式， 然后调用wasserstein_distance函数
'''
def W_distence(realvalue,prevalue,bath_size):
    #计算直方图频率  采用字典的方式
 w=14
 EMD=np.zeros([bath_size,1])
 realvalue=np.around(realvalue,2)
 prevalue=np.around(prevalue,2)
#求出一个时间窗口内不相同的样本和其出现的次数
 for epch in range(bath_size):

   re_onelist_data= realvalue[epch, 0:w]
   re_onelist_data=list(re_onelist_data)
   re_unique_data = np.unique(realvalue[epch, :])
   re_unique_data_fre=[]
   for i in re_unique_data:
       re_unique_data_fre.append(re_onelist_data.count(i))  #真实样本的出现的次数
   # 将求出样本出现的频率
   for j in range(len(re_unique_data_fre)):
           re_unique_data_fre[j] = re_unique_data_fre[j] / w  #时间窗口内真实样本的出现的频率
   # print("真实样本",re_unique_data)
   # print("真实样本出现的频率:",re_unique_data_fre)

   pre_onelist_data = prevalue[epch,0:w]
   pre_onelist_data = list(pre_onelist_data)
   pre_unique_data = np.unique(prevalue[epch,:])
   pre_unique_data_fre = []
   for i in pre_unique_data:
       pre_unique_data_fre.append(pre_onelist_data.count(i))
    #将求出样本出现的频率
   for jj in range(len(pre_unique_data_fre)):
       pre_unique_data_fre[jj]=pre_unique_data_fre[jj]/w
   # print("预测样本",pre_unique_data)
   # print("预测样本出现的频率:",pre_unique_data_fre)
   EMD[epch,:] = wasserstein_distance(re_unique_data, pre_unique_data, re_unique_data_fre, pre_unique_data_fre)
 return EMD

'''
下面的代码将进行滑动窗口的实现
EDM将是一个时序变化过程。
                         窗口的大小W=14, 滑动步长W_step = 1                
'''
def W_distence_W_step(realvalue,prevalue,bath_size):
    #计算直方图频率  采用字典的方式
 w=14
    #需要调整数组大小  等会计算。。
 EMD=np.zeros([bath_size,3])
 realvalue=np.around(realvalue,2)
 prevalue=np.around(prevalue,2)
#求出一个时间窗口内不相同的样本和其出现的次数
 for epch in range(w,(bath_size-1)):

   re_onelist_data= realvalue[epch, 0:w]
   re_onelist_data=list(re_onelist_data)
   re_unique_data = np.unique(realvalue[epch, :])
   re_unique_data_fre=[]
   for i in re_unique_data:
       re_unique_data_fre.append(re_onelist_data.count(i))
   # 将求出样本出现的频率
   for j in range(len(re_unique_data_fre)):
           re_unique_data_fre[j] = re_unique_data_fre[j] / w
   # print("真实样本",re_unique_data)
   # print("真实样本出现的频率:",re_unique_data_fre)

   pre_onelist_data = prevalue[epch,0:w]
   pre_onelist_data = list(pre_onelist_data)
   pre_unique_data = np.unique(prevalue[epch,:])
   pre_unique_data_fre = []
   for i in pre_unique_data:
       pre_unique_data_fre.append(pre_onelist_data.count(i))
    #将求出样本出现的频率
   for jj in range(len(pre_unique_data_fre)):
       pre_unique_data_fre[jj]=pre_unique_data_fre[jj]/w
   # print("预测样本",pre_unique_data)
   # print("预测样本出现的频率:",pre_unique_data_fre)
   # EMD[epch,:] = wasserstein_distance(re_unique_data, pre_unique_data, re_unique_data_fre, pre_unique_data_fre)
   EMD[epch, :] = wasserstein_distance(re_unique_data, pre_unique_data, re_unique_data_fre, pre_unique_data_fre)
 return EMD











