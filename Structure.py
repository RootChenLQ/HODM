#coding:utf-8
import pandas as pd
#print('import Detection_Methods_Short.py')
#数据加载标记 类型
'''
pd_type =  ['TimeStamp','ID','Fluo',\
                    'Temperature1','Temperature2','Temperature3','Temperature4','Temperature5','Temperature6',\
                    'Light','WindDirection','WindSpeed']
#大气气象站 数据类型
meoto_pd_type =  ['ID','Year','Month','Day','Hour','Minute','Second',\
                   'TimeStamp','Sequence Number','Temperature',\
                    'Surface Temperature','Light','Humidity',\
                    'Soil Moisture','Watermark','Rain Meter','Wind Speed',\
                     'Wind Direction']
'''
intel_lab_type = ['Date','Time','Epoch','ID','Temperature','Humidity','Light','Voltage']
dimensions = 3
#sample_size = 300
point_size = 300  # 求H时使用
prop_threshold = 1


#定义子节点中，数据包发送给父节点的结构：子节点汇总数据特征，发送给簇头节点   51bytes
# dataframe
'''
'LLS', linear sum （T、H、L）
'LLSS', linear sum of square（T、H、L）
'TemperatureMax',  
'TemperatureMin',
'HumidityMax',
'HumidityMin',
'LightMax',
'LightMin',
'LN'  linear num（T、H、L）
'''
MN_configure_type = ['LLS_T','LLS_H','LLS_V',
                     'LLSS_T','LLSS_H','LLSS_V',
                     'TemperatureMax','TemperatureMin',
                     'HumidityMax','HumidityMin',
                     'VoltageMax','VoltageMin',
                     'LN_T','LN_H','LN_V']
MN_selfconfigure_DF = pd.DataFrame(columns = MN_configure_type)
MN_selfconfigure_DF['LLS_T'] = MN_selfconfigure_DF['LLS_T'].astype('float32')
MN_selfconfigure_DF['LLS_H'] = MN_selfconfigure_DF['LLS_H'].astype('float32')
MN_selfconfigure_DF['LLS_V'] = MN_selfconfigure_DF['LLS_V'].astype('float32')
##
MN_selfconfigure_DF['LLSS_T'] = MN_selfconfigure_DF['LLSS_T'].astype('float32')
MN_selfconfigure_DF['LLSS_H'] = MN_selfconfigure_DF['LLSS_H'].astype('float32')
MN_selfconfigure_DF['LLSS_V'] = MN_selfconfigure_DF['LLSS_V'].astype('float32')
##
MN_selfconfigure_DF['TemperatureMax'] = MN_selfconfigure_DF['TemperatureMax'].astype('float32')
MN_selfconfigure_DF['TemperatureMin'] = MN_selfconfigure_DF['TemperatureMin'].astype('float32')
MN_selfconfigure_DF['HumidityMax'] = MN_selfconfigure_DF['HumidityMax'].astype('float32')
MN_selfconfigure_DF['HumidityMin'] = MN_selfconfigure_DF['HumidityMin'].astype('float32')
MN_selfconfigure_DF['VoltageMax'] = MN_selfconfigure_DF['VoltageMax'].astype('float32')
MN_selfconfigure_DF['VoltageMin'] = MN_selfconfigure_DF['VoltageMin'].astype('float32')
##
MN_selfconfigure_DF['LN_T'] = MN_selfconfigure_DF['LN_T'].astype('uint8')
MN_selfconfigure_DF['LN_H'] = MN_selfconfigure_DF['LN_H'].astype('uint8')
MN_selfconfigure_DF['LN_V'] = MN_selfconfigure_DF['LN_V'].astype('uint8')


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
#定义簇头节点中，数据包发送给簇内节点的结构
CH_configure_type = ['NT_g','NH_g','NV_g',
                     'mean_T','mean_H','mean_V',  #内存计算需要
                     'std_T','std_H','std_V',     #内存计算需要
#                      'TemperatureMaxNorm_g','TemperatureMinNorm_g',#计算vol uint16
#                      'HumidityMaxNorm_g','HumidityMinNorm_g',
#                     'VoltageMaxNorm_g','VoltageMinNorm_g',
                    #'N',
                     #'H_MIN',
                     'H_MAX',
                     'Vol','B','C']
                     #,'D']  #HyperGrid网络信息
CH_configure_DF = pd.DataFrame(columns = CH_configure_type) 
CH_configure_DF['NT_g'] = CH_configure_DF['NT_g'].astype('uint16')
CH_configure_DF['NH_g'] = CH_configure_DF['NH_g'].astype('uint16')
CH_configure_DF['NV_g'] = CH_configure_DF['NV_g'].astype('uint16')
##
CH_configure_DF['mean_T'] = CH_configure_DF['mean_T'].astype('float32')
CH_configure_DF['mean_H'] = CH_configure_DF['mean_H'].astype('float32')
CH_configure_DF['mean_V'] = CH_configure_DF['mean_V'].astype('float32')
##
CH_configure_DF['std_T'] = CH_configure_DF['std_T'].astype('float32')
CH_configure_DF['std_H'] = CH_configure_DF['std_H'].astype('float32')
CH_configure_DF['std_V'] = CH_configure_DF['std_V'].astype('float32')
##

# CH_configure_DF['TemperatureMaxNorm_g'] = CH_configure_DF['TemperatureMaxNorm_g'].astype('float32')
# CH_configure_DF['TemperatureMinNorm_g'] = CH_configure_DF['TemperatureMinNorm_g'].astype('float32')
# CH_configure_DF['HumidityMaxNorm_g'] = CH_configure_DF['HumidityMaxNorm_g'].astype('float32')
# CH_configure_DF['HumidityMinNorm_g'] = CH_configure_DF['HumidityMinNorm_g'].astype('float32')
# CH_configure_DF['VoltageMaxNorm_g'] = CH_configure_DF['VoltageMaxNorm_g'].astype('float32')
# CH_configure_DF['VoltageMinNorm_g'] = CH_configure_DF['VoltageMinNorm_g'].astype('float32')
#HyperGrid结构
# CH_configure_DF['H_MIN'] = CH_configure_DF['H_MIN'].astype('float32')
CH_configure_DF['H_MAX'] = CH_configure_DF['H_MAX'].astype('float32')
CH_configure_DF['Vol'] = CH_configure_DF['Vol'].astype('uint8')
CH_configure_DF['B'] = CH_configure_DF['B'].astype('uint8')
CH_configure_DF['C'] = CH_configure_DF['C'].astype('uint8')
# CH_configure_DF['D'] = CH_configure_DF['D'].astype('float32')

#print(CH_configure_DF.dtypes)

#储存pos 和num 信息
pos_configure_type = ['pos','num']
# 本地数据位置信息、及数量信息   MN -> CH   #K* 额外附加
SUM_configure_DF = pd.DataFrame(columns = pos_configure_type) 
SUM_configure_DF['pos'] = SUM_configure_DF['pos'].astype('uint32')
SUM_configure_DF['num'] = SUM_configure_DF['num'].astype('uint8')
# 簇内数据位置信息、及数量信息   CH -> MN   #K 额外附加
NP_configure_DF = pd.DataFrame(columns = pos_configure_type)
NP_configure_DF['pos'] = NP_configure_DF['pos'].astype('uint32')
NP_configure_DF['num'] = NP_configure_DF['num'].astype('uint8')

#intel 数据集存储结构
#Filled_DF_Type = ['Temperature','Humidity','Light','Voltage']
Filled_DF_Type = ['Temperature','Humidity','Voltage']
QueueBuffer_DF = pd.DataFrame(columns = Filled_DF_Type) 
QueueBuffer_DF['Temperature'] = QueueBuffer_DF['Temperature'].astype('float32')
QueueBuffer_DF['Humidity'] = QueueBuffer_DF['Humidity'].astype('float32')
#QueueBuffer_DF['Light'] = QueueBuffer_DF['Light'].astype('float32')
QueueBuffer_DF['Voltage'] = QueueBuffer_DF['Voltage'].astype('float32')



#output structure
Output_DF_Type = ['Exp','ID','anomalyName','anomalyType','TN','FN','FP','TP','ACC','FPR','TPR','P','F1','Update_times','runtime']
Output_DF = pd.DataFrame(columns = Output_DF_Type) 