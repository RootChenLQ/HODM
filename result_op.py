import pandas as pd 
import numpy as np 
import Structure
import os
def result_op(filename,outname):
    # operation group by 'ID','anomalyName','anomalyType'
    #求各个异常类型的平均性能
    try:
        dataframe = pd.read_csv(filename,names = Structure.Output_DF_Type,skiprows = 1,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    #print(dataframe.head())
    # 只留下需要处理的列
    #Output_DF_Type = ['Exp','ID','anomalyName','anomalyType',
    # 'TN','FN','FP','TP','ACC','FPR','TPR','P','F1','Update_times','runtime']

    cols = [col for col in dataframe.columns if col  in['TN','FN','FP','TP','ACC','FPR','TPR','P','F1']]
    # 分组的列
    gp_col = ['ID','anomalyName','anomalyType']
    #gp_col = ['anomalyName','anomalyType']
    # 根据分组计算平均值
    df_mean = dataframe.groupby(gp_col)[cols].mean()
    #print(df_mean.head())
    outname = 'results/refine/'+outname[0:-4] +'out.csv'
    df_mean.to_csv(outname)

def result_op2(filename,outname):
    ## operation group by 'ID','anomalyName'
    # 求异常的平均性能
    try:
        dataframe = pd.read_csv(filename,names = Structure.Output_DF_Type,skiprows = 1,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    #print(dataframe.head())
    # 只留下需要处理的列
    #Output_DF_Type = ['Exp','ID','anomalyName','anomalyType',
    # 'TN','FN','FP','TP','ACC','FPR','TPR','P','F1','Update_times','runtime']

    cols = [col for col in dataframe.columns if col  in['TN','FN','FP','TP','ACC','FPR','TPR','P','F1']]
    # 分组的列
    gp_col = ['ID','anomalyName']
    #gp_col = ['anomalyName','anomalyType']
    # 根据分组计算平均值
    df_mean = dataframe.groupby(gp_col)[cols].mean()
    #print(df_mean.head())
    outname = 'results/refine/'+outname[0:-4] +'draw_out.csv'
    df_mean.to_csv(outname)
if __name__ == "__main__":
    #result_op('results/LOFE0.csv','LOFE0')
    
    for info in os.listdir('/Users/rootchen/Desktop/PythonWorkspace/HODM/HODM/results'):
        '''
        print(info) # csv name
        if os.path.isdir(info):
            print('yes')
        '''

        #print(info)

      
        #info = 'LOFE0.csv'
        domain = os.path.abspath(r'/Users/rootchen/Desktop/PythonWorkspace/HODM/HODM/results') #获取文件夹的路径
        datafile = os.path.join(domain,info) #将路径与文件名结合起来就是每个文件的完整路径
        #data = pd.read_csv(info)
        if info[-4:] == '.csv':
            print(info)
            result_op(datafile,info)
        else:
            pass


        '''
        if os.path.isdir(datafile):
            print('ignore folder')
            pass
        elif os.path.isfile(datafile):
            result_op2(datafile,info)
            '''
        #print(data)
    