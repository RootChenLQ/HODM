#coding:utf-8
import pandas as pd 
import numpy as np 

if __name__ == "__main__":
    datefile1 = 'datasets/shuttle_train.csv'
    #Filled_DF_Type = ['time','Rad_Flow','Fpv_Close','Fpv_Open','High','Bypass','Bpv_Close','Bpv_Open','9','Label']
    Filled_DF_Type = ['1','2','3','4','5','6','7','8','9','label']
    try:
        data1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确') 
    print(data1)
    print(data1['label'].unique())