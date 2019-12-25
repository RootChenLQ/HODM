#coding:utf-8
import pandas as pd 



def run_model():
    print("Hello world")
    pass


if __name__ == "__main__":
    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    #data
    datefile1 = 'datasets/node43op.csv'
    datefile2 = 'datasets/node44op.csv'
    datefile3 = 'datasets/node45op.csv'
    try:
        data1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    try:
        data2 = pd.read_csv(datefile2,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    try:
        data3 = pd.read_csv(datefile3,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    data1 
    data2
    data3
