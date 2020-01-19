#引用库函数
import numpy as np
import matplotlib.pyplot as plt #引入绘图库
import pandas as pd  
# import warnings
# warnings.filterwarnings("ignore")  #忽略版本报错
#intel 实验室


if __name__ == "__main__":
    np.random.seed(0)
    titlesize = 20
    xylabelsize = 10
    sticksize = 8
    textsize = 8
    Filled_DF_Type1 = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    
    Filled_DF_Type2 = ['index','detectedIndex']
    '''
    datefile1 = 'picture/node43.csv'
    datefile2 = 'picture/0.csv'
    saveName = 'picture/1.jpg'
    '''
    '''
    datefile1 = 'picture/node44.csv'
    datefile2 = 'picture/1.csv'
    saveName = 'picture/2.jpg'
    '''
    
    datefile1 = 'picture/node45.csv'
    datefile2 = 'picture/2.csv'
    saveName = 'picture/3.jpg'
    
    try:
        data1 = pd.read_csv(datefile1,names = Filled_DF_Type1,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    data1 = data1[0:10000].copy()
    print(data1.head())
    try:
        data2 = pd.read_csv(datefile2,names = Filled_DF_Type2,skiprows = 1,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')
    print(data2.head())
    detected_pos = data2['detectedIndex'].tolist()
    #数据为温度值0-2000

    legend_font = {"family":"Times New Roman"} #使用新罗马字体
    # plt.plot(x,data1[:,0][0:2000])
    #画三个节点及异常值的坐标图

    plt.figure(figsize =(10,8))
   
    #temperature
    #plt.subplot(1, 3, 1)
    #
    plt.subplot(3, 1, 1)
    data1['Temperature'].plot(c='k')
    for index in detected_pos:
        plt.scatter(index,data1.loc[index]['Temperature'],marker='o',c='r')
    plt.ylabel('Temperature (℃)',fontsize = xylabelsize)
    plt.yticks(fontsize=sticksize)
    plt.xticks(fontsize=sticksize)
    plt.legend(loc = 'best',prop = legend_font)   
    # plt.legend(loc = 'best',fontsize='12')
    #plt.xlabel('# of samples',fontsize = xylabelsize)



    plt.subplot(3, 1, 2)
    data1['Humidity'].plot(c='k')
    for index in detected_pos:
        plt.scatter(index,data1.loc[index]['Humidity'],marker='o',c='r')
    plt.ylabel('Humidity (%)',fontsize = xylabelsize)
    plt.yticks(fontsize=sticksize)
    plt.xticks(fontsize=sticksize)
    plt.legend(loc = 'best',prop = legend_font)   
    #plt.xlabel('# of samples',fontsize = xylabelsize)
    # plt.legend(loc = 'best',fontsize='12')

    #noise
    #plt.subplot(1, 3, 3)
    plt.subplot(3, 1, 3)
    data1['Voltage'].plot(c='k')
    for index in detected_pos:
        plt.scatter(index,data1.loc[index]['Voltage'],marker='o',c='r')
    plt.ylabel('Voltage (v)',fontsize = xylabelsize)
    plt.yticks(fontsize=sticksize)
    plt.xticks(fontsize=sticksize)
    # plt.legend(loc = 'best',fontsize='12')
    plt.xlabel('# of samples',fontsize = xylabelsize)


    plt.legend(loc = 'best',prop = legend_font)           
    

    plt.subplots_adjust(bottom=0.1, top=0.95, left=.1, right=.95)
    plt.savefig(saveName,dpi=500)
    plt.show()


