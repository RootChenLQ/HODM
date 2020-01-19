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
    xylabelsize = 20
    sticksize = 16
    textsize = 14

    Filled_DF_Type = ['Temperature','Humidity','Voltage'] # intel 属性为此三者，sensorscope不同，为方便代码改修，直接异常标记
    datefile1 = 'result.csv'

    try:
        data1 = pd.read_csv(datefile1,names = Filled_DF_Type,sep=',')
    except IOError:
        print('文件读取异常！,请输入正确的文件名，或者查看文件目录是否正确')



    #数据为温度值0-2000

    legend_font = {"family":"Times New Roman"} #使用新罗马字体
    # plt.plot(x,data1[:,0][0:2000])
    #画三个节点及异常值的坐标图
    plt.figure(figsize =(18,13))

    #short
    data2 = data1['Temperature'][0:6000].copy()

    #plt.subplot(1, 3, 1)
    #
    plt.subplot(2, 2, 1)
    data2[1000]=50
    data2[1400]=100
    x = np.arange(6000)
    plt.plot(x,data2[0:6000],c='k')
    #plt.scatter(1000,50,c='r',s=100,marker='o');
    #plt.scatter(1400,100,c='r',s=100,marker='o');
    #plt.xlabel('# of Samples',fontsize = 16)
    plt.ylabel('Sensor Reading',fontsize = xylabelsize)
    plt.yticks(fontsize=sticksize)
    plt.xticks(fontsize=sticksize)
    # plt.legend(loc = 'best',fontsize='12')
    plt.xlabel('(a) Outlier Faults',fontsize = xylabelsize)

    #constant
    #plt.subplot(1, 3, 2)
    plt.subplot(2, 2, 2)
    data3 = data1['Temperature'][0:6000].copy()
    for i in range(2000,2600):
        data3[i]=50
        #plt.scatter(i,50,c='r');
        
    for i in range(4000,4600):
        data3[i]=50
        #plt.scatter(i,50,c='r');

    x = np.arange(6000)
    plt.plot(x,data3[0:6000],c='k')
    #plt.xlabel('# of Samples',fontsize = xylabelsize)
    plt.ylabel('Sensor Reading',fontsize = xylabelsize)
    plt.yticks(fontsize=sticksize)
    plt.xticks(fontsize=sticksize)
    plt.xlabel('(b) Stuck-at Faults',fontsize = xylabelsize)
    # plt.legend(loc = 'best',fontsize='12')

    #noise
    #plt.subplot(1, 3, 3)
    plt.subplot(2, 2, 3)
    data4 = data1['Voltage'][0:200].copy()
    for i in range(0,200):
        if i%7 ==0:
            data4[i]+= np.random.randn()*0.1
            #plt.scatter(i,data4[i],c='r',marker='o');
    x = np.arange(200)
    plt.plot(x,data4[0:200],c='k')
    #plt.xlabel('# of Samples',fontsize = xylabelsize)
    plt.ylabel('Sensor Reading',fontsize = xylabelsize)
    plt.yticks(fontsize=sticksize)
    plt.xticks(fontsize=sticksize)
    # plt.legend(loc = 'best',fontsize='12')
    plt.xlabel('(c) Noisy Faults',fontsize = xylabelsize)

    plt.subplot(2, 2, 4)
    #titlesize = 16
    #xylabelsize = 16 
    xrange = 2788
    x = range(2788)
    changed_pos = 500
    insert_pos = 1500
    #plt.figure(figsize =(10,8))
    #温度 
    # plt.subplot(1, 3, 1)
    # plt.title('Temperature',fontsize = titlesize)
    plt.plot(x,data1['Temperature'][0:xrange],c='k',label = 'normal data')


    plt.scatter(changed_pos,data1['Temperature'][changed_pos],c='',s=80,marker='o',edgecolors='g',label = 'changed data')

    # plt.text(200, data1[500,0]-0.3, r'$\sum_{i=0}^\infty x_i$', fontsize=20)
    plt.text(changed_pos-280, data1['Temperature'][changed_pos]-0.4,'(%d,%.2f)'%(changed_pos,data1['Temperature'][changed_pos]),ha='center', va='bottom', fontsize=textsize)  

    plt.scatter(insert_pos,data1['Temperature'][insert_pos],c='b',s=80,marker='o',label = 'original data')
    plt.text(insert_pos, data1['Temperature'][insert_pos]+0.1, '(%d,%.2f)'%(insert_pos,data1['Temperature'][insert_pos]),ha='center', va='bottom', fontsize=textsize)  

    plt.scatter(changed_pos,data1['Temperature'][insert_pos],c='r',s=80,marker='*',label = 'inserted data')
    plt.text(changed_pos, data1['Temperature'][insert_pos]+0.1, '(%d,%.2f)'%(changed_pos,data1['Temperature'][insert_pos]),ha='center', va='bottom', fontsize=textsize)  

    # plt.plot(x,data1[:,0][0:xrange],label = 'node18')
    # plt.plot(x,data2[:,0][0:xrange],label = 'node21')
    # plt.plot(x,data3[:,0][0:xrange],label = 'node23')
    plt.ylabel('Sensor Reading',fontsize = xylabelsize)
    plt.xlabel('(d) Contextual Fault',fontsize = xylabelsize)



    plt.plot([changed_pos,changed_pos],[data1['Temperature'][changed_pos],data1['Temperature'][insert_pos]],color='black', linestyle='-.')
    plt.text(changed_pos+150, 20, '∆T=%.2f'%(data1['Temperature'][insert_pos]-data1['Temperature'][changed_pos]),ha='center', va='bottom', fontsize=textsize)  
    plt.yticks(fontsize=sticksize)
    plt.xticks(fontsize=sticksize)
    #plt.legend(loc = 'best',fontsize='12')
    plt.legend(loc = 'best',prop = legend_font,numpoints=1)           
    plt.savefig('Various types of sensor faults191211.jpg',dpi=500)


    plt.subplots_adjust(bottom=.01, top=.99, left=.01, right=.99)
    plt.show()


