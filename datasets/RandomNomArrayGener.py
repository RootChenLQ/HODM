#coding:UTF-8
import numpy as np 
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
if __name__ == "__main__":
    mean = [20,40,5]
    std = [1,1,1]
    noise_mean = [30,50,10]
    noise_std = [2,2,2]
    data_all = np.array([0,0,0,0])
    datasize = 50000
    begin = 10000
    insert_list = np.random.choice(np.arange(0,datasize),int(datasize*0.01))
    for i in range(datasize):
        if np.random.random()<0.05:
            type = np.random.choice([0,1,2])
            if type == 0:
                #change mean
                mean = mean + 0.05 * (2*np.random.random(3)-1)
            elif type == 1:
                std = std + 0.001 * (np.random.random(3))
            else:
                mean = mean + 0.05 * (2*np.random.random(3)-1)
                std = std + 0.001 * np.random.random(3)
        if i in insert_list:
            data = np.random.normal(noise_mean,noise_std,(1,3))  
            data = np.append(data,1)
        else:
            data = np.random.normal(mean,std,(1,3))  
            data = np.append(data,0)
        data_all = np.vstack((data_all,data))
    data_all = np.delete(data_all,0,axis=0)
    print(data_all)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for j in range(5000):
        if data_all[j,3] == 0:
            ax.scatter3D(data_all[j,0], data_all[j,1], data_all[j,2],c = 'green')
        else:
            ax.scatter3D(data_all[j,0], data_all[j,1], data_all[j,2],c = 'red')
         
    #ax.scatter3D(data_all[:,0], data_all[:,1], data_all[:,2],c = data_all[:,3],cmap = ['b','r'])
    plt.show()
    
    np.savetxt("datasets/SimData1.csv",data_all,delimiter=',')    
    '''
    batch = 100
    for i in range(10):
        label = np.zeros((batch,1))
        mean = mean + np.random.random(3)
        std = std + 0.1*np.random.random(3)
        data = np.random.normal(mean,std, (batch,3)) 
        #insert_points = 
        insert_l = np.array([])
        if(i>2):
            insert_l = np.random.choice(np.arange(len(data)),int(len(data)*0.01))
        for j in insert_l:
            data[j] = np.random.normal(noise_mean,noise_std, (1,3)) 
            label[j,0] = 1

        data = np.hstack((data,label))
        data_all = np.vstack((data_all,data))
        #print(data)

    data_all = np.delete(data_all,0,axis=0)
    print(data_all)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for j in range(len(data_all)):
        if data_all[j,3] == 0:
            ax.scatter3D(data_all[j,0], data_all[j,1], data_all[j,2],c = 'green')
        else:
            ax.scatter3D(data_all[j,0], data_all[j,1], data_all[j,2],c = 'red')
         
    #ax.scatter3D(data_all[:,0], data_all[:,1], data_all[:,2],c = data_all[:,3],cmap = ['b','r'])
    plt.show()

    np.savetxt("datasets/SimData1.csv",data_all,delimiter=',')
    '''