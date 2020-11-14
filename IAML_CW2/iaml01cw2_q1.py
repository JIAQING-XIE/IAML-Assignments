


#--- Code for loading the data set and pre-processing --->
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

import os
from iaml01cw2_helpers import *
from sklearn.decomposition import PCA
from sklearn.metrics import  mean_squared_error as mse
from math import sqrt


data_file = os.path.join(os.getcwd(),'data')
Xtrn, Ytrn, Xtst, Ytst = load_FashionMNIST(data_file)
#1. Back up
Xtrn_orig = Xtrn[:]
Xtst_orig = Xtst[:]

#2. Divided by 255.0
Xtrn = Xtrn / 255.0
Xtst = Xtst / 255.0

#3. Calculate mean value of Xtrn's columns, shape=(784,)
Xmean = np.sum(Xtrn,axis = 0)/Xtrn.shape[0]
#print(Xmean.shape)


#4.Subtract Xmean from each row of Xtrn and Xtst, and store the result in Xtrn_nm
#and Xtst_nm, respectively.
Xtrn_nm = Xtrn - Xmean
Xtst_nm = Xtst - Xmean


# Q1.1
def iaml01cw2_q1_1():
    print(Xtrn_nm[0,0:4])
    print(Xtrn_nm[-1,0:4])
 

#iaml01cw2_q1_1()   # comment this out when you run the function


# Q1.2
def iaml01cw2_q1_2():
    # O(N^2)
    record = []
    for j in range(0,10):
        temp = []
        for i in range(Xtrn_nm.shape[0]):
            if Ytrn[i] == j:
                temp.append(i)
        record.append(list(temp))
        #print(len(temp)) #each class 6000 samples
    record = np.array(record)   # id in each class
    #print(record.shape)         #(10,6000)
    print(record.shape)

    dis = []
    vector = []
    for i in range(record.shape[0]):
        mean_vec = 0
        
        temp = []
        for j in range(record.shape[1] ):
            mean_vec = mean_vec + Xtrn[record[i][j],:]
        mean_vec = mean_vec / record.shape[1]
        vector.append(mean_vec)
        for j in range(record.shape[1]):
            a = np.linalg.norm(Xtrn[record[i][j],:]-mean_vec) ### This is much faster than self-defined eucildean distance function
            temp.append(a)
        dis.append(temp)
    dis = np.array(dis) # ->distance (10, 6000)

    index = []
    for i in range(dis.shape[0]):
        arr = dis[i,:]
        #print(len(arr))
        max_2 = arr.argsort()[::-1][0:2]
        
        min_2 = arr.argsort()[::-1][dis.shape[1]-2:dis.shape[1]]
        min_2 = np.append(min_2,max_2)

        index.append(list(min_2))
    #print(index) # 1-closet, 2-closet, 2-furthest, 1-furthest (10,4)

    #--------- print ---------#
    vector = np.array(vector)



    f, axs = plt.subplots(10,5,figsize=(30,40))

    for i in range(10):
        for j in range(5):
            plt.subplot(10, 5, 5* i + j +1)
            plt.axis('off')
            if j == 0:
                a = vector[i,:].reshape((28,28))
                plt.imshow(a, cmap = "gray_r")
                plt.title("mean vector of class: {}".format(i),y=-0.2,fontsize=24,fontweight='bold')
            else:
                a = Xtrn_nm[record[i][index[i][j-1]],:].reshape((28,28))
                plt.imshow(a, cmap = "gray_r")
                plt.title("class: {}, id: {}".format(i,record[i][index[i][j-1]]),y=-0.2,fontsize=24,fontweight='bold')
    
    plt.savefig("IAML_CW2_Q1_2.png")
        
                
#iaml01cw2_q1_2()   # comment this out when you run the function

# Q1.3
def iaml01cw2_q1_3():
    pca = PCA()
    pca.fit(Xtrn_nm)
    print(pca.explained_variance_[0:5]) #first five principal components

#
#iaml01cw2_q1_3()   # comment this out when you run the function


# Q1.4
def iaml01cw2_q1_4():
    pca = PCA()
    pca.fit(Xtrn_nm)
    
    #print(pca.explained_variance_ratio_) #pca n principal components fit

    y = pca.explained_variance_ratio_.cumsum()
    
    x90 = 0
    y90 = 0

    x95 = 0
    y95 = 0

    x99 = 0
    y99 = 0

    mini90 = 1
    mini95 = 1
    mini99 = 1

    for i in range(len(y)):
        if abs(y[i]-0.9)< mini90 and y[i] > 0.9:
            mini90 = abs(y[i]-0.9)
            y90 = y[i]
            x90 = i
        else:
            if abs(y[i] -0.95) < mini95 and y[i] > 0.95:
                mini95 = abs(y[i] -0.95)
                y95 = y[i]
                x95 = i
            else:
                if abs(y[i] -0.99) < mini99 and y[i] > 0.99:
                    mini99 = abs(y[i] -0.99)
                    y99 = y[i]
                    x99 = i


    

    x = [i for i in range(len(y))]
    print(len(x))
    plt.plot(x,y)
    plt.xlabel("Number of Principal Compoents (N)")
    plt.ylabel("Cumulative explained variance ratio (R)")
    #----- Speical percentile, like 90% and 95% -----#
    plt.scatter(x90, y90, c = 'red')
    plt.scatter(x95, y95, c = 'red')
    plt.scatter(x99, y99, c = 'red')

    plt.text(x90 + 10, y90 - 0.03, (round(x90, 2), round(y90,2)), color='b')
    plt.text(x95 + 10, y95 - 0.03, (round(x95, 2), round(y95,2)), color='b')
    plt.text(x99 + 10, y99 - 0.04, (round(x99, 2), round(y99,2)), color='b')

    plt.savefig("IAML_CW2_Q1_4.png")
    plt.show()
#
#iaml01cw2_q1_4()   # comment this out when you run the function


# Q1.5
def iaml01cw2_q1_5():

    pca = PCA(n_components=10)
    pca.fit(Xtrn_nm)
    a = pca.transform(Xtrn_nm)
    components = pca.components_
    print(components.shape)


    f, axs = plt.subplots(2,5,figsize=(15,6))


    for i in range(2):
        for j in range(5):
            plt.subplot(2, 5, 5* i + j +1,aspect='equal')
            plt.imshow(components[i * 5 +j].reshape((28,28)), cmap = 'gray_r')
            plt.title("PC {}".format(5 * i + j +1))
            plt.axis('off')
    plt.savefig("IAML_CW2_Q1_5.png") 

#
#iaml01cw2_q1_5()   # comment this out when you run the function


# Q1.6
def iaml01cw2_q1_6():
    K = [5, 20, 50, 200]
    rmse = []
    a = [0 for i in range(10)]
    b = [0 for i in range(10)]
    #find the first sample in the class:
    for i in range(len(Ytrn)):
        if a[Ytrn[i]]!=1:
            b[Ytrn[i]] = i
            a[Ytrn[i]] = a[Ytrn[i]] + 1
    

    for ii in K:
        temp =[]
        
        pca = PCA(n_components=ii)
        for jj in range(10):
            x_reduced = pca.fit_transform(Xtrn_nm[np.where(Ytrn == jj)])
            x_recovered = pca.inverse_transform(x_reduced)
            temp.append(sqrt(mse(Xtrn_nm[b[jj],:], x_recovered[0,:])))
        rmse.append(temp)
        
    rmse = np.array(rmse).T
    np.savetxt('Q1_6_ans.txt', rmse)

#iaml01cw2_q1_6()   # comment this out when you run the function


# Q1.7
def iaml01cw2_q1_7():
    K = [5, 20, 50, 200]
    a = [0 for i in range(10)]
    b = [0 for i in range(10)]
    #find the first sample in the class:
    for i in range(len(Ytrn)):
        if a[Ytrn[i]]!=1:
            b[Ytrn[i]] = i
            a[Ytrn[i]] = a[Ytrn[i]] + 1
    
    f, axs = plt.subplots(10,4,figsize=(30,40))
    count = 0
    for ii in K:
        count = count + 1
         
        pca = PCA(n_components=ii)

        count2 = 0

        for jj in range(10):
            x_reduced = pca.fit_transform(Xtrn_nm[np.where(Ytrn == jj)])
            x_recovered = pca.inverse_transform(x_reduced)
            plt.subplot(10,4, (count -1) + 4 * count2+1)
            plt.imshow((x_recovered[0,:] + Xmean).reshape((28,28)), cmap ="gray_r")
            plt.axis('off')
            plt.title("class : {}, K = {}".format(Ytrn[b[jj]] , ii), y = -0.4, fontsize=30,fontweight='bold')
            count2 = count2 + 1

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.5)

    plt.savefig("IAML_CW2_Q1_7.png")
    plt.show()


#iaml01cw2_q1_7()   # comment this out when you run the function


# Q1.8
def iaml01cw2_q1_8():


    pca = PCA(n_components=2)
    x_reduced = pca.fit_transform(Xtrn_nm)
    plt.figure(figsize=(20,20))
    ax = plt.subplot(1,1,1,)

    
    print(x_reduced[np.where(Ytrn == 1),0].reshape((6000,)))

    #------ plot -------#
    values = range(10)
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scaMap = plt.cm.ScalarMappable(norm = cNorm  ,cmap = "coolwarm")
    for i in range(10):
        colorval = scaMap.to_rgba(values[i])
        ax.scatter(x_reduced[np.where(Ytrn == i),0].reshape((6000,)), x_reduced[np.where(Ytrn == i),1].reshape((6000,)),label = i, s =2, color = colorval)
    #ax.scatter(x_reduced[:,0], x_reduced[:,1], s =2, c = Ytrn ,cmap="coolwarm")
    




    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right',fontsize = 20)
    plt.xlabel("Principal Component 1",fontsize = 20)
    plt.ylabel("Principal Component 2", fontsize = 20)
    plt.tick_params(labelsize=20)
    plt.savefig("IAML_CW2_Q1_8.png")
    plt.show()
#
#iaml01cw2_q1_8()   # comment this out when you run the function
