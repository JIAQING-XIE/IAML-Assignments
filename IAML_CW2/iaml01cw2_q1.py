
##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the filename of this file, 'iaml01cw2_q1.py', which is the file name you should use when you submit your code for this question.
#  - You should define the functions shown below in your code.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission
##########################################################

#--- Code for loading the data set and pre-processing --->
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
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
        record.append(temp)
        #print(len(temp)) #each class 6000 samples
    record = np.array(record)   # id in each class
    #print(record.shape)         #(10,6000)

    dis = []
    vector = []
    for i in range(record.shape[0]):
        mean_vec = 0
        
        temp = []
        for j in range(record.shape[1] ):
            mean_vec = mean_vec + Xtrn_nm[record[i][j],:]
        mean_vec = mean_vec / 6000
        vector.append(mean_vec)
        for j in range(record.shape[1]):
            a = np.linalg.norm(Xtrn_nm[record[i][j],:]-mean_vec) ### This is much faster than self-defined eucildean distance function
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
    #print(index) # 1-largest, 2-largest, 2-smallest, 1-smallest (10,4)

    #--------- print ---------#
    vector = np.array(vector)



    f, axs = plt.subplots(10,5,figsize=(20,20))

    for i in range(10):
        for j in range(5):
            plt.subplot(10, 5, 5* i + j +1)
            plt.axis('off')
            if j == 0:
                a = vector[i,:].reshape((28,28))
                plt.imshow(a, cmap = "gray_r")
                plt.title("mean_vector of class: {}".format(i),y=-0.2)
            else:
                a = Xtrn_nm[record[i][index[i][j-1]],:].reshape((28,28))
                plt.imshow(a, cmap = "gray_r")
                plt.title("class: {}, sample: {}".format(i,record[i][index[i][j-1]]),y=-0.2)
    
    plt.savefig("IAML_CW2_Q1_2.png")
        
                
# iaml01cw2_q1_2()   # comment this out when you run the function

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
    
    print(pca.explained_variance_ratio_) #pca n principal components fit
    plt.plot(x,pca.explained_variance_ratio_.cumsum())
    plt.savefig("IAML_CW2_Q1_4.png")
    plt.show()
#
# iaml01cw2_q1_4()   # comment this out when you run the function


# Q1.5
def iaml01cw2_q1_5():

    pca = PCA()
    pca.fit(Xtrn_nm)
    a = pca.fit_transform(Xtrn_nm)

    f, axs = plt.subplots(2,5,figsize=(15,6))


    for i in range(2):
        for j in range(5):
            plt.subplot(2, 5, 5* i + j +1,aspect='equal')
            plt.scatter(a[:,j + 5*i], len(a[:,j + 5*i]) *[0])
            plt.ylim(-10, 10)
            plt.xlim(-10,13)
            plt.title("PC {}".format(5 * i + j +1), y = -0.4)
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
        x_reduced = pca.fit_transform(Xtrn_nm)
        x_recovered = pca.inverse_transform(x_reduced)
        #print(x_reduced.shape)
        #print(x_recovered.shape)
        for j in b:
            temp.append(sqrt(mse(Xtrn_nm[j,:], x_recovered[j,:])))
        rmse.append(temp)


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
    
    f, axs = plt.subplots(10,4,figsize=(20,20))
    count = 0
    for ii in K:
        count = count + 1
         
        pca = PCA(n_components=ii)
        x_reduced = pca.fit_transform(Xtrn_nm)
        x_recovered = pca.inverse_transform(x_reduced)
        count2 = 0
        for j in b:
            plt.subplot(10,4, (count -1) + 4 * count2+1)
            plt.imshow((x_recovered[j,:] + Xmean).reshape((28,28)), cmap ="gray_r")
            plt.axis('off')
            plt.title("class : {}, reconstructed, K = {}".format(Ytrn[j] , ii), y = -0.4)
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
    #plt.figure(figsize=(20,20))
    fig, ax = plt.subplots(figsize=(15,15))
    scatter = ax.scatter(x_reduced[:,0], x_reduced[:,1], c = Ytrn ,cmap="coolwarm")
    legend1 = ax.legend(*scatter.legend_elements(num=10),
                    loc="upper left", title="Classes",fontsize = 20)
    ax.add_artist(legend1)
    plt.xlabel("Principal Component 1",fontsize = 20)
    plt.ylabel("Principal Component 2", fontsize = 20)
    
    plt.tick_params(labelsize=20)
    
    plt.savefig("IAML_CW2_Q1_8.png")

    plt.show()
#
iaml01cw2_q1_8()   # comment this out when you run the function
