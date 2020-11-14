 
##########################################################
#  Python script template for Question 3 (IAML Level 10)
#  Note that:
#  - You should not change the filename of this file, 'iaml01cw2_q3.py', which is the file name you should use when you submit your code for this question.
#  - You should define the functions shown below in your code.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission
##########################################################

#--- Code for loading the data set and pre-processing --->
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns
from iaml01cw2_helpers import *
import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

file_path = os.path.join(os.getcwd(),'data')


Xtrn,Ytrn,Xtst,Ytst = load_CoVoST2(file_path)



#<----

# Q3.1
def iaml01cw2_q3_1():
    kmeans = KMeans(n_clusters=22, random_state=1)
    kmeans.fit(Xtrn)
    print(kmeans.cluster_centers_.shape)

    sums = 0

    def squared(numpy_A, numpy_B):
        return sum((numpy_B -numpy_A)**2)
            
    for i in range(Xtrn.shape[0]):
        sums+=squared(Xtrn[i,:],kmeans.cluster_centers_[kmeans.labels_[i]])
    
    print(sums) # total sums
    for j in range(22):
        total_num = 0
        for i in range(Xtrn.shape[0]):
            if(kmeans.labels_[i] == j):
                total_num+=1
        print("Numbers of samples assigned to cluster {} is: {}".format(j, total_num))




#
#iaml01cw2_q3_1()   # comment this out when you run the function

# Q3.2
def iaml01cw2_q3_2():

    mean_vector = []

    for j in range(22):
        temp = Xtrn[np.where(Ytrn == j)]
        temp = np.array(temp)
        mean_vector.append(list(temp.mean(axis = 0)))
    mean_vector = np.array(mean_vector) # mean vector of each class (22, 26)

    kmeans = KMeans(n_clusters=22, random_state=1).fit(Xtrn)
    centers = kmeans.cluster_centers_
    

    pca = PCA(n_components=2)
    zz = pca.fit_transform(mean_vector) # PCA on mean vector 
    center_ = pca.transform(centers)
    lang =[i for i in range(22)]

    # ------- plot --------#
    plt.figure(figsize=(15,15))
    plt.scatter(center_[:, 0], center_[:, 1], c='blue', label ='cluster centers', s = 12)
    for i in range(center_.shape[0]):
        plt.annotate(lang[i],xy = (center_[i, 0], center_[i, 1]),xytext = (center_[i, 0]+0.01, center_[i, 1]+0.01), color = 'blue', size = 16)
    plt.scatter(zz[:, 0], zz[:, 1], c='red', label = 'mean vectors', s= 12)
    for i in range(zz.shape[0]):
        plt.annotate(lang[i],xy = (zz[i, 0], zz[i, 1]),xytext = (zz[i, 0]+0.01, zz[i, 1]+0.01), color = 'red', size = 16)
    plt.xlabel('Principal Component 1', fontsize = 20)
    plt.ylabel('Principal Component 2', fontsize = 20)
    plt.legend()
    plt.savefig("IAML_CW2_Q3_2.png")
    plt.show()

#
#iaml01cw2_q3_2()   # comment this out when you run the function

# Q3.3
def iaml01cw2_q3_3():

    # read file
    file_path = os.path.join(os.getcwd(), 'data')
    lang = pd.read_csv(file_path + '/languages.txt',header = None, sep= ' ')
    lang_label = []
    # add appropriate labels
    for i in range(22):
        if i <10:
            lang_label.append(lang.iloc[i,2])
        else:
            lang_label.append(lang.iloc[i,1])

    

    
    mean_vector = []
    
    for j in range(22):
        temp = Xtrn[np.where(Ytrn == j)]
        mean_vector.append(list(temp.mean(axis = 0)))
    mean_vector = np.array(mean_vector) # mean vector of each class (22, 26)
    plt.figure(figsize=(10,10))
    Z = sch.linkage(mean_vector, method =  'ward')
    p = sch.dendrogram(Z, orientation='right', labels= lang_label)
    plt.xlabel("distance (Ward)", fontsize = 20)
    plt.ylabel("Languages", fontsize = 20)
    plt.savefig('IAML_CW2_Q3_3.png')
    
    plt.show()
   
#
#iaml01cw2_q3_3()   # comment this out when you run the function

# Q3.4
def iaml01cw2_q3_4():

    # read file
    file_path = os.path.join(os.getcwd(), 'data')
    lang = pd.read_csv(file_path + '/languages.txt',header = None, sep= ' ')
    lang_label = []
    # add appropriate labels
    for i in range(22):
        if i <10:
            for k in range(3):
                lang_label.append(lang.iloc[i,3] + ' ' + str(k))
        else:
            for k in range(3):
                lang_label.append(lang.iloc[i,2] + '' + str(k))

    print(lang_label)

    kmeans = KMeans(n_clusters=3, random_state=1) # apply k-Means algorithm
    clusters = []
    for i in range(22):
        kmeans.fit(Xtrn[np.where(Ytrn == i)])
        for j in range(3):
            clusters.append(kmeans.cluster_centers_[j,:])
    clusters = np.array(clusters)
    print(clusters)
    plt.figure(figsize=(13,5))
    Z = sch.linkage(clusters, method = 'ward')
    p = sch.dendrogram(Z, labels=lang_label,leaf_rotation=45)
    plt.xlabel('Language Cluster Centers', fontsize = 10)
    plt.ylabel('distance (ward)', fontsize = 15)
    plt.savefig("IAML_CW2_Q3_4_ward.png")
    plt.show()
    
    plt.figure(figsize=(13,5))
    Z = sch.linkage(clusters, method = 'single')
    p = sch.dendrogram(Z, labels=lang_label,leaf_rotation=45)
    plt.xlabel('Language Cluster Centers', fontsize = 10)
    plt.ylabel('distance (single)', fontsize = 15)
    plt.savefig("IAML_CW2_Q3_4_single.png")
    plt.show()
    
    plt.figure(figsize=(13,5))
    Z = sch.linkage(clusters, method = 'complete')
    p = sch.dendrogram(Z, labels=lang_label,leaf_rotation=45)
    plt.xlabel('Language Cluster Centers', fontsize = 10)
    plt.ylabel('distance (complete)', fontsize = 15)
    plt.savefig("IAML_CW2_Q3_4_complete.png")
    plt.show()
    


#
iaml01cw2_q3_4()   # comment this out when you run the function

# Q3.5
def iaml01cw2_q3_5():
    cov_type = ['diag', 'full']
    K = [1,3,5,10,15]
    diag_train = []
    full_train = []

    diag_test = []
    full_test = []

    for k in K:
        for covtype in cov_type:
            model = GaussianMixture(n_components=k,covariance_type=covtype)
            model.fit(Xtrn[np.where(Ytrn == 0)])
            #print(model.score(Xtrn[np.where(Ytrn == 0)]))
            score_train = model.score(Xtrn[np.where(Ytrn == 0)])
            score_test = model.score(Xtst[np.where(Ytst == 0)])
            if covtype == 'diag':
                diag_train.append(score_train)
                diag_test.append(score_test)
            else:
                full_train.append(score_train)
                full_test.append(score_test)

    #plt.figure()
    

    plt.scatter(K, diag_train, label = 'train diagonal covariance')
    plt.scatter(K, full_train, label ='train full covariance')
    plt.xlabel('number of mixture components (K)', fontsize=10)
    plt.ylabel('per-sample average log-likelihood', fontsize=10)
    plt.title('Training Set', fontsize=10)


    plt.scatter(K, diag_test, label = 'test diagonal covariance')
    plt.scatter(K, full_test, label ='test full covariance')
    plt.xlabel('number of mixture components (K)', fontsize=10)
    plt.ylabel('per-sample average log-likelihood', fontsize=10)
    plt.title('Test Set', fontsize=10)
    plt.legend()
    plt.savefig('Q3_5.png')
    plt.show()

            
#
#iaml01cw2_q3_5()   # comment this out when you run the function

