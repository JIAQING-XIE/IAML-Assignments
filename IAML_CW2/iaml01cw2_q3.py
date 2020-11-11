 
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


    pca = PCA(n_components=2)
    

    kmeans = KMeans(n_clusters=22, random_state=1).fit(Xtrn)

    zz = pca.fit_transform(mean_vector) # PCA on mean vector 



    centers = kmeans.cluster_centers_
    center_ = pca.transform(centers)
    plt.figure(figsize=(10,10))




    plt.scatter(center_[:, 0], center_[:, 1], c='blue', s= 1)
    plt.scatter(zz[:, 0], zz[:, 1], c='red',s = 1)

    plt.savefig("IAML_CW2_Q3_2.png")
    plt.show()

#
iaml01cw2_q3_2()   # comment this out when you run the function

# Q3.3
def iaml01cw2_q3_3():
    mean_vector = []
    for j in range(22):
        temp = Xtrn[np.where(Ytrn == j)]
        mean_vector.append(list(temp.mean(axis = 0)))
    mean_vector = np.array(mean_vector) # mean vector of each class (22, 26)
    
    Z = sch.linkage(mean_vector, method =  'ward')
    p = sch.dendrogram(Z, orientation='right')
    plt.savefig('IAML_CW2_Q3_3.png')
    
    plt.show()
   
#
#iaml01cw2_q3_3()   # comment this out when you run the function

# Q3.4
def iaml01cw2_q3_4():
    kmeans = KMeans(n_clusters=3, random_state=1)
    clusters = []
    for i in range(22):
        kmeans.fit(Xtrn[np.where(Ytrn == i)])
        for j in range(3):
            clusters.append(kmeans.cluster_centers_[j,:])
    clusters = np.array(clusters)
    print(clusters)
    plt.figure(figsize=(20,20))
    Z = sch.linkage(clusters, method = 'ward')
    p = sch.dendrogram(Z)
    plt.savefig("Q3_4_ward.png")
    plt.show()
    
    plt.figure(figsize=(20,20))
    Z = sch.linkage(clusters, method = 'single')
    p = sch.dendrogram(Z)
    plt.savefig("Q3_4_single.png")
    plt.show()
    
    plt.figure(figsize=(20,20))
    Z = sch.linkage(clusters, method = 'complete')
    p = sch.dendrogram(Z)
    plt.savefig("Q3_4_complete.png")
    plt.show()
    


#
#iaml01cw2_q3_4()   # comment this out when you run the function

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

