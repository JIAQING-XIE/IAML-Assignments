
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

file_path = os.path.join(os.getcwd(),'data')
Xtrn,Ytrn,Xtst,Ytst = load_CoVoST2(file_path)



#<----

# Q3.1
def iaml01cw2_q3_1():
    kmeans = KMeans(n_clusters=22, random_state=1).fit(Xtrn)
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
        temp = []
        for i in range(Xtrn.shape[0]):
            if Ytrn[i] == j:
                temp.append(Xtrn[i,:])
        temp = np.array(temp)
        mean_vector.append(list(temp.mean(axis = 0)))
    mean_vector = np.array(mean_vector) # mean vector of each class (22, 26)

    pca = PCA(n_components = 2)
    z = pca.fit_transform(mean_vector)

    kmeans = KMeans(n_clusters=22, random_state=1).fit(Xtrn)

    z2 = pca.transform(kmeans.cluster_centers_)


    plt.figure(figsize=(10,10))
    plt.scatter(z[:,0],z[:,1], c = 'red')
    plt.scatter(z2[:,0], z2[:,1], c = 'blue')
    plt.show()

#
#iaml01cw2_q3_2()   # comment this out when you run the function

# Q3.3
def iaml01cw2_q3_3():
    mean_vector = []
    for j in range(22):
        temp = []
        for i in range(Xtrn.shape[0]):
            if Ytrn[i] == j:
                temp.append(Xtrn[i,:])
        temp = np.array(temp)
        mean_vector.append(list(temp.mean(axis = 0)))
    mean_vector = np.array(mean_vector) # mean vector of each class (22, 26)
    Z = sch.linkage(mean_vector, method =  'ward', metrics ='')
    p = sch.dendrogram(Z, orientation='right')
    plt.savefig('IAML_CW2_Q3_3.png')
    plt.show()
   
#
iaml01cw2_q3_3()   # comment this out when you run the function
'''
# Q3.4
def iaml01cw2_q3_4():
#
# iaml01cw2_q3_4()   # comment this out when you run the function

# Q3.5
def iaml01cw2_q3_5():
#
# iaml01cw2_q3_5()   # comment this out when you run the function
'''
