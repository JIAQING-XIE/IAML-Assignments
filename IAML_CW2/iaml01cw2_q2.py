
##########################################################
#  Python script template for Question 2 (IAML Level 10)
#  Note that
#  - You should not change the filename of this file, 'iaml01cw2_q2.py', which is the file name you should use when you submit your code for this question.
#  - You should define the functions shown below in your code.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define helper functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission
##########################################################

#--- Code for loading the data set and pre-processing --->
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from iaml01cw2_helpers import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
#<----

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





# Q2.1
def iaml01cw2_q2_1():
    

    lr = LogisticRegression()
    lr.fit(Xtrn_nm,Ytrn)
    Ytst_pred = lr.predict(Xtst_nm)
    print("Test Accuracy: {}".format(accuracy_score(Ytst,Ytst_pred)))   # print classification accuracy for test cases
    print("Confusion Matrix:")
    print(confusion_matrix(Ytst, Ytst_pred)) # print confusion matrix for test cases

#
#iaml01cw2_q2_1()   # comment this out when you run the function

# Q2.2
def iaml01cw2_q2_2():
    svc = SVC(kernel='rbf', C = 1.0, gamma='auto')
    svc.fit(Xtrn_nm, Ytrn)
    #svc.fit(Xtst_nm,Ytst)
    Ytst_pred =  svc.predict(Xtst_nm)
    print("Test Accuracy: {}".format(accuracy_score(Ytst,Ytst_pred)))   # print classification accuracy for test cases
    print("Confusion Matrix: ")
    print(confusion_matrix(Ytst, Ytst_pred)) # print confusion matrix for test cases
    

#
#iaml01cw2_q2_2()   # comment this out when you run the function

# Q2.3
def iaml01cw2_q2_3():
    #1 Fit the model first
    lr = LogisticRegression()
    lr.fit(Xtrn_nm,Ytrn)           
                                                                
    # suit the new points with pca to original space
    # then predited labels with lr
    pca = PCA(n_components=2)
    z = pca.fit_transform(Xtrn_nm) # z.shape = (60000,2) // fit pca model
    std_pc1 = np.std(z[:,0]) # sigma_1
    std_pc2 = np.std(z[:,1]) # sigma_2

    new_fit_x = np.linspace(-5 * std_pc1, 5 * std_pc1, 100)
    new_fit_y = np.linspace(-5 * std_pc2, 5 * std_pc2, 100)

    X,Y = np.meshgrid(new_fit_x, new_fit_y)
    a = np.c_[X.ravel(), Y.ravel()] # PCA space

    temp = pca.inverse_transform(a)
    pred = lr.predict(temp)
    pred1 = pred.reshape((100,100))

    cmap = plt.get_cmap('coolwarm', 10)
    labels = [i for i in range(10)]
    aa = plt.contourf(X, Y, pred1, levels = labels, cmap = cmap)
    plt.colorbar(aa)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("IAML_CW2_Q2_3.png")
    plt.show()
    
#
#iaml01cw2_q2_3()   # comment this out when you run the function

# Q2.4
def iaml01cw2_q2_4():
    svc = SVC(kernel='rbf', C = 1.0, gamma='auto')
    svc.fit(Xtrn_nm, Ytrn) 


    # suit the new points with pca to original space
    # then predited labels with lr
    pca = PCA(n_components=2)
    z = pca.fit_transform(Xtrn_nm) # z.shape = (60000,2) // fit pca model
    

    std_pc1 = np.std(z[:,0]) # sigma_1
    std_pc2 = np.std(z[:,1]) # sigma_2

    new_fit_x = np.linspace(-5 * std_pc1, 5 * std_pc1, 100)
    new_fit_y = np.linspace(-5 * std_pc2, 5 * std_pc2, 100)
    X,Y = np.meshgrid(new_fit_x, new_fit_y)
    a = np.c_[X.ravel(), Y.ravel()]

    temp = pca.inverse_transform(a)
    pred = svc.predict(temp)
    #pred = svc.predict(a)
    pred1 = pred.reshape((100,100))


    cmap = plt.get_cmap('coolwarm', 10)

    labels = [i for i in range(10)]
    aa = plt.contourf(X, Y, pred1,  levels = labels, cmap = cmap)
    
    plt.colorbar(aa)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig("IAML_CW2_Q2_4.png")
    plt.show()

#iaml01cw2_q2_4()   # comment this out when you run the function

# Q2.5
def iaml01cw2_q2_5():
    Xsmall = []
    Ysmall = []
    a = [0 for i in range(10)]


    for i in range(len(Ytrn)):
        if a[Ytrn[i]] < 1000:
            a[Ytrn[i]]+=1
            Xsmall.append(Xtrn_nm[i,:])
            Ysmall.append(Ytrn[i])
        else:
            continue

    Xsmall = np.array(Xsmall)
    Ysmall = np.array(Ysmall)
    b = np.linspace(-2,3,10)
    c = [10**i for i in b]
    

    mean_acc = []

    maximum = 0
    optimum_c = 0

    for param in c:
        print("param")
        svc = SVC(C=param, kernel='rbf', gamma='auto')
        arr = cross_val_score(svc,Xsmall,Ysmall,cv=StratifiedKFold(3))
        #arr = cross_val_score(svc,Xsmall,Ysmall,cv=KFold(3))
        mean_acc.append(arr.mean())
        if maximum < arr.mean():
            maximum = arr.mean()
            optimum_c = param
    
    print("--------- Optimal Value --------")
    print("Optimal C: {}".format(optimum_c))
    print("Best Mean Accuracy: {}".format(maximum))

    plt.scatter(b,mean_acc)
    plt.xlabel("log-scale of C")
    plt.ylabel("mean accuracy (CV)")
    plt.savefig("IAML_CW2_Q2_5.png")
    plt.show()
    #print(mean_acc)



#
#iaml01cw2_q2_5()   # comment this out when you run the function

# Q2.6 
def iaml01cw2_q2_6():
    b = np.linspace(-2,3,10)
    c = [10**i for i in b]
    c_optiomal = c[6]
    svc = SVC(C= c_optiomal, gamma='auto',kernel='rbf')
    svc.fit(Xtrn_nm,Ytrn)
    Ytrn_pred = svc.predict(Xtrn_nm)
    Ytst_pred =svc.predict(Xtst_nm)

    print("training accuracy:{}".format(accuracy_score(Ytrn,Ytrn_pred)))
    print("test accuracy:{}".format(accuracy_score(Ytst,Ytst_pred)))
#
iaml01cw2_q2_6()   # comment this out when you run the function
