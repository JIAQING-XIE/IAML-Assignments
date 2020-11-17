# IAML 2020 - INFR10069 (Level 10) Assignment 1
This is the repository for explaining Assignment 1 for IAML 2020.

## Question 1
1(a)  You can have the description of the data by dataframe **A**.describe(). Firstly you should load data with pd.read_csv or pd.read_table. Needed packages: ```python numpy, pandas```

1(b)  Just fit a linear model with LinearRegression(fit_intercept = 0). You can concatenate(append) an array of 1 to original data. Needed packages: ```python sklearn.linear_model.LinearRegression```. report the weight parameter by LR.coef_

1(c) Use Lr.fit(), then predict the answer with input data. Plot the answer and input on a two dimensional plane with matplotlib.pyplot

1(d) The concept of least square multiplication method, also named the closed-form solution for a linear regression model.
```python
x = np.array(data.iloc[:,0:2]) #x
y = np.array(data.iloc[:,2]) #y
T_x = np.linalg.pinv(x.T.dot(x)) #T_x = (x^T x)^-1
w = T_x.dot(x.T.dot(y)) #w = (x^T x)^-1 (x^T y
```

1(e) The concept of MSE. One limitation is variance-bias trade off. You can find that in the book *The Elements of Statistical Learning*, or just google it. The method is rectified to L1-norm (Lasso) or L2-norm(Ridge)

1(f) Report MSE between predicted and raw data. Used packages: ```python sklearn.metrics.mean_squared_error ```

1(g) Using the equation y = Wx + b, plot MSE between the predicted and raw data. All 100 points in total. Used packages: ```python matplotlib.pyplot```

## Question 2
2(a) You should first preprocess the data into the wanted values by the transformation equation. Then used linear model to fit it.
Used packages: ```python sklearn.preprocessing.PolynomialFeatures``` and ```python sklearn.linear_model.LinearRegression```

2(b) Report Mean Squared Error, as detailed in Q1. Used packages:```python sklearn.metrics.mean_squared_error ```

2(c) When model hyper-parameter M equals to 3 or 4, they result in almost the same performance and they fit the data well. This can be found in the plot shown in part (a) where fitted curves under these two models are overlapped. Both two models are better than models with M = 1 and M = 2. I would choose the model with M = 3. Firstly it reaches almost the best performance among all four models. Secondly, compared to the fourth model, it has fewer calculation time with the time complex- ity O(N3). It really matters when numbers of input data are large. Moreover, using the model with M = 4 may cause over-fitting problem when encountering unseen data while applying the model with M = 1 and M = 2 can cause under-fitting problem.
 

2(d) Just like the process in 2(a). Be careful with the concatenation!!!
