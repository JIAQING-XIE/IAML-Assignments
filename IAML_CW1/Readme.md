# IAML 2020 - INFR10069 (Level 10) Assignment 1
This is the repository for Assignment 1 for IAML 2020.

## Question 1
1(a)  You can have the description of the data by dataframe **A**.describe(). Firstly you should load data with pd.read_csv or pd.read_table. Needed packages: numpy, pandas

1(b)  Just fit a linear model with LinearRegression(fit_intercept = 0). You can concatenate(append) an array of 1 to original data. Needed packages: sklearn.linear_model.LinearRegression. report the weight parameter by LR.coef_

1(c) Use Lr.fit(), then predict the answer with input data. Plot the answer and input on a two dimensional plane with matplotlib.pyplot

1(d) 
```python
x = np.array(data.iloc[:,0:2]) #x
y = np.array(data.iloc[:,2]) #y
T_x = np.linalg.pinv(x.T.dot(x)) #T_x = (x^T x)^-1
w = T_x.dot(x.T.dot(y)) #w = (x^T x)^-1 (x^T y
```

