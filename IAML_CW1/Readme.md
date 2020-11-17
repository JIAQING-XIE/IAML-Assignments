# IAML 2020 - INFR10069 (Level 10) Assignment 1
This is the repository for explaining Assignment 1 for IAML 2020. (65/80)
The losed point is due to runnning environment conflicts and two small explanations. So be careful that you are using the scikit-learn==0.19.1
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

## Question 3
3(a) Describe the data as the method in Q1 (a)

3(b) You should calculate mean for each class for each dimension and scatter them. Used packages: ```python matplotlib.pyplot```

3(c) The default criteria of deciding the impurity of the node is Gini impurity. One obvious advantage is that entropy calculation
needs logarithm operation, which costs more time than multiplication operation that is required by Gini impurity. Besides, we know intuitively that it tends to choose the most frequent class of at the branches.(Explanation got full points)

3(d)The potential problem which small values of maximum depth of the tree may occur is that the impurity is high at the top nodes of the tree. It will result in under- fitting problem. Maximum depth with unsuitable large values will cause over-fitting problems. Also, if numbers of data is large, it will spend more time on calculation. Expected time complexity is O(nlgn) since we choose lg(n) features in our tree. Time complexity will become O(n2) if we add more numbers of features to the branches until it reaches the number of maximum input features.(Explanation got full points)
 
3(e)Training three SupportVectorClassifiers with different maximum depths--2, 8 and 20. Used packages: ```python sklearn.tree.DecisionTreeClassifier``` and ```python sklearn.metrics.accuracy_score``` The second model with the maximum depth of 8 preforms best. Since value 8 is the closet to expected maximum depth lg(n) where n is equal to 136. ln(136) is equal to 7.09.

3(f)The top three most important attributes are x50, y48 and y29, of which the im- portance is equal to 0.3304, 0.0900 and 0.0883 in order respectively. The attribute with the highest importance is x50. It makes sense since top of the upper lip, whose horizontal value is represented by x50 will extend when one person is smiling. It is common sense although the disparity is not obvious in the figure shown in part(b).
 
3(g)One limitation is that more than half of the coefficients(feature importance) are equal to zero(71/136) according to the reported feature_importances_. It means that those features are not sensitive to label prediction, which is redundant to classifier.
 
