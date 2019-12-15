#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 00:43:09 2019

@author: dior
"""

# QUESTION 1
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 

# Present the training and testing data points
##################################################
x_train = np.array([ [0],[2],[3],[5] ])                   
y_train = [1,4,9,16] 

x_test = np.array([ [1],[4] ])     
y_test = [3,12]

area = np.pi*30

plt.figure(figsize=(16,7))
plt.scatter(x_train, y_train, s=area, c='blue', alpha=1)
plt.scatter(x_test, y_test, s=area, c='red', alpha=1)
plt.title('Poly with 4-d')
plt.xlabel('x')
plt.ylabel('y')
##################################################

# LinearRegression
##################################################
# lin = LinearRegression() 
# lin.fit(x_train,y_train) 

# plt.plot(x_train, lin.predict(x_train), color = 'red') 
################################################## 
  

# D = 0 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 0) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin0 = LinearRegression() 
lin0.fit(X_poly,y_train) 
plt.plot(x_train, lin0.predict(poly.fit_transform(x_train)), color = 'purple', label = 'p = 0') 
train_result0 = lin0.predict(poly.fit_transform(x_train))  
test_result0 = lin0.predict(poly.fit_transform(x_test))
    
# D = 1 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 1) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin1 = LinearRegression() 
lin1.fit(X_poly,y_train) 
plt.plot(x_train, lin1.predict(poly.fit_transform(x_train)), color = 'green', label = 'p = 1')    
train_result1 = lin1.predict(poly.fit_transform(x_train))   
test_result1 = lin1.predict(poly.fit_transform(x_test))

# D = 2 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 2) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin2 = LinearRegression() 
lin2.fit(X_poly,y_train) 
plt.plot(x_train, lin2.predict(poly.fit_transform(x_train)), color = 'grey', label = 'p =2')     
train_result2 = lin2.predict(poly.fit_transform(x_train))  
test_result2 = lin2.predict(poly.fit_transform(x_test))   
    
# D = 3 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 3) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin3 = LinearRegression() 
lin3.fit(X_poly,y_train) 
plt.plot(x_train, lin3.predict(poly.fit_transform(x_train)), color = 'orange', label = 'p = 3')
train_result3 = lin3.predict(poly.fit_transform(x_train))       
test_result3 = lin3.predict(poly.fit_transform(x_test)) 
    
# D = 4 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin4 = LinearRegression() 
lin4.fit(X_poly,y_train) 
plt.plot(x_train, lin4.predict(poly.fit_transform(x_train)), color = 'blue', label = 'p =4')     
train_result4 = lin4.predict(poly.fit_transform(x_train))  
test_result4 = lin4.predict(poly.fit_transform(x_test))

# D = 8 
################################################## 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 8) 
X_poly = poly.fit_transform(x_train) 
poly.fit(X_poly, y_train) 

lin20 = LinearRegression() 
lin20.fit(X_poly,y_train) 
      
test_result20 = lin20.predict(poly.fit_transform(x_test))


plt.legend()    
plt.show() 



# Training Result
################################################
 
train_result = ['','','','']
train_result = np.row_stack((train_result,train_result0)) 
train_result = np.row_stack((train_result,train_result1)) 
train_result = np.row_stack((train_result,train_result2)) 
train_result = np.row_stack((train_result,train_result3)) 
train_result = np.row_stack((train_result,train_result4)) 

 
train_result = np.delete(train_result, (0), axis=0)

print("------ train_result ------")
print(train_result)


# Test Result
################################################
 
# x_train_matrix = [0,0,0]
# x_train_matrix = np.row_stack((x_train_matrix,x_1)) 
# x_train_matrix = np.row_stack((x_train_matrix,x_2)) 
# x_train_matrix = np.row_stack((x_train_matrix,x_3)) 
# x_train_matrix = np.row_stack((x_train_matrix,x_4)) 
# x_train_matrix = np.delete(x_train_matrix, (0), axis=0)

test_result = ['','']
test_result = np.row_stack((test_result,test_result0)) 
test_result = np.row_stack((test_result,test_result1)) 
test_result = np.row_stack((test_result,test_result2)) 
test_result = np.row_stack((test_result,test_result3)) 
test_result = np.row_stack((test_result,test_result4)) 
test_result = np.delete(test_result, (0), axis=0)

print("------ test_result ------")
print(test_result)



# Typical Bias
################################################
x_test = np.array([ [1],[4] ])     
y_test = [3,12]


Typical_Bias_0 = pow( (float(test_result[0][0]) - float(x_test[0])),2 ) + pow( (float(test_result[0][1]) - float(x_test[1])), 2)/2
Typical_Bias_1 = pow( (float(test_result[1][0]) - float(x_test[0])),2 ) + pow( (float(test_result[1][1]) - float(x_test[1])), 2)/2
Typical_Bias_2 = pow( (float(test_result[2][0]) - float(x_test[0])),2 ) + pow( (float(test_result[2][1]) - float(x_test[1])), 2)/2
Typical_Bias_3 = pow( (float(test_result[3][0]) - float(x_test[0])),2 ) + pow( (float(test_result[3][1]) - float(x_test[1])), 2)/2
Typical_Bias_4 = pow( (float(test_result[4][0]) - float(x_test[0])),2 ) + pow( (float(test_result[4][1]) - float(x_test[1])), 2)/2

 
Typical_Bias = []
Typical_Bias.append(Typical_Bias_0)
Typical_Bias.append(Typical_Bias_1)
Typical_Bias.append(Typical_Bias_2)
Typical_Bias.append(Typical_Bias_3)
Typical_Bias.append(Typical_Bias_4)

print(Typical_Bias)
 
# Variance
################################################
 
Variance = []
k = 0 

print(len(Typical_Bias))

for i in range(len(Typical_Bias)): 
    Variance.append( Typical_Bias[k]/2 )
    k += 1
 
print(Variance)

# Total Error
################################################
Total_Error = [] 

for i in range(len(test_result)): 
    Total_Error.append(float(test_result[i][0]) - float(x_test[0])  +  float(test_result[i][1]) - float(x_test[1]) )

print(Total_Error)

# Training Error
################################################
x_train = np.array([ [0],[2],[3],[5] ])                   
y_train = [1,4,9,16] 

training_error = []
for i in train_result:# Get every training result
    
    k = 0 
    training_error_sum = 0
    for j in i:  
        training_error_sum = training_error_sum + pow( y_train[int(k)] - float(i[k]), 2 )       
        k += 1
    
    training_error.append(training_error_sum/4)

print("------- Training Error ------")
print(training_error)
     
 

# Test Error
################################################
testing_error = []

for i in test_result:# Get every training result
    
    k = 0 
    testing_error_sum = 0
    for j in i:  
        testing_error_sum = testing_error_sum + pow( y_test[int(k)] - float(i[k]), 2 )       
        k += 1
    
    testing_error.append(testing_error_sum/4)

print("------- Testing Error ------")
print(testing_error)
     

# Plot
################################################
x_point = [0,1,2,3,4]

plt.figure(figsize=(14,7))

plt.plot(x_point, Typical_Bias , color = 'yellow', label = 'Typical_Bias')
plt.plot(x_point, Variance , color = 'blue', label = 'Variance')
plt.plot(x_point, Total_Error , color = 'pink', label = 'Total_Error')
plt.plot(x_point, training_error , color = 'black', label = 'Training_error')
plt.plot(x_point, testing_error , color = 'grey', label = 'Testing_error')     
  
plt.title('Error')
plt.xlabel('x')
plt.ylabel('y')

plt.legend()    
plt.show() 

######################################################
# (3) Explain why each of the five curves has the shape displayed in part (2).
# --> Here we can see that when d=2 the error come to the minumum. Why? Because 1) when d is 0 or 1, it is not enough to fit the data probably.2) While when d > 2, the curve is overfited - which means the curve is fluctuate too muc - which away from the ground truth more and more when d become large



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # data visualization library
from scipy.optimize import minimize
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold

boston_dataset = load_boston()
df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

print (df.head(5))

#the data can be downloaded from "https://github.com/jcrouser/islr-python/blob/master/data/Smarket.csv"
df = pd.read_csv('Smarket.csv', usecols=range(0,10), index_col=0, parse_dates=True)

X = df[['Lag1','Lag2']].values
Y = df['Today'].values


print(X[:,1])

print(len(X))
print(len(Y))

##########################################
def forward(X_1,X_2,params):
    # 3d Line Z = aX_1 + bX_2 +c
    return X_1.dot(params[:1]) + X_2.dot(params[1:2]) + params[2]

def cost_function(params,X_1,X_2,y,p):
    error_vector = y - forward(X_1,X_2,params)
    return np.linalg.norm(error_vector, ord=p)    
##########################################

kf = KFold(n_splits=5)
kf.get_n_splits(X)

#print(kf)
#KFold(n_splits=5, random_state=None, shuffle=False)
 
#print("-----------------------------")
#print(type(kf.split(X)))
#

 
L2_Norm_P1 = 0
L2_Norm_P2 = 0 
for train_index, test_index in kf.split(X):
  #  print("TRAIN:", train_index, "TEST:", test_index)
    
    ### For X part
    X_train, X_test = X[train_index], X[test_index]
    ### For Y part
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    ############################ P = 1
    output1 = minimize(  cost_function, [0.5,0.5,1], args=(  np.c_[X_train[:,0]], np.c_[X_train[:,1]],Y_train,1   )   )
    y_hat1 = forward(np.c_[X_train[:,0]],np.c_[X_train[:,1]], output1.x) # print(y_hat1)
    
    ### Validation
    L2_Norm_Sum = 0

    for i in range(len(Y_test)): 
         L2 = np.power((y_hat1[i] - Y_test[i]),2)
         L2_Norm_Sum = L2_Norm_Sum + L2
         
    # Add all together     
    L2_Norm_P1 = L2_Norm_P1 + np.sqrt(L2_Norm_Sum)
    
    
    ############################ P = 2
    ### Training
    output2 = minimize(  cost_function, [0.5,0.5,1], args=(  np.c_[X_train[:,0]], np.c_[X_train[:,1]],Y_train,2   )   )
    y_hat2 = forward(np.c_[X_train[:,0]],np.c_[X_train[:,1]], output2.x) # print(y_hat2)
    
    ### Validation
    L2_Norm_Sum = 0

    for i in range(len(Y_test)): 
         L2 = np.power((y_hat2[i] - Y_test[i]),2)
         L2_Norm_Sum = L2_Norm_Sum + L2
         
    # Add all together  
    L2_Norm_P2 = L2_Norm_P2 + np.sqrt(L2_Norm_Sum)
    
    
L2_Norm_P1_Mean = L2_Norm_P1/5
L2_Norm_P2_Mean = L2_Norm_P2/5


print(L2_Norm_P1_Mean)
print(L2_Norm_P2_Mean)

############################################################
# Please compare the mean values across all the 5 folds together. Justify your results. 
# --> I believe the reason is that, after cross validation, the error can be minimized. Thus even we trained with different P value, the result is similar





# QUESTION 2
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================







# QUESTION 3
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
# ==========================================================================================================
 

import numpy as np
import pandas as pd 
import seaborn as sns   

df = pd.read_csv('hw1_input.csv', usecols=range(0,10), index_col=0, parse_dates=True)

sns.pairplot(df)


# =============================================================================
# (2) Logistic Regression Regularization Comparison with Bootstrapping:  

#Using 80% of the data as a training set and 20% as a testing set in each bootstrap repeated 1000 times each, please implement and compare the average and the standard deviation of coefficients obtained from Ridge regression and LASSO regularized logistic regression. 

#By averaging the coefficients it is meant that all the different coefficient values for a specific variable is averaged. Coefficient average values then are to be compared by plotting them against each other. Are all input features necessary? Please describe how categorical features are handled.

# (3) Please plot the ROC curve for both models for a single bootstrap data. What are the area under the curve measurements?
# (4) What is the optimal decision threshold to maximize the f1 score?
# (5) Please provide a mean and standard deviation for the AUROCs for each model . (6) Please provide a mean and standard deviation for the f1 score for each model.
# =============================================================================
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score 














