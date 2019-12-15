# =============================================================================
# =============================================================================
# =============================================================================
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











