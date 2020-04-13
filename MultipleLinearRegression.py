import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read dataset_train
dataset_train=pd.read_csv("50_Startups_Train.csv")
dataset_test=pd.read_csv("50_Startups_Test.csv")

# Split dataset_train to input X and outcome Y
X_train=dataset_train.iloc[:,0:4].values
Y_train=dataset_train.iloc[:,-1].values

X_test=dataset_test.iloc[:,0:4].values
Y_test=dataset_test.iloc[:,-1].values


# Tien xu ly
ohe = OneHotEncoder(handle_unknown='ignore')
X_train_state=ohe.fit_transform(dataset_train[['State']]).toarray()
X_train=np.concatenate((X_train_state,X_train),axis=1)

ohe1 = OneHotEncoder(handle_unknown='ignore')
X_test_state=ohe1.fit_transform(dataset_test[['State']]).toarray()
X_test=np.concatenate((X_test_state,X_test),axis=1)

# Training model
lin_reg =LinearRegression()
lin_reg.fit(X_train,Y_train)

# Valuation score
print('score data_train=' , lin_reg.score(X_train,Y_train))
print('score data_test=' , lin_reg.score(X_test,Y_test))

def compare(i_example):
    x=X_test[i_example:i_example+1]
    y=Y_test[i_example]
    y_pred=lin_reg.predict(x)[0]
    state = ohe.inverse_transform(x[:,0:3])
    print(x[:,3:6],state,y,y_pred)

for i in range(len(X_test)):
    compare(i)


