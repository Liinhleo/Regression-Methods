import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#read dataset_train
dataset_train=pd.read_csv("BostonTrain.csv")
dataset_test=pd.read_csv("BostonTest.csv")

# Split dataset_train to input X and outcome Y
X_train = dataset_train.iloc[:,0:13].values
Y_train = dataset_train.iloc[:,-1].values

X_test = dataset_train.iloc[:,0:13].values
Y_test = dataset_train.iloc[:,-1].values

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
    print(y_pred)

for i in range(len(X_test)):
    compare(i)


