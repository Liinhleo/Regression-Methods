import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

#read dataset
datatrain = pd.read_csv("kc_house_Train.csv")
datatest = pd.read_csv("kc_house_Test.csv",nrows=400)

# Preprocessing (['date'])
format = '%Y%m%dT000000'
# Train
df_train = pd.DataFrame(columns=["Days"])
for i in range(0,len(datatrain['date'].index)):
    date1 = datetime.strptime(str(datatrain.iloc[0][2]), format)
    date2 = datetime.strptime(str(datatrain.iloc[i][2]), format)
    df_train=df_train.append({'Days':(date2-date1).days}, ignore_index=True)

df_test = pd.DataFrame(columns=["Days"])
for i in range(0,len(datatest['date'].index)):
    date1 = datetime.strptime(str(datatest.iloc[0][2]), format)
    date2 = datetime.strptime(str(datatest.iloc[i][2]), format)
    df_test=df_test.append({'Days':(date2-date1).days}, ignore_index=True)

# Insert date_column to dataset
datatrain.insert(1, "Days", df_train, True)
datatest.insert(1, "Days", df_test, True)

# Split dataset to input X_train/X_test and outcome Y_train/Y_test
X_train = datatrain.drop(labels = ['Unnamed: 0','price','date'], axis = 1)
Y_train = datatrain['price']
X_test = datatest.drop(labels = ['Unnamed: 0','price','date'], axis = 1)
Y_test = datatest['price']

# Training model
lin_reg =LinearRegression()
lin_reg.fit(X_train,Y_train)

# Valuation score
print('score data_train=' , lin_reg.score(X_train,Y_train))
print('score data_test=' , lin_reg.score(X_test,Y_test))


# Evaluation
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
    
y = Y_test
y_pred=lin_reg.predict(X_test)
print("RMSE: ",sqrt(mean_squared_error(y, y_pred)))
print("MSE: ",mean_squared_error(y, y_pred))
print("MAE: ",mean_absolute_error(y, y_pred))



def compare(i_example):
    x=X_test[i_example:i_example+1]
    y=Y_test[i_example]
    y_pred=lin_reg.predict(x)[0]
    print(y,y_pred)


for i in range(len(X_test)):
    compare(i)



